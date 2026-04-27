//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::{
    fmt,
    sync::Arc,
    collections::{HashMap, HashSet},
};

use dashmap::DashMap;

use smg::{
    core::{
        WorkerRegistry,
        Job, JobQueue, Worker,
    },
};


/// Health checker handle with graceful shutdown.
///
/// The checker sleeps until the next worker is due for a health check,
/// so it wakes only when there is actual work to do.
pub struct HealthChecker {
    handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_notify: Arc<tokio::sync::Notify>,
}

impl fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthChecker").finish()
    }
}

impl HealthChecker {
    pub fn new(
        handle: tokio::task::JoinHandle<()>,
        shutdown_notify: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            handle: Some(handle),
            shutdown_notify,
        }
    }

    /// Shutdown the health checker gracefully.
    /// Wakes the sleeping task immediately so it can exit cleanly.
    /// Prefer this over dropping when you can `.await` — it lets the
    /// current health-check iteration finish instead of aborting mid-flight.
    pub async fn shutdown(&mut self) {
        self.shutdown_notify.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for HealthChecker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

#[derive(Debug)]
pub struct WorkerRegistryExt{
    pub worker_registry: Arc<WorkerRegistry>,
}

impl WorkerRegistryExt {

    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry: worker_registry
        }
    }

    /// Start a deadline-driven health checker for all workers in the registry.
    ///
    /// Each worker is checked according to its own `health_config.check_interval_secs`.
    /// The task sleeps until the next worker is due, so it only wakes when there is
    /// actual work to do — zero CPU when idle, no polling.
    pub fn start_health_checker(
        &self,
        default_interval_secs: u64,
        remove_unhealthy: bool,
        job_queue: Option<Arc<JobQueue>>,
    ) -> HealthChecker {
        let shutdown_notify = Arc::new(tokio::sync::Notify::new());
        let shutdown_clone = shutdown_notify.clone();
        
        // 将 Vec<(WorkerId, Arc<dyn Worker>)> 转换为 Arc<DashMap<WorkerId, Arc<dyn Worker>>>
        let workers_vec = self.worker_registry.get_all_with_ids();
        let workers_map = Arc::new(DashMap::new());
        for (worker_id, worker) in workers_vec {
            workers_map.insert(worker_id, worker);
        }
        let workers_ref = workers_map;
        
        let job_queue = if remove_unhealthy { job_queue } else { None };

        #[expect(
            clippy::disallowed_methods,
            reason = "Health checker loop: runs for the lifetime of the registry, handle is stored in HealthChecker and abort() is called on drop"
        )]
        let handle = tokio::spawn(async move {
            // next_check[url] = Instant when the worker is next due for a health check.
            let mut next_check: HashMap<String, tokio::time::Instant> = HashMap::new();

            loop {
                let now = tokio::time::Instant::now();

                // Snapshot current workers from the registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Sync schedule with registry: add new workers, prune removed
                // and disabled ones so stale deadlines don't cause wakeups.
                let checkable_urls: HashSet<String> = workers
                    .iter()
                    .filter(|w| !w.metadata().health_config.disable_health_check)
                    .map(|w| w.url().to_string())
                    .collect();
                next_check.retain(|url, _| checkable_urls.contains(url));
                for url in &checkable_urls {
                    next_check.entry(url.clone()).or_insert(now);
                }

                // Collect workers whose deadline has passed
                let due_workers: Vec<_> = workers
                    .iter()
                    .filter(|w| !w.metadata().health_config.disable_health_check)
                    .filter(|w| {
                        next_check
                            .get(w.url())
                            .is_some_and(|deadline| now >= *deadline)
                    })
                    .cloned()
                    .collect();

                // Run due health checks in parallel and schedule the next deadline
                if !due_workers.is_empty() {
                    for worker in &due_workers {
                        let secs = worker.metadata().health_config.check_interval_secs;
                        let secs = if secs > 0 {
                            secs
                        } else {
                            default_interval_secs
                        };
                        next_check.insert(
                            worker.url().to_string(),
                            now + tokio::time::Duration::from_secs(secs),
                        );
                    }
                    let futs: Vec<_> = due_workers
                        .into_iter()
                        .map(|w| async move {
                            let failed = w.check_health_async().await.is_err();
                            (w, failed)
                        })
                        .collect();
                    let checked_workers = futures::future::join_all(futs).await;

                    // Only remove workers whose health check actually failed
                    // this tick. Workers that are unhealthy but passing checks
                    // (e.g. mesh-synced, pre-activation) are recovering — leave
                    // them alone until they either become healthy or truly fail.
                    if let Some(ref job_queue) = job_queue {
                        for (worker, failed) in &checked_workers {
                            if !worker.is_healthy() && *failed {
                                let url = worker.url().to_string();
                                tracing::warn!(
                                    worker_url = %url,
                                    "Removing unhealthy worker from registry"
                                );
                                next_check.remove(&url);
                                if let Err(e) = job_queue
                                    .submit(Job::RemoveWorker { url: url.clone() })
                                    .await
                                {
                                    tracing::error!(
                                        worker_url = %url,
                                        error = %e,
                                        "Failed to submit worker removal job"
                                    );
                                }
                            }
                        }
                    }
                }

                // Sleep until the earliest deadline or until shutdown is signalled.
                // If the registry is empty, sleep for the default interval then re-scan
                // (new workers may have been added).
                let sleep_until = next_check.values().min().copied().unwrap_or_else(|| {
                    now + tokio::time::Duration::from_secs(default_interval_secs)
                });

                tokio::select! {
                    () = tokio::time::sleep_until(sleep_until) => {}
                    () = shutdown_clone.notified() => {
                        tracing::debug!("Registry health checker shutting down");
                        break;
                    }
                }
            }
        });

        HealthChecker::new(handle, shutdown_notify)
    }

}

