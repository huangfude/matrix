//! Worker initialization and addition functionality
//!
//! Provides functionality to initialize workers from router configuration
//! and submit AddWorker jobs to the job queue.
use std::sync::Arc;
use tracing::{info, debug, warn};

use smg::config::{RouterConfig, RoutingMode};
use smg::app_context::AppContext;
use openai_protocol::{
    model_card::ModelCard,
    worker::{RuntimeType, WorkerModels, WorkerSpec, WorkerType},
};
use smg::core::Job;

use crate::utils::get_model_info;

/// Initialize workers from router configuration
///
/// This function creates AddWorker jobs based on the routing mode and submits them
/// to the job queue for asynchronous processing.
pub async fn initialize_workers_from_config(
    static_models: Option<Vec<String>>,
    router_config: &RouterConfig,
    context: &Arc<AppContext>,
) -> Result<String, String> {
    // Create iterator of (url, worker_type) tuples based on mode
    let api_key = router_config.api_key.clone();
    let mut worker_count = 0;

    // Create iterator of (url, worker_type, bootstrap_port) tuples based on mode
    let workers: Vec<(String, &str, Option<u16>)> = match &router_config.mode {
        RoutingMode::Regular { worker_urls } => worker_urls
            .iter()
            .map(|url| (url.clone(), "regular", None))
            .collect(),
        RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } => {
            let prefill_workers = prefill_urls
                .iter()
                .map(|(url, port)| (url.clone(), "prefill", *port));

            let decode_workers =
                decode_urls.iter().map(|url| (url.clone(), "decode", None));

            prefill_workers.chain(decode_workers).collect()
        }
        RoutingMode::OpenAI { worker_urls }
        | RoutingMode::Anthropic { worker_urls }
        | RoutingMode::Gemini { worker_urls } => {
            let provider_name = router_config.mode_type();
            return submit_external_worker_jobs(
                worker_urls,
                provider_name,
                router_config,
                context,
            )
            .await;
        }
    };

    debug!(
        "Creating AddWorker jobs for {} workers from config",
        workers.len()
    );

    // Process all workers with unified loop
    for (url, worker_type, bootstrap_port) in workers {
        let model_id = if let Some(ref models) = static_models {
            if !models.is_empty() {
                info!("Using model ID for worker at URL {}", url);
                // 正常范围内取对应值，超出则取最后一个
                let idx = worker_count.min(models.len() - 1);
                models.get(idx).cloned()
            } else {
                None // 如果 static_models 存在但为空，设置为 None
            }
        } else {
            None // 如果 static_models 不存在，设置为 None
        };
        
        // 如果 model_id 为 None，则通过 get_model_info 获取模型信息
        let model_id = if model_id.is_none() {
            match get_model_info(&url).await {
                Ok(model_info) => {
                    debug!("Successfully retrieved model info for URL {}: {}", url, model_info.id);
                    Some(model_info.id)
                }
                Err(e) => {
                    warn!("Failed to get model info for URL {}: {}, using None", url, e);
                    None
                }
            }
        } else {
            model_id
        };

        let url_for_error = url.clone(); // Clone for error message
        let proto_worker_type = match worker_type {
            "prefill" => WorkerType::Prefill,
            "decode" => WorkerType::Decode,
            _ => WorkerType::Regular,
        };
        let mut spec = WorkerSpec::new(url);
        spec.worker_type = proto_worker_type;
        spec.api_key.clone_from(&api_key);
        spec.bootstrap_port = bootstrap_port;

        let model = ModelCard::new(model_id.unwrap_or_default());
        spec.models = WorkerModels::from(vec![model]);
        // Health config is resolved at worker build time from router
        // defaults + per-worker overrides (spec.health). No need to
        // set spec.health here since these workers have no overrides.
        spec.max_connection_attempts =
            router_config.health_check.success_threshold.max(1) * 10;
        let config = spec;

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        if let Some(queue) = context.worker_job_queue.get() {
            queue.submit(job).await.map_err(|e| {
                format!(
                    "Failed to submit AddWorker job for {worker_type} worker {url_for_error}: {e}"
                )
            })?;
            worker_count += 1;
        } else {
            return Err("JobQueue not available".to_string());
        }
    }

    // 等待所有worker jobs完成
    if worker_count > 0 {
        info!("Waiting for {} worker initialization jobs to complete...", worker_count);
        let job_queue = context.worker_job_queue.get()
            .ok_or("JobQueue not available")?;
        
        let mut wait_count = 0;
        let max_wait_attempts = 5;
        
        // 收集所有worker URLs以便检查状态
        let worker_urls: Vec<String> = match &router_config.mode {
            RoutingMode::Regular { worker_urls } => worker_urls.clone(),
            RoutingMode::PrefillDecode { prefill_urls, decode_urls, .. } => {
                let mut urls = Vec::new();
                urls.extend(prefill_urls.iter().map(|(url, _)| url.clone()));
                urls.extend(decode_urls.clone());
                urls
            }
            _ => Vec::new(),
        };        
        
        loop {
            let mut completed_jobs = 0;
            let mut successful_jobs = 0;
            
            for url in &worker_urls {
                if let Some(status) = job_queue.get_status(url) {
                    match status.status.as_str() {
                        "failed" => {
                            completed_jobs += 1;
                        }
                        _ => {
                            // 仍在处理中 (pending, processing 等)
                        }
                    }
                } else {
                    // 状态映射中不存在该记录，表示任务已成功完成
                    debug!("Worker {} completed successfully (status removed)", url);
                    completed_jobs += 1;
                    successful_jobs += 1;
                }
            }
            
            if completed_jobs >= worker_count {
                info!(
                    "All {} worker initialization jobs completed ({} successful)",
                    worker_count, successful_jobs
                );
                break;
            }
            
            wait_count += 1;
            if wait_count >= max_wait_attempts {
                warn!("Max wait attempts reached. Exiting loop.");
                break;
            }
                        
            // 等待0.1秒
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            
        }
    }
    
    Ok(format!("Submitted {worker_count} AddWorker jobs"))
}

/// Submit AddWorker jobs for external provider endpoints (OpenAI/Anthropic/Gemini).
async fn submit_external_worker_jobs(
    worker_urls: &[String],
    provider_name: &str,
    router_config: &RouterConfig,
    context: &Arc<AppContext>,
) -> Result<String, String> {
    let api_key = router_config.api_key.clone();
    let mut submitted_count = 0;

    for url in worker_urls {
        let url_for_error = url.clone();
        let config = build_external_worker_config(url, api_key.clone(), router_config);

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        if let Some(queue) = context.worker_job_queue.get() {
            queue.submit(job).await.map_err(|e| {
                format!("Failed to submit AddWorker job for {provider_name} endpoint {url_for_error}: {e}")
            })?;
            submitted_count += 1;
        } else {
            return Err("JobQueue not available".to_string());
        }
    }

    if submitted_count == 0 {
        info!("{provider_name} mode: no worker URLs provided");
        return Ok(format!(
            "{provider_name} mode: no worker URLs to initialize"
        ));
    }

    Ok(format!(
        "Submitted {submitted_count} AddWorker jobs for {provider_name} endpoints"
    ))
}

/// Build a `WorkerSpec` for an external API endpoint (OpenAI/Anthropic mode).
fn build_external_worker_config(
    url: &str,
    api_key: Option<String>,
    router_config: &RouterConfig,
) -> WorkerSpec {
    let mut spec = WorkerSpec::new(url);
    spec.runtime_type = RuntimeType::External;
    spec.api_key = api_key;
    // Health config is resolved at worker build time from router
    // defaults + per-worker overrides (spec.health). No need to
    // set spec.health here since these workers have no overrides.
    spec.max_connection_attempts = router_config.health_check.success_threshold.max(1) * 10;
    spec
}