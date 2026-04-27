use std::sync::Arc;
use tracing::{info, warn};
use smg::routers::http::pd_router::PDRouter;

#[derive(Debug, Clone)]
pub struct PDRouterExt {
    pub pd_router: Arc<PDRouter>,
}

impl PDRouterExt {

    pub fn new(pd_router: PDRouter) -> Self {
        PDRouterExt {
            pd_router: Arc::new(pd_router),
        }
    }

    /// Start profiling on a backend server
    pub async fn start_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = super::dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/start_profile", base_url);
        match self.pd_router.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Started profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to start profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error starting profiling on {}: {}", base_url, e);
            }
        }
    }

    /// Stop profiling on a backend server
    pub async fn stop_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = super::dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/stop_profile", base_url);
        match self.pd_router.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Stopped profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to stop profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error stopping profiling on {}: {}", base_url, e);
            }
        }
    }

}