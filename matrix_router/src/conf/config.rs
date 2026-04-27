use serde::{Deserialize, Serialize};

use smg::config::PolicyConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixConfig {
    // 添加matrix router自定义的配置项
    pub routing_mode_ext: RoutingModeExt,
    pub static_models: Option<Vec<String>>,
    pub data_plane_api_keys: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type_ext")]
pub enum RoutingModeExt {
    #[serde(rename = "vllm_prefill_decode")]
    VllmPrefillDecode {
        /// Prefill worker URLs with optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        /// Decode worker URLs
        decode_urls: Vec<String>,
        /// Optional separate policy for prefill workers
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        /// Optional separate policy for decode workers
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
        /// ZMQ service discovery address (e.g., "0.0.0.0:30001")
        #[serde(skip_serializing_if = "Option::is_none")]
        discovery_address: Option<String>,
        // Profiling settings
        enable_profiling: bool,
        profile_timeout_secs: u64,
        // Data parallelism settings
        intra_node_data_parallel_size: usize,
    },
    #[serde(rename = "tbd")]
    // To Be Decided 占位
    TBD,
}