// vLLM PD (Prefill-Decode) Router Implementation
// This module extends PDRouter to handle vLLM-specific two-stage processing
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{self, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    completion::CompletionRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
};

use smg::{
    app_context::AppContext,
    core::{BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType, HashRing, WorkerLoadGuard, AttachedBody},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        RouterTrait,
        http::{
            pd_router::PDRouter,
            pd_types::PDRouterError,
        },
    },
};

use super::{
    dp_utils, logprobs_merge,
    pd_router_ext::PDRouterExt,
    vllm_service_discovery::{ServiceRegistry, ServiceType},
};

use crate::utils::model_utils;

/// vLLM PD Router that extends PDRouter with vLLM-specific request handling
#[derive(Debug)]
pub struct VllmPDRouter {
    /// Underlying PD router for most functionality
    pd_router_ext: PDRouterExt,
    /// Service discovery registry for dynamic ZMQ address resolution
    service_registry: Arc<ServiceRegistry>,
    /// HTTP client for making requests to discovered services
    http_client: reqwest::Client,
    /// Policy registry for load balancing
    policy_registry: Arc<PolicyRegistry>,
    /// Whether this router uses service discovery (true) or direct URLs (false)
    use_discovery: bool,
    /// Enable profiling calls to vLLM workers
    enable_profiling: bool,
    /// Profiling timeout in seconds
    profile_timeout_secs: u64,
    /// Active profiling timeout tasks keyed by worker URL
    profiling_tasks: Arc<Mutex<HashMap<String, tokio::task::AbortHandle>>>,
    /// Intra-node data parallel size for DP-aware routing (automatically enabled when > 1)
    intra_node_data_parallel_size: usize,
}

// vLLM PD router config
#[derive(Clone)]
pub struct VllmPDConfig {
    pub enable_profiling: bool,
    pub profile_timeout_secs: u64,
    pub intra_node_data_parallel_size: usize,
}

impl VllmPDRouter {
    /// Generate vLLM-specific request ID with prefill/decode addressing
    fn generate_vllm_request_id(prefill_addr: &str, decode_addr: &str) -> String {
        let uuid = Uuid::new_v4().to_string().replace('-', "");
        format!(
            "___prefill_addr_{}___decode_addr_{}_{}",
            prefill_addr, decode_addr, uuid
        )
    }

    /// Get ZMQ address for a worker URL using service discovery
    fn get_zmq_address(&self, http_url: &str, service_type: ServiceType) -> String {
        // Extract just the host:port from the URL
        let http_address = http_url.replace("http://", "").replace("https://", "");

        // Try to get ZMQ address from service discovery
        if let Some(zmq_addr) = self
            .service_registry
            .get_zmq_address(&http_address, service_type.clone())
        {
            debug!(
                "Using discovered ZMQ address: {} ({:?}) -> {}",
                http_address, service_type, zmq_addr
            );
            return zmq_addr;
        }

        // Fallback: use HTTP address as ZMQ address
        debug!(
            "No ZMQ discovery result for {} ({:?}), using fallback: {}",
            http_address, service_type, http_address
        );
        http_address
    }

    /// Helper: Start profiling on a backend server with timeout
    async fn start_profiling(&self, worker_url: &str) {
        // Only profile if enabled
        if !self.enable_profiling {
            return;
        }

        // Start profiling on the worker
        self.pd_router_ext.start_profiling(worker_url).await;

        // Spawn a timeout task that will call stop_profiling if timeout is reached
        let timeout_secs = self.profile_timeout_secs;
        let worker_url_owned = worker_url.to_string();
        let pd_router_clone = self.pd_router_ext.clone();
        let profiling_tasks_clone = self.profiling_tasks.clone();

        let task_handle = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(timeout_secs)).await;

            info!(
                "Profiling timeout reached for {}, stopping profiling",
                worker_url_owned
            );
            pd_router_clone.stop_profiling(&worker_url_owned).await;

            // Remove ourselves from the tasks map
            let mut tasks = profiling_tasks_clone.lock().await;
            tasks.remove(&worker_url_owned);
        });

        // Store the abort handle
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(old_handle) = tasks.insert(worker_url.to_string(), task_handle.abort_handle()) {
            // Cancel any existing timeout task for this worker
            old_handle.abort();
        }
    }

    /// Helper: Stop profiling on a backend server and cancel timeout task
    async fn stop_profiling(&self, worker_url: &str) {
        // Only stop profiling if it was enabled
        if !self.enable_profiling {
            return;
        }

        // Cancel the timeout task if it exists
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(handle) = tasks.remove(worker_url) {
            handle.abort();
            info!("Cancelled profiling timeout task for {}", worker_url);
        }

        // Stop profiling on the worker
        self.pd_router_ext.stop_profiling(worker_url).await;
    }

    /// Modify request for prefill stage (set max_tokens=1)
    /// - For inference/v1/generate: patch sampling_params.max_tokens and sampling_params.min_tokens
    /// - For other endpoints (fallback): patch top-level max_tokens, max_completion_tokens, min_tokens
    ///
    /// stream=false and stream_options removal are always applied at top level.
    fn prepare_prefill_request(mut request: Value, path: &str) -> Value {
        if path.contains("inference/v1/generate") {
            // Generate API: max_tokens and min_tokens are in sampling_params
            if let Some(sampling_params) = request.get_mut("sampling_params") {
                sampling_params["max_tokens"] = json!(1);
                // Also adjust min_tokens to ensure min_tokens <= max_tokens
                // This is required because vLLM validates that min_tokens <= max_tokens
                if let Some(min_tokens) = sampling_params.get("min_tokens").and_then(|v| v.as_u64())
                {
                    if min_tokens > 1 {
                        sampling_params["min_tokens"] = json!(1);
                    }
                }
            } else {
                // Create sampling_params with prefill defaults when missing
                request["sampling_params"] = json!({"max_tokens": 1, "min_tokens": 1});
            }
        } else {
            // Fallback: OpenAI-style endpoints (chat/completions)
            request["max_tokens"] = json!(1);
            if request.get("max_completion_tokens").is_some() {
                request["max_completion_tokens"] = json!(1);
            }
            // Also adjust min_tokens to ensure min_tokens <= max_tokens
            // This is required because vLLM validates that min_tokens <= max_tokens
            if let Some(min_tokens) = request.get("min_tokens").and_then(|v| v.as_u64()) {
                if min_tokens > 1 {
                    request["min_tokens"] = json!(1);
                }
            }
        }
        // Force non-streaming for prefill to get JSON response with kv_transfer_params
        request["stream"] = json!(false);
        // Remove stream_options since we're setting stream=false
        if let Some(obj) = request.as_object_mut() {
            obj.remove("stream_options");
        }
        request
    }

    /// Convert service discovery instances to Worker objects for policy selection
    fn instances_to_workers(instances: &[(String, String)]) -> Vec<Arc<dyn Worker>> {
        instances
            .iter()
            .map(|(http_addr, _zmq_addr)| {
                let full_url =
                    if http_addr.starts_with("http://") || http_addr.starts_with("https://") {
                        http_addr.clone()
                    } else {
                        format!("http://{}", http_addr)
                    };
                Arc::new(
                    BasicWorkerBuilder::new(full_url)
                    .worker_type(WorkerType::Regular)
                    .build()
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Select worker using policy-based load balancing
    fn select_worker_with_policy(
        &self,
        instances: &[(String, String)],
        is_prefill: bool,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
    ) -> Option<usize> {
        if instances.is_empty() {
            return None;
        }

        // Convert instances to workers for policy selection
        let workers = Self::instances_to_workers(instances);

        // Get the appropriate policy
        let policy = if is_prefill {
            self.policy_registry.get_prefill_policy()
        } else {
            self.policy_registry.get_decode_policy()
        };

        // Use policy to select worker
        policy.select_worker(
            &workers, 
            &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                })
    }

    /// Process vLLM request using pure service discovery
    async fn process_vllm_request(
        &self,
        request_json: Value,
        path: &str,
        headers: Option<&HeaderMap>,
        model_id: &str,
    ) -> Response {
        debug!("Processing vLLM request for path: {}", path);
        debug!(
            "Request JSON: {}",
            serde_json::to_string_pretty(&request_json).unwrap_or_default()
        );

        // Get available instances from service discovery
        let prefill_instances = self.service_registry.get_prefill_instances();
        let decode_instances = self.service_registry.get_decode_instances();

        debug!(
            "Found {} prefill instances, {} decode instances from service discovery",
            prefill_instances.len(),
            decode_instances.len()
        );

        if prefill_instances.is_empty() || decode_instances.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No workers available via service discovery: {} prefill, {} decode",
                    prefill_instances.len(),
                    decode_instances.len()
                ),
            )
                .into_response();
        }

        // Use policy-based load balancing to select prefill and decode workers
        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();

        // Get cached hash ring for consistent hashing
        let hash_ring = self.pd_router_ext.pd_router.worker_registry
            .get_hash_ring(model_id);

        let prefill_idx =
            match self.select_worker_with_policy(&prefill_instances, true, 
                request_str, headers, hash_ring.clone()) {
                Some(idx) => idx,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

        let decode_idx = match self.select_worker_with_policy(&decode_instances, false, 
            request_str, headers, hash_ring)
        {
            Some(idx) => idx,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Decode policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let (prefill_http, prefill_zmq) = &prefill_instances[prefill_idx];
        let (decode_http, decode_zmq) = &decode_instances[decode_idx];

        let prefill_policy_name = self.policy_registry.get_prefill_policy().name();
        let decode_policy_name = self.policy_registry.get_decode_policy().name();

        debug!(
            "vLLM policy-based routing: prefill={}({}) [policy:{}], decode={}({}) [policy:{}]",
            prefill_http,
            prefill_zmq,
            prefill_policy_name,
            decode_http,
            decode_zmq,
            decode_policy_name
        );

        // Process two-stage vLLM request with discovered endpoints
        match self
            .process_vllm_two_stage_request_discovered(
                request_json,
                &prefill_instances[prefill_idx],
                &decode_instances[decode_idx],
                path,
                headers,
            )
            .await
        {
            Ok(response) => {
                debug!("Two-stage processing completed successfully");
                response
            }
            Err(e) => {
                error!("Two-stage processing failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request processing failed: {}", e),
                )
                    .into_response()
            }
        }
    }

    /// Two-stage request processing for vLLM disaggregated mode using discovered endpoints
    async fn process_vllm_two_stage_request_discovered(
        &self,
        request_json: Value,
        prefill_instance: &(String, String),
        decode_instance: &(String, String),
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Response, String> {
        let (prefill_http, prefill_zmq) = prefill_instance;
        let (decode_http, decode_zmq) = decode_instance;

        debug!("ENTERED process_vllm_two_stage_request_discovered method");
        let start_time = Instant::now();
        debug!(
            "Prefill: HTTP={}, ZMQ={}, Decode: HTTP={}, ZMQ={}, Path: {}",
            prefill_http, prefill_zmq, decode_http, decode_zmq, path
        );

        let request_id = Self::generate_vllm_request_id(prefill_zmq, decode_zmq);
        debug!(
            "Generated vLLM request ID for P2P coordination: {}",
            request_id
        );

        // DO NOT add P2P metadata to internal request_id - let vLLM generate clean internal IDs
        // The P2P metadata will be sent in X-Request-Id header instead

        // Prepare prefill request (max_tokens=1 to force prefill-only mode)
        let mut prefill_request = Self::prepare_prefill_request(request_json.clone(), path);

        // Add kv_transfer_params for NixlConnector support at top level
        // This enables the prefill instance to prepare for remote decode
        prefill_request["kv_transfer_params"] = json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": Value::Null,
            "remote_block_ids": Value::Null,
            "remote_host": Value::Null,
            "remote_port": Value::Null
        });

        debug!("Added kv_transfer_params to prefill request for NixlConnector support");

        let prefill_request_str = serde_json::to_string(&prefill_request)
            .map_err(|e| format!("Failed to serialize prefill request: {}", e))?;

        // Stage 1: Send to prefill server with max_tokens=1 and P2P coordination header
        debug!(
            "Stage 1: Sending prefill-only request (max_tokens=1) to prefill server at http://{}",
            prefill_http
        );

        // Extract dp_rank from prefill_http if intra_node_data_parallel_size > 1
        let (prefill_base_http, prefill_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            let prefill_url = format!("http://{}", prefill_http);
            let (base, rank) = dp_utils::parse_worker_url(&prefill_url);
            let base_http = base.replace("http://", "").replace("https://", "");
            (base_http, rank)
        } else {
            (prefill_http.to_string(), None)
        };

        // Start profiling on prefill server
        self.start_profiling(&format!("http://{}", prefill_base_http))
            .await;

        let mut prefill_request_builder = self
            .http_client
            .post(format!("http://{}{}", prefill_base_http, path))
            .header("Content-Type", "application/json")
            .header("X-Request-Id", &request_id); // P2P coordination metadata in header

        // Propagate trace headers and add X-data-parallel-rank header using shared utilities
        prefill_request_builder =
            model_utils::propagate_trace_headers(prefill_request_builder, headers);
        prefill_request_builder =
            dp_utils::add_dp_rank_header(prefill_request_builder, prefill_dp_rank);
        if let Some(rank) = prefill_dp_rank {
            debug!(
                "Added X-data-parallel-rank={} header to prefill request",
                rank
            );
        }

        let prefill_response = match prefill_request_builder
            .body(prefill_request_str)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let _duration = start_time.elapsed();
                return Err(format!(
                    "Prefill request failed to {}: {}",
                    prefill_http, e
                ));
            }
        };

        let prefill_status = prefill_response.status();
        debug!("Prefill server responded with status: {}", prefill_status);

        if !prefill_status.is_success() {
            let _duration = start_time.elapsed();
            let error_body = prefill_response.text().await.unwrap_or_default();
            return Err(format!(
                "Prefill server error {}: {}",
                prefill_status, error_body
            ));
        }

        // Extract kv_transfer_params from prefill response
        let prefill_response_text = prefill_response.text().await.map_err(|e| {
            format!(
                "Failed to read prefill response from {}: {}",
                prefill_http, e
            )
        })?;

        debug!("Prefill response body: {}", prefill_response_text);

        let prefill_response_json: Value = serde_json::from_str(&prefill_response_text)
            .map_err(|e| format!("Failed to parse prefill response as JSON: {}", e))?;

        // Extract kv_transfer_params from prefill response if present
        let kv_transfer_params = prefill_response_json.get("kv_transfer_params").cloned();

        if let Some(ref params) = kv_transfer_params {
            debug!(
                "Extracted kv_transfer_params from prefill response: {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            );
        } else {
            debug!("No kv_transfer_params found in prefill response, will proceed without them");
        }

        // Prepare decode request with kv_transfer_params from prefill response at top level
        let mut decode_request = request_json.clone();
        if let Some(params) = kv_transfer_params {
            decode_request["kv_transfer_params"] = params;
            debug!("Added kv_transfer_params to decode request");
        }

        let decode_request_str = serde_json::to_string(&decode_request)
            .map_err(|e| format!("Failed to serialize decode request: {}", e))?;

        // Stop profiling on prefill server after its work is done
        self.stop_profiling(&format!("http://{}", prefill_base_http))
            .await;

        // Stage 2: Send to decode server with original request and same P2P coordination header
        debug!(
            "Stage 2: Sending original request to decode server at http://{}",
            decode_http
        );

        // Extract dp_rank from decode_http if intra_node_data_parallel_size > 1
        let (decode_base_http, decode_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            let decode_url = format!("http://{}", decode_http);
            let (base, rank) = dp_utils::parse_worker_url(&decode_url);
            let base_http = base.replace("http://", "").replace("https://", "");
            (base_http, rank)
        } else {
            (decode_http.to_string(), None)
        };

        // Start profiling on decode server
        self.start_profiling(&format!("http://{}", decode_base_http))
            .await;

        let mut decode_request_builder = self
            .http_client
            .post(format!("http://{}{}", decode_base_http, path))
            .header("Content-Type", "application/json")
            .header("X-Request-Id", &request_id); // Same P2P coordination metadata in header

        // Propagate trace headers and add X-data-parallel-rank header using shared utilities
        decode_request_builder =
            model_utils::propagate_trace_headers(decode_request_builder, headers);
        decode_request_builder =
            dp_utils::add_dp_rank_header(decode_request_builder, decode_dp_rank);
        if let Some(rank) = decode_dp_rank {
            debug!(
                "Added X-data-parallel-rank={} header to decode request",
                rank
            );
        }

        let decode_response = match decode_request_builder.body(decode_request_str).send().await {
            Ok(resp) => resp,
            Err(e) => {
                let _duration = start_time.elapsed();
                return Err(format!(
                    "Decode request failed to {}: {}",
                    decode_http, e
                ));
            }
        };

        debug!(
            "Decode server responded with status: {}",
            decode_response.status()
        );

        // Stop profiling on decode server after response received
        self.stop_profiling(&format!("http://{}", decode_base_http))
            .await;

        // Record PD metrics
        let _duration = start_time.elapsed();

        // Check if logprobs merging is needed
        let needs_logprobs = request_json.get("logprobs").is_some()
            || request_json
                .get("echo")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let is_streaming = request_json
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If logprobs requested and non-streaming, merge prefill and decode logprobs
        if needs_logprobs && !is_streaming {
            debug!("Logprobs requested and non-streaming - merging prefill and decode logprobs");

            let status = decode_response.status();
            let headers = decode_response.headers().clone();
            let decode_body = decode_response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read decode response: {}", e))?;

            // Parse decode response as JSON
            let mut decode_json: Value = serde_json::from_slice(&decode_body)
                .map_err(|e| format!("Failed to parse decode response as JSON: {}", e))?;

            // Merge logprobs from prefill into decode response
            let merged =
                logprobs_merge::merge_logprobs_in_json(&prefill_response_json, &mut decode_json);
            if merged {
                debug!("Successfully merged logprobs from prefill and decode responses");
            } else {
                warn!("No logprobs were merged (might be expected if logprobs not in response)");
            }

            // Serialize merged response
            let merged_body = serde_json::to_vec(&decode_json)
                .map_err(|e| format!("Failed to serialize merged response: {}", e))?;

            let mut response_builder = Response::builder().status(status);
            for (name, value) in headers.iter() {
                response_builder = response_builder.header(name, value);
            }

            let response = response_builder
                .body(Body::from(merged_body))
                .map_err(|e| format!("Failed to build response: {}", e))?;

            Ok(response)
        } else {
            // No logprobs merging needed - return decode response as-is
            debug!(
                "No logprobs merging needed (streaming={}, needs_logprobs={})",
                is_streaming, needs_logprobs
            );

            let status = decode_response.status();
            let headers = decode_response.headers().clone();
            let body = decode_response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read decode response: {}", e))?;

            let mut response_builder = Response::builder().status(status);
            for (name, value) in headers.iter() {
                response_builder = response_builder.header(name, value);
            }

            let response = response_builder
                .body(Body::from(body))
                .map_err(|e| format!("Failed to build response: {}", e))?;

            Ok(response)
        }
    }

    /// Two-stage request processing for vLLM disaggregated mode
    ///
    /// This function handles fine-grained load tracking: the prefill worker's load is only
    /// incremented during the prefill phase, and the decode worker's load is only incremented
    /// during the decode phase. This accurately reflects the sequential nature of PD disaggregation.
    ///
    /// For streaming requests: returns first token after prefill, then asynchronously processes
    /// decode request and streams the remaining tokens.
    async fn process_vllm_two_stage_request(
        &self,
        original_request: Value,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
        path: &str,
        headers: Option<&HeaderMap>,
    ) -> Result<Response, PDRouterError> {
        debug!("ENTERED process_vllm_two_stage_request method");
        let start_time = Instant::now();
        debug!(
            "Prefill worker: {}, Decode worker: {}, Path: {}",
            prefill_worker.url(),
            decode_worker.url(),
            path
        );

        // Check if streaming is requested
        let is_streaming = original_request
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Increment prefill load at the start of the prefill phase
        prefill_worker.increment_load();

        let prefill_zmq_addr = self.get_zmq_address(prefill_worker.url(), ServiceType::Prefill);
        let decode_zmq_addr = self.get_zmq_address(decode_worker.url(), ServiceType::Decode);
        let request_id = Self::generate_vllm_request_id(&prefill_zmq_addr, &decode_zmq_addr);

        debug!("Generated vLLM request ID: {}", request_id);
        debug!("🔍 vLLM Proxy Comparison:");
        debug!("  📋 vLLM Proxy Request ID format: ___prefill_addr_{{zmq_addr}}___decode_addr_{{zmq_addr}}_{{uuid}}");
        debug!("  📋 Our Request ID format: ___prefill_addr_{{http_addr}}___decode_addr_{{http_addr}}_{{uuid}}");
        debug!("  📋 vLLM Proxy headers: Authorization: Bearer $OPENAI_API_KEY, X-Request-Id: {{request_id}}");
        debug!(
            "  📋 Our headers: Authorization: Bearer $OPENAI_API_KEY, X-Request-Id: {{request_id}}"
        );

        // Stage 1: Prepare prefill request with max_tokens=1 and kv_transfer_params
        let mut prefill_request = Self::prepare_prefill_request(original_request.clone(), path);

        // Add kv_transfer_params for NixlConnector support at top level
        // This enables the prefill instance to prepare for remote decode
        prefill_request["kv_transfer_params"] = json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": Value::Null,
            "remote_block_ids": Value::Null,
            "remote_host": Value::Null,
            "remote_port": Value::Null
        });

        debug!("Added kv_transfer_params to prefill request for NixlConnector support");

        // Use endpoint_url() to get the base URL without @rank suffix,
        // avoiding IPv6+DP URL corruption (same fix as Router and PDRouter)
        let prefill_base_url = prefill_worker.base_url().to_string();
        let prefill_dp_rank = prefill_worker.dp_rank();
        let prefill_url = prefill_worker.endpoint_url(path);

        debug!(
            "🚀 vLLM Stage 1 - Prefill: {} with request_id: {}",
            prefill_url, request_id
        );
        if let Some(rank) = prefill_dp_rank {
            debug!("📤 Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "📤 Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "📤 Prefill request payload: {}",
            serde_json::to_string_pretty(&prefill_request).unwrap_or_default()
        );

        // Start profiling on prefill server
        self.start_profiling(&prefill_base_url).await;

        let mut prefill_request_builder = self
            .pd_router_ext.pd_router
            .client
            .post(&prefill_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Propagate trace headers and add X-data-parallel-rank header using shared utilities
        prefill_request_builder =
            model_utils::propagate_trace_headers(prefill_request_builder, headers);
        prefill_request_builder =
            dp_utils::add_dp_rank_header(prefill_request_builder, prefill_dp_rank);

        let prefill_response = match prefill_request_builder.json(&prefill_request).send().await {
            Ok(resp) => resp,
            Err(e) => {
                prefill_worker.decrement_load();
                let _duration = start_time.elapsed();
                return Err(PDRouterError::NetworkError {
                    message: format!("Prefill request failed to {}: {}", prefill_url, e),
                });
            }
        };

        debug!("📥 Prefill response status: {}", prefill_response.status());
        debug!(
            "📥 Prefill response headers: {:?}",
            prefill_response.headers()
        );

        // Extract prefill response body to get kv_transfer_params
        let prefill_bytes = match prefill_response.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                prefill_worker.decrement_load();
                let _duration = start_time.elapsed();
                return Err(PDRouterError::NetworkError {
                    message: format!(
                        "Failed to read prefill response from {}: {}",
                        prefill_url, e
                    ),
                });
            }
        };

        debug!(
            "📥 Prefill response body size: {} bytes",
            prefill_bytes.len()
        );
        if prefill_bytes.len() < 1024 {
            debug!(
                "📥 Prefill response body content: {}",
                String::from_utf8_lossy(&prefill_bytes)
            );
        }

        // Parse prefill response to extract kv_transfer_params
        let prefill_response_json: Value = match serde_json::from_slice(&prefill_bytes) {
            Ok(json) => json,
            Err(e) => {
                prefill_worker.decrement_load();
                let _duration = start_time.elapsed();
                return Err(PDRouterError::NetworkError {
                    message: format!("Failed to parse prefill response as JSON: {}", e),
                });
            }
        };

        // Extract kv_transfer_params from prefill response if present
        let kv_transfer_params = prefill_response_json.get("kv_transfer_params").cloned();

        if let Some(ref params) = kv_transfer_params {
            debug!(
                "Extracted kv_transfer_params from prefill response: {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            );
        } else {
            debug!("No kv_transfer_params found in prefill response, will proceed without them");
        }

        // Stop profiling on prefill server after its work is done
        self.stop_profiling(&prefill_base_url).await;

        // Prefill phase complete: decrement prefill load, increment decode load
        prefill_worker.decrement_load();
        decode_worker.increment_load();

        debug!("✅ vLLM Stage 1 completed, starting Stage 2 - Decode");

        // Prepare decode request with kv_transfer_params from prefill response at top level
        let mut decode_request = original_request.clone();
        if let Some(params) = kv_transfer_params {
            decode_request["kv_transfer_params"] = params;
            debug!("Added kv_transfer_params to decode request");
        }

        // Use endpoint_url() to get the base URL without @rank suffix,
        // avoiding IPv6+DP URL corruption (same fix as Router and PDRouter)
        let decode_base_url = decode_worker.base_url().to_string();
        let decode_dp_rank = decode_worker.dp_rank();
        let decode_url = decode_worker.endpoint_url(path);

        debug!(
            "🚀 vLLM Stage 2 - Decode: {} with request_id: {}",
            decode_url, request_id
        );
        if let Some(rank) = decode_dp_rank {
            debug!("📤 Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "📤 Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "📤 Decode request payload: {}",
            serde_json::to_string_pretty(&decode_request).unwrap_or_default()
        );

        // Start profiling on decode server
        self.start_profiling(&decode_base_url).await;

        let mut decode_request_builder = self
            .pd_router_ext.pd_router
            .client
            .post(&decode_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Propagate trace headers and add X-data-parallel-rank header using shared utilities
        decode_request_builder =
            model_utils::propagate_trace_headers(decode_request_builder, headers);
        decode_request_builder =
            dp_utils::add_dp_rank_header(decode_request_builder, decode_dp_rank);

        // Handle streaming and non-streaming differently
        if is_streaming {
            // Streaming mode: return first token immediately, then async stream decode response
            debug!("Streaming mode: will return first token and async stream decode response");

            // Extract prefill content for first token
            // Optimization: avoid JSON clone by deserializing from JSON string directly
            let prefill_response_str = prefill_response_json.to_string();
            let prefill_chat_completion_resp: ChatCompletionResponse =
                serde_json::from_str(&prefill_response_str)
                .map_err(|e| PDRouterError::NetworkError {
                        message: format!("Failed to build response: {}", e),
                    })?;

            // Optimization: single pass to collect both content and reasoning_content
            let (prefill_contents, prefill_reasoning_contents): (Vec<String>, Vec<String>) =
                prefill_chat_completion_resp.choices
                    .iter()
                    .map(|c| (
                        c.message.content.as_ref().cloned().unwrap_or_default(),
                        c.message.reasoning_content.as_ref().cloned().unwrap_or_default()
                    ))
                    .unzip();

            // Optimization: avoid join if single choice (common case)
            let content_str = if prefill_contents.len() == 1 {
                prefill_contents[0].clone()
            } else {
                prefill_contents.concat()
            };

            let reasoning_content: Value = if prefill_reasoning_contents.is_empty() {
                    json!(null)
                } else if prefill_reasoning_contents.len() == 1 {
                    json!(prefill_reasoning_contents[0].clone())
                } else {
                    json!(prefill_reasoning_contents.concat())
                };

            let first_payload = json!({
                "id": prefill_chat_completion_resp.id,
                "object": "chat.completion.chunk",
                "created": prefill_chat_completion_resp.created,
                "model": prefill_chat_completion_resp.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content_str,
                        "reasoning_content": reasoning_content,
                    },
                    "logprobs": Value::Null,
                    "finish_reason": Value::Null,
                }],
                "prompt_token_ids": Value::Null,
            });

            let first_token = format!(
                "data: {}\n\n",
                serde_json::to_string(&first_payload).unwrap_or_default()
            );


            let mut x_headers = HeaderMap::new();
            x_headers.insert("X-Request-Id", http::HeaderValue::from_str(&request_id).unwrap());


            let (tx, rx) = tokio::sync::mpsc::channel(256);
            // Send first token immediately
            let _ = tx.send(Ok(axum::body::Bytes::from(first_token))).await;

            // Spawn async task to handle decode request and stream remaining tokens
            // Workers will be decremented in the spawned task when decode completes
            tokio::spawn(async move {

                let decode_result = decode_request_builder.json(&decode_request).send().await;

                // Stream decode response
                match decode_result {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                        debug!("Decode response status: {}", status);

                        let byte_stream = res.bytes_stream();
                        futures_util::pin_mut!(byte_stream);

                        // Optimization: use skip() to cleanly skip first N chunks
                        let mut remaining_stream = byte_stream.skip(3);
                        while let Some(chunk_result) = remaining_stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    let is_done = chunk.as_ref().starts_with(b"data: [DONE]");

                                    if tx.send(Ok(chunk)).await.is_err() {
                                        break;
                                    }

                                    if is_done {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(format!("Stream error: {}", e))).await;
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Decode request error: {}", e);
                    }
                }
            });

            let stream = ReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = StatusCode::OK;
            x_headers.insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("text/event-stream"));
            *response.headers_mut() = x_headers;

            // Wrap response with guards for proper load management
            // Guards will handle decrementing load when dropped after response is sent
            let response = AttachedBody::wrap_response(response, vec![
                WorkerLoadGuard::new(prefill_worker, headers),
                WorkerLoadGuard::new(decode_worker, headers),
            ]);

            Ok(response)
        } else {
            // Non-streaming mode: wait for decode response and return complete response
            debug!("Non-streaming mode: will wait for complete decode response");

            let decode_response = match decode_request_builder.json(&decode_request).send().await {
                Ok(resp) => resp,
                Err(e) => {
                    decode_worker.decrement_load();
                    let _duration = start_time.elapsed();
                    return Err(PDRouterError::NetworkError {
                        message: format!("Decode request failed to {}: {}", decode_url, e),
                    });
                }
            };

            // Stop profiling on decode server after response received
            self.stop_profiling(&decode_base_url).await;

            // Decode phase complete: decrement decode load
            decode_worker.decrement_load();

            let status = decode_response.status();
            let headers = decode_response.headers().clone();

            info!("📥 Decode response status: {}", status);
            info!("📥 Decode response headers: {:?}", headers);

            // Record PD metrics
            let _duration = start_time.elapsed();

            // Check if logprobs merging is needed
            let needs_logprobs = original_request.get("logprobs").is_some()
                || original_request
                    .get("echo")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

            // If logprobs requested and non-streaming, merge prefill and decode logprobs
            if needs_logprobs {
                debug!("Logprobs requested - merging prefill and decode logprobs");

                // Read decode response body
                let decode_body =
                    decode_response
                        .bytes()
                        .await
                        .map_err(|e| PDRouterError::NetworkError {
                            message: format!(
                                "Failed to read decode response from {}: {}",
                                decode_url, e
                            ),
                        })?;

                // Parse decode response as JSON
                let mut decode_json: Value =
                    serde_json::from_slice(&decode_body).map_err(|e| PDRouterError::NetworkError {
                        message: format!("Failed to parse decode response as JSON: {}", e),
                    })?;

                // Merge logprobs from prefill into decode response
                let merged =
                    logprobs_merge::merge_logprobs_in_json(&prefill_response_json, &mut decode_json);
                if merged {
                    debug!("Successfully merged logprobs from prefill and decode responses");
                } else {
                    warn!("No logprobs were merged (might be expected if logprobs not in response)");
                }

                // Serialize merged response
                let merged_body =
                    serde_json::to_vec(&decode_json).map_err(|e| PDRouterError::NetworkError {
                        message: format!("Failed to serialize merged response: {}", e),
                    })?;

                let mut response_builder = Response::builder().status(status);
                for (key, value) in headers.iter() {
                    if key != "transfer-encoding" && key != "content-length" {
                        response_builder = response_builder.header(key, value);
                    }
                }

                response_builder.body(Body::from(merged_body)).map_err(|e| {
                    PDRouterError::NetworkError {
                        message: format!("Failed to build response from {}: {}", decode_url, e),
                    }
                })
            } else {
                // No logprobs merging needed - return decode response as-is
                debug!("No logprobs merging needed");

                let mut response_builder = Response::builder().status(status);
                for (key, value) in headers.iter() {
                    if key != "transfer-encoding" && key != "content-length" {
                        response_builder = response_builder.header(key, value);
                    }
                }

                let body = Body::from_stream(decode_response.bytes_stream());
                response_builder
                    .body(body)
                    .map_err(|e| PDRouterError::NetworkError {
                        message: format!("Failed to build response from {}: {}", decode_url, e),
                    })
            }
        }
    }

    /// Create a new vLLM PD router
    /// Supports two modes:
    /// 1. Discovery mode: discovery_address is Some, prefill_urls and decode_urls are empty
    /// 2. Direct URL mode: discovery_address is None, prefill_urls and decode_urls are provided
    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        discovery_address: Option<String>,
        vllm_pd_config: VllmPDConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Self, String> {
        if let Some(ref addr) = discovery_address {
            // Discovery mode
            info!(
                "VllmPDRouter::new called in discovery mode with address: {}",
                addr
            );

            // Create underlying PD router with empty worker lists (they'll be discovered dynamically)
            let pd_router = PDRouter::new(ctx).await?;
            let pd_router_ext = PDRouterExt::new(pd_router);

            // Initialize service discovery
            let mut service_registry = ServiceRegistry::new();

            info!("Starting vLLM service discovery on {}", addr);
            service_registry
                .start_listener(addr)
                .await
                .map_err(|e| format!("Failed to start service discovery: {}", e))?;

            info!("VllmPDRouter created successfully with pure service discovery");

            Ok(Self {
                pd_router_ext,
                service_registry: Arc::new(service_registry),
                http_client: reqwest::Client::new(),
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: true,
                enable_profiling: vllm_pd_config.enable_profiling,
                profile_timeout_secs: vllm_pd_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: vllm_pd_config.intra_node_data_parallel_size,
            })
        } else {
            // Direct URL mode (same as PDRouter)
            info!(
                "VllmPDRouter::new called in direct URL mode with {} prefill, {} decode workers",
                prefill_urls.len(),
                decode_urls.len()
            );

            // Create underlying PD router with provided worker lists
            let pd_router = PDRouter::new(ctx).await?;
            let pd_router_ext = PDRouterExt::new(pd_router);

            // No service discovery in direct URL mode
            let service_registry = ServiceRegistry::new();

            info!("VllmPDRouter created successfully with direct URLs");

            // let prefill_workers = pd_router.worker_registry.get_prefill_workers();
            // let decode_workers = pd_router.worker_registry.get_decode_workers();
            // let prefill_policy = ctx.policy_registry.get_prefill_policy();
            // let decode_policy = ctx.policy_registry.get_decode_policy();

            // if prefill_policy.requires_initialization() {
            //     info!("Initializing prefill policy with workers.");
            //     prefill_policy.init_workers(&prefill_workers);
            // }
            // if decode_policy.requires_initialization() {
            //     info!("Initializing decode policy with workers.");
            //     decode_policy.init_workers(&decode_workers);
            // }
            // info!("Initializing prefill and decode policies with workers.");

            Ok(Self {
                pd_router_ext,
                service_registry: Arc::new(service_registry),
                http_client: reqwest::Client::new(),
                policy_registry: ctx.policy_registry.clone(),
                use_discovery: false,
                enable_profiling: vllm_pd_config.enable_profiling,
                profile_timeout_secs: vllm_pd_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: vllm_pd_config.intra_node_data_parallel_size,
            })
        }
    }

    /// Get a reference to the underlying PDRouter's worker registry
    /// This allows access to worker information for refresh operations
    pub fn worker_registry(&self) -> &WorkerRegistry {
        &self.pd_router_ext.pd_router.worker_registry
    }
}

// Delegate most RouterTrait methods to the underlying PDRouter,
// but override specific ones for vLLM behavior
#[async_trait]
impl RouterTrait for VllmPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.pd_router_ext.pd_router.health_generate(req).await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.pd_router_ext.pd_router.get_models(req).await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.pd_router_ext.pd_router.route_generate(headers, body, model_id).await
    }

    // Override OpenAI-compatible routes for vLLM two-stage processing
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        info!(
            "vLLM route_chat called, use_discovery={}",
            self.use_discovery
        );

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing
            info!("Using service discovery mode, processing vLLM two-stage request");

            // Convert to generic request and use vLLM processing
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized chat request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Process vLLM two-stage request with service discovery
            self.process_vllm_request(request_json, "/v1/chat/completions", headers, model_id)
                .await
        } else {
            // Direct URL mode - implement routing logic here (not delegating to PDRouter)
            info!("Using direct URL mode with VllmPDRouter's own routing logic");

            // Convert request to JSON
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized chat request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Get prefill and decode workers from worker_registry
            let prefill_workers = self.pd_router_ext.pd_router.worker_registry.get_prefill_workers();
            let decode_workers = self.pd_router_ext.pd_router.worker_registry.get_decode_workers();

            info!(
                "Found {} prefill workers, {} decode workers from worker_registry",
                prefill_workers.len(),
                decode_workers.len()
            );

            if prefill_workers.is_empty() || decode_workers.is_empty() {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "No workers available: {} prefill, {} decode",
                        prefill_workers.len(),
                        decode_workers.len()
                    ),
                )
                    .into_response();
            }

            // Select workers using policy
            let request_text = serde_json::to_string(&request_json).ok();
            let request_str = request_text.as_deref();

            let prefill_policy = self.policy_registry.get_prefill_policy();
            let decode_policy = self.policy_registry.get_decode_policy();

            // Get cached hash ring for consistent hashing
            let hash_ring = self.pd_router_ext.pd_router.worker_registry
                .get_hash_ring(model_id);

            let worker_info = SelectWorkerInfo {
                    request_text: request_str,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                };

            let prefill_idx = match prefill_policy.select_worker(&prefill_workers, &worker_info) {
                Some(idx) => idx,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let decode_idx = match decode_policy.select_worker(&decode_workers, &worker_info) {
                Some(idx) => idx,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Decode policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let prefill_worker = &prefill_workers[prefill_idx];
            let decode_worker = &decode_workers[decode_idx];
            // Load tracking is handled inside process_vllm_two_stage_request for fine-grained
            // tracking: prefill load only during prefill phase, decode load only during decode phase.

            info!(
                "Chat: Selected prefill={} [policy:{}], decode={} [policy:{}]",
                prefill_worker.url(),
                prefill_policy.name(),
                decode_worker.url(),
                decode_policy.name()
            );

            // Execute dual dispatch with vLLM two-stage processing
            let resp = match self
                .process_vllm_two_stage_request(
                    request_json,
                    prefill_worker.clone(),
                    decode_worker.clone(),
                    "/v1/chat/completions",
                    headers,
                )
                .await
            {
                Ok(response) => {
                    info!("Two-stage processing completed successfully");
                    response
                }
                Err(e) => {
                    info!("Two-stage processing failed: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request processing failed: {}", e),
                    )
                        .into_response()
                }
            };
            resp
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: &str,
    ) -> Response {
        info!(
            "vLLM route_completion called, use_discovery={}",
            self.use_discovery
        );

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing
            info!("Using service discovery mode, processing vLLM two-stage request");

            // Convert to generic request and use vLLM processing
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized completion request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Process vLLM two-stage request with service discovery
            self.process_vllm_request(request_json, "/v1/completions", headers, model_id)
                .await
        } else {
            // Direct URL mode - implement routing logic here (not delegating to PDRouter)
            info!("Using direct URL mode with VllmPDRouter's own routing logic");

            // Convert request to JSON
            let request_json = match serde_json::to_value(body) {
                Ok(json) => {
                    debug!(
                        "Serialized completion request: {}",
                        serde_json::to_string_pretty(&json).unwrap_or_default()
                    );
                    json
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                        .into_response()
                }
            };

            // Get prefill and decode workers from worker_registry
            let prefill_workers = self.pd_router_ext.pd_router.worker_registry.get_prefill_workers();
            let decode_workers = self.pd_router_ext.pd_router.worker_registry.get_decode_workers();

            info!(
                "Found {} prefill workers, {} decode workers from worker_registry",
                prefill_workers.len(),
                decode_workers.len()
            );

            if prefill_workers.is_empty() || decode_workers.is_empty() {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "No workers available: {} prefill, {} decode",
                        prefill_workers.len(),
                        decode_workers.len()
                    ),
                )
                    .into_response();
            }

            // Select workers using policy
            let request_text = serde_json::to_string(&request_json).ok();
            let request_str = request_text.as_deref();

            let prefill_policy = self.policy_registry.get_prefill_policy();
            let decode_policy = self.policy_registry.get_decode_policy();

            // Get cached hash ring for consistent hashing
            let hash_ring = self.pd_router_ext.pd_router.worker_registry
                .get_hash_ring(model_id);

            let worker_info = SelectWorkerInfo {
                    request_text: request_str,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                };

            let prefill_idx = match prefill_policy.select_worker(&prefill_workers, &worker_info) {
                Some(idx) => idx,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let decode_idx = match decode_policy.select_worker(&decode_workers, &worker_info) {
                Some(idx) => idx,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Decode policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

            let prefill_worker = &prefill_workers[prefill_idx];
            let decode_worker = &decode_workers[decode_idx];
            // Load tracking is handled inside process_vllm_two_stage_request for fine-grained
            // tracking: prefill load only during prefill phase, decode load only during decode phase.

            info!(
                "Completion: Selected prefill={} [policy:{}], decode={} [policy:{}]",
                prefill_worker.url(),
                prefill_policy.name(),
                decode_worker.url(),
                decode_policy.name()
            );

            // Execute dual dispatch with vLLM two-stage processing
            let resp = match self
                .process_vllm_two_stage_request(
                    request_json,
                    prefill_worker.clone(),
                    decode_worker.clone(),
                    "/v1/completions",
                    headers,
                )
                .await
            {
                Ok(response) => {
                    info!("Two-stage processing completed successfully");
                    response
                }
                Err(e) => {
                    info!("Two-stage processing failed: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request processing failed: {}", e),
                    )
                        .into_response()
                }
            };
            resp
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        self.pd_router_ext.pd_router.route_rerank(headers, body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        "vllm_pd"
    }

}
