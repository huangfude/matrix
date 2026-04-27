//! 模型工具模块
//!
//! 提供 HTTP 客户端和模型信息处理功能

use std::time::Duration;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde_json::Value;
use serde::{Deserialize, Serialize};
use axum::http::HeaderMap;

// HTTP client for metadata fetching
pub static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
});

/// 模型信息结构体
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: Option<i64>,
    pub owned_by: Option<String>,
    pub root: Option<String>,
    pub parent: Option<String>,
}

/// 从 /v1/models 端点获取模型信息
pub async fn get_model_info(url: &str) -> Result<ModelInfo, String> {
    let base_url = url.trim_end_matches('/');
    let model_info_url = format!("{}/v1/models", base_url);

    let req = HTTP_CLIENT.get(&model_info_url);

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", model_info_url, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            model_info_url
        ));
    }

    let json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", model_info_url, e))?;

    // 不支持 LoRA Adapters
    let raw_model = json["data"].as_array()
                .and_then(|arr| arr.first()).ok_or("Data is empty!")?;
    // 通过owned_by字段获取？
    let model_info: ModelInfo = serde_json::from_value(raw_model.clone())
                .map_err(|e| format!("Failed to parse backend info: {}", e))?;

    Ok(model_info)
}


/// Header names for W3C Trace Context (OpenTelemetry) propagation
pub const TRACE_HEADER_NAMES: &[&str] = &["traceparent", "tracestate", "baggage"];

/// Propagate OpenTelemetry trace headers to a reqwest RequestBuilder
///
/// This enables distributed tracing across service boundaries by forwarding
/// W3C Trace Context headers from incoming requests to outgoing backend requests.
pub fn propagate_trace_headers(
    request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    propagate_headers(request, headers, TRACE_HEADER_NAMES)
}

/// Propagate specific headers from incoming request to outgoing reqwest RequestBuilder
///
/// This is a general-purpose helper for selectively forwarding headers by name.
/// Only headers whose names match the provided list (case-insensitive) are propagated.
///
/// # Arguments
/// * `request` - The reqwest RequestBuilder to add headers to
/// * `headers` - Optional incoming headers to check
/// * `header_names` - List of header names to propagate (matched case-insensitively)
///
/// # Returns
/// The RequestBuilder with matching headers added
pub fn propagate_headers(
    mut request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
    header_names: &[&str],
) -> reqwest::RequestBuilder {
    if let Some(h) = headers {
        for (k, v) in h.iter() {
            if header_names
                .iter()
                .any(|&name| k.as_str().eq_ignore_ascii_case(name))
            {
                request = request.header(k, v);
            }
        }
    }
    request
}
