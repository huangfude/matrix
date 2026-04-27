use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use rand::Rng;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{error, field::Empty, info, info_span, warn, Span};

use smg::{
    observability::{
        metrics::{method_to_static_str, Metrics},
    },
    routers::error::extract_error_code_from_response,
};

#[derive(Clone)]
pub struct AuthConfig {
    /// Precomputed SHA-256 hashes of API keys, used for constant-time comparison
    /// that doesn't leak key length via timing.
    api_key_hashes: Option<Vec<[u8; 32]>>,
}

impl AuthConfig {
    pub fn new(api_keys: Option<Vec<String>>) -> Self {
        Self {
            api_key_hashes: api_keys.map(|keys| {
                keys.into_iter()
                    .map(|k| Sha256::digest(k.as_bytes()).into())
                    .collect()
            }),
        }
    }
}

/// Middleware to validate Bearer token against configured API key.
/// Only active when router has an API key configured.
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(expected_hashes) = &auth_config.api_key_hashes {
        // 只有当配置了至少一个 API 密钥时才进行验证
        if !expected_hashes.is_empty() {
            let token = request.headers()
                .get(header::AUTHORIZATION)
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.strip_prefix("Bearer "));

            let authorized = token.is_some_and(|t| {
                let token_hash = Sha256::digest(t.as_bytes());
                expected_hashes.iter().any(|expected_hash| {
                    token_hash.as_slice()
                        .ct_eq(expected_hash)
                        .unwrap_u8()
                        == 1
                })
            });
            
            if !authorized {
                return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
            }
        }
    }

    next.run(request).await
}

/// Alphanumeric characters for request ID generation (as bytes for O(1) indexing)
const REQUEST_ID_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Generate OpenAI-compatible request ID based on endpoint.
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else if path.contains("/responses") {
        "resp-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    // Use byte array indexing (O(1)) instead of chars().nth() (O(n))
    let mut rng = rand::rng();
    let random_part: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..REQUEST_ID_CHARS.len());
            REQUEST_ID_CHARS[idx] as char
        })
        .collect();

    format!("{}{}", prefix, random_part)
}

/// Extension type for storing request ID
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Tower Layer for request ID middleware
#[derive(Clone)]
pub struct RequestIdLayer {
    headers: Arc<Vec<String>>,
}

impl RequestIdLayer {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers: Arc::new(headers),
        }
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Tower Service for request ID middleware
#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
    headers: Arc<Vec<String>>,
}

impl<S> Service<Request> for RequestIdMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let headers = self.headers.clone();

        // Extract request ID from headers or generate new one
        let mut request_id = None;

        for header_name in headers.iter() {
            if let Some(header_value) = req.headers().get(header_name) {
                if let Ok(value) = header_value.to_str() {
                    request_id = Some(value.to_string());
                    break;
                }
            }
        }

        let request_id = request_id.unwrap_or_else(|| generate_request_id(req.uri().path()));

        // Insert request ID into request extensions for other middleware/handlers to use
        req.extensions_mut().insert(RequestId(request_id.clone()));

        let id_parts: Vec<&str> = request_id.split('-').collect();
        let x_request_id = id_parts.get(1).or_else(|| id_parts.get(0)).map_or("UNKOWN", |v| v);
        let x_request_id_str = x_request_id.to_string();

        let x_req_id_value = req.headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .unwrap_or(x_request_id)
            .to_string(); 
        req.headers_mut().insert("x-request-id", HeaderValue::from_str(&x_req_id_value).unwrap());
        
        // Call the inner service
        let future = self.inner.call(req);

        Box::pin(async move {
            let mut response = future.await?;

            // Add request ID to response headers
            response.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&x_request_id_str)
                    .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id")),
            );

            Ok(response)
        })
    }
}

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        info_span!(
            target: "matrix_router::otel-trace",
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            error = Empty,
            module = "matrix_router"
        )
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        let method = method_to_static_str(request.method().as_str());
        let path = normalize_path_for_metrics(request.uri().path());
        Metrics::record_http_request(method, &path);

        // Log the request start
        info!(
            target: "matrix_router::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug)]
pub struct ResponseLogger {
    _start_time: Instant,
}

impl Default for ResponseLogger {
    fn default() -> Self {
        Self {
            _start_time: Instant::now(),
        }
    }
}

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();
        let status_code = status.as_u16();

        let error_code = extract_error_code_from_response(response);

        // Layer 1: HTTP metrics
        Metrics::record_http_response(status_code, error_code);

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "matrix_router::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "matrix_router::response",
                "request failed with client error"
            );
        } else {
            info!(
                target: "matrix_router::response",
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger::default())
}


/// Normalize path for metrics to avoid high cardinality.
/// Replaces dynamic segments (IDs, UUIDs) with `{id}` placeholder.
/// Only allocates when normalization is needed; uses single-pass with byte offsets.
fn normalize_path_for_metrics(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut segment_start = 0;
    let mut segment_idx = 0;
    let mut result: Option<String> = None;

    for (pos, &b) in bytes.iter().enumerate() {
        if b == b'/' || pos == bytes.len() - 1 {
            // Determine segment end (include last char if not a slash)
            let segment_end = if b == b'/' { pos } else { pos + 1 };
            let segment = &path[segment_start..segment_end];

            // Check segments after index 2 for dynamic IDs
            if segment_idx > 2 && !segment.is_empty() && is_dynamic_id(segment) {
                // Initialize result with everything before this segment
                let result = result.get_or_insert_with(|| {
                    let mut s = String::with_capacity(path.len());
                    s.push_str(&path[..segment_start]);
                    s
                });
                result.push_str("{id}");
            } else if let Some(ref mut r) = result {
                // Already normalizing, append this segment as-is
                r.push_str(segment);
            }

            // Add slash after segment (except at end)
            if b == b'/' {
                if let Some(ref mut r) = result {
                    r.push('/');
                }
                segment_start = pos + 1;
                segment_idx += 1;
            }
        }
    }

    result.unwrap_or_else(|| path.to_owned())
}

/// Check if segment looks like a dynamic ID (prefixed ID, UUID, or numeric).
#[inline]
fn is_dynamic_id(s: &str) -> bool {
    // Prefixed IDs: resp_xxx, chatcmpl_xxx (len > 10 with underscore)
    if s.len() > 10 && s.contains('_') {
        return true;
    }
    // UUIDs: 32+ hex chars with dashes
    if s.len() >= 32 && s.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-') {
        return true;
    }
    // Numeric IDs
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}