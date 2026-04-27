use std::sync::Arc;

use axum::{
    http::HeaderMap,
    response::Response,
};
use tracing::debug;

use openai_protocol::rerank::RerankRequest;
use smg::routers::{RouterTrait, http::router::Router};

/// RouterExt 用于处理 V1 API 的 rerank 请求
/// 当 router 为 Router 类型时，需要特殊处理
pub struct RouterExt;

impl RouterExt {
    pub fn new(_router: Arc<dyn RouterTrait>) -> Self {
        RouterExt
    }

    pub async fn route_rerank(
        router: Arc<dyn RouterTrait>,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: &str,
    ) -> Response {
        let router_any = router.as_any();
        if let Some(concrete_router) = router_any.downcast_ref::<Router>() {
            debug!("Routing rerank request to Router");
            // 直接返回 route_typed_request 的响应，不进行额外处理
            // 因为 route_typed_request 已经返回了正确的格式
            concrete_router
                .route_typed_request(headers, body, "/v1/rerank", model_id)
                .await
        } else {
            debug!(
                "Routing rerank request to router type: {}",
                router.router_type()
            );
            router.route_rerank(headers, body, model_id).await
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use axum::{
        http::{HeaderMap, HeaderValue},
        response::Response,
    };
    use async_trait::async_trait;
    use openai_protocol::rerank::RerankRequest;
    use smg::routers::{
        RouterTrait,
    };

    // Mock RouterTrait implementation for testing
    #[derive(Debug)]
    struct MockRouter {
        response_content: String,
    }

    impl MockRouter {
        fn new(response_content: &str) -> Self {
            Self {
                response_content: response_content.to_string(),
            }
        }
    }

    #[async_trait]
    impl RouterTrait for MockRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn router_type(&self) -> &'static str {
            "MockRouter"
        }

        async fn route_rerank(
            &self,
            _headers: Option<&HeaderMap>,
            _body: &RerankRequest,
            _model_id: &str,
        ) -> Response {
            Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(self.response_content.clone().into())
                .unwrap()
        }
    }

    // Create test data
    fn create_test_rerank_request() -> RerankRequest {
        RerankRequest {
            query: "test query".to_string(),
            documents: vec!["document 1".to_string(), "document 2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(2),
            return_documents: true,
            rid: None,
            user: None,
        }
    }

    fn create_test_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test-token"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers
    }

    #[test]
    fn test_router_ext_new() {
        let _mock_router = Arc::new(MockRouter::new(""));

        // RouterExt is a zero-sized struct, so we just verify it can be created
        // The actual functionality is in the associated functions
        assert_eq!(size_of::<RouterExt>(), 0);
    }

    #[tokio::test]
    async fn test_route_rerank_with_other_router_type() {
        // Test when router is not a concrete Router type
        let mock_router = Arc::new(MockRouter::new("other_router_response"));
        
        let headers = create_test_headers();
        let body = create_test_rerank_request();
        let model_id = "test-model";
        
        let response = RouterExt::route_rerank(
            mock_router.clone(),
            Some(&headers),
            &body,
            model_id,
        ).await;
        
        // Verify response
        assert_eq!(response.status(), 200);
        
        // Extract response body
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert_eq!(body_str, "other_router_response");
    }

    #[tokio::test]
    async fn test_route_rerank_without_headers() {
        // Test route_rerank with None headers
        let mock_router = Arc::new(MockRouter::new("no_headers_response"));
        
        let body = create_test_rerank_request();
        let model_id = "test-model";
        
        let response = RouterExt::route_rerank(
            mock_router.clone(),
            None,
            &body,
            model_id,
        ).await;
        
        // Verify response
        assert_eq!(response.status(), 200);
        
        // Extract response body
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert_eq!(body_str, "no_headers_response");
    }

    #[tokio::test]
    async fn test_route_rerank_with_empty_documents() {
        // Test with empty documents list
        let mock_router = Arc::new(MockRouter::new("empty_docs_response"));
        
        let headers = create_test_headers();
        let mut body = create_test_rerank_request();
        body.documents.clear();
        
        let response = RouterExt::route_rerank(
            mock_router.clone(),
            Some(&headers),
            &body,
            "test-model",
        ).await;
        
        // Verify response
        assert_eq!(response.status(), 200);
        
        // Extract response body
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert_eq!(body_str, "empty_docs_response");
    }

    #[tokio::test]
    async fn test_route_rerank_with_long_query() {
        // Test with long query string
        let mock_router = Arc::new(MockRouter::new("long_query_response"));
        
        let headers = create_test_headers();
        let mut body = create_test_rerank_request();
        body.query = "a".repeat(10000); // Very long query
        
        let response = RouterExt::route_rerank(
            mock_router.clone(),
            Some(&headers),
            &body,
            "test-model",
        ).await;
        
        // Verify response
        assert_eq!(response.status(), 200);
        
        // Extract response body
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert_eq!(body_str, "long_query_response");
    }
}