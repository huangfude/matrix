use serde::{Deserialize, Serialize};
use openai_protocol::{
    UNKNOWN_MODEL_ID,
    rerank::RerankRequest,
};

/// V1 API compatibility format for rerank requests
/// Matches Python's V1RerankReqInput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1RerankReqInput {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        let v1_model = if let Some(model) = v1.model 
            { 
                model 
            } else { 
                UNKNOWN_MODEL_ID.to_string() 
            };
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: v1_model,
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        }
    }
}
