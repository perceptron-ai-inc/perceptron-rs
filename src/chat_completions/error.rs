use reqwest::StatusCode;
use serde::Deserialize;

/// OpenAI-compatible error detail.
#[derive(Debug, Deserialize, Clone)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// OpenAI-compatible error response.
#[derive(Debug, Deserialize, Clone)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

/// Errors from the chat completions API.
#[derive(Debug, thiserror::Error)]
pub enum ChatCompletionError {
    /// HTTP request failed (network error, DNS, timeout, etc.)
    #[error("Request failed: {0}")]
    RequestFailed(String),

    /// API returned a non-success status code with a structured error.
    #[error("API error ({status}): {}", detail.message)]
    ApiError {
        /// HTTP status code
        status: StatusCode,
        /// Parsed error detail from the API
        detail: ApiErrorDetail,
    },

    /// Failed to parse the API response.
    #[error("Failed to parse response: {0}")]
    ParseFailed(String),
}
