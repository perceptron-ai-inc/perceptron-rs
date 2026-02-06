use crate::chat_completions::error::ApiErrorDetail;
use crate::chat_completions::ChatCompletionError;

/// Errors that can occur when using the Perceptron SDK.
#[derive(Debug, thiserror::Error)]
pub enum PerceptronError {
    /// HTTP request failed (network error, DNS, timeout, etc.)
    #[error("Request failed: {0}")]
    RequestFailed(String),

    /// API returned a non-success status code.
    #[error("API error ({status}): {}", detail.message)]
    ApiError {
        status: u16,
        detail: ApiErrorDetail,
    },

    /// Failed to parse the API response.
    #[error("Failed to parse response: {0}")]
    ParseFailed(String),
}

impl From<ChatCompletionError> for PerceptronError {
    fn from(err: ChatCompletionError) -> Self {
        match err {
            ChatCompletionError::RequestFailed(msg) => PerceptronError::RequestFailed(msg),
            ChatCompletionError::ApiError { status, detail } => {
                PerceptronError::ApiError {
                    status: status.as_u16(),
                    detail,
                }
            }
            ChatCompletionError::ParseFailed(msg) => PerceptronError::ParseFailed(msg),
        }
    }
}
