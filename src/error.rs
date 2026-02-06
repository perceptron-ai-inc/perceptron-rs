use crate::chat_completions::ChatCompletionError;

/// Errors that can occur when using the Perceptron SDK.
#[derive(Debug, thiserror::Error)]
pub enum PerceptronError {
    /// An error from the chat completions API.
    #[error(transparent)]
    ChatCompletion(#[from] ChatCompletionError),
}
