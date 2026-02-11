mod chat_completions;
mod client;
mod error;
mod types;

pub use chat_completions::error::ApiErrorDetail;
pub use client::{Perceptron, PerceptronClient};
pub use error::PerceptronError;
pub use types::{AnalyzeImageRequest, AnalyzeImageResponse, OutputFormat};
