mod chat_completions;
mod client;
mod error;
mod pointing;
mod types;

pub use chat_completions::error::ApiErrorDetail;
pub use client::{Perceptron, PerceptronClient};
pub use error::PerceptronError;
pub use types::{
    AnalyzeRequest, BoundingBox, CaptionRequest, CaptionStyle, DetectRequest, GenerationParams, OcrMode, OcrRequest,
    OutputFormat, Point, Pointing, PointingResponse, Polygon, TextResponse,
};
