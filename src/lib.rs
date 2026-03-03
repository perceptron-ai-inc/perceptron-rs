mod chat_completions;
mod client;
mod error;
mod media;
mod parsing;
mod pointing;
mod types;

pub use chat_completions::error::ApiErrorDetail;
pub use client::{Perceptron, PerceptronClient};
pub use error::PerceptronError;
pub use media::{Media, MediaFormat, MediaType};
pub use pointing::{BoundingBox, Point, Pointing, Polygon};
pub use types::{
    AnalyzeRequest, CaptionRequest, CaptionStyle, DetectRequest, OcrMode, OcrRequest, OutputFormat, PointingResponse,
    TextResponse,
};
