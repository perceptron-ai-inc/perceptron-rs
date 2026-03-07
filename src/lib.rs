mod api;
mod client;
mod error;
mod media;
mod models;
mod parsing;
mod pointing;
pub mod prompting;
mod types;

pub use client::{Perceptron, PerceptronClient};
pub use error::ApiErrorDetail;
pub use error::PerceptronError;
pub use media::{Media, MediaFormat, Modality};
pub use models::{Model, SamplingParameter};
pub use pointing::{BoundingBox, Point, Pointing, Polygon};
pub use prompting::{
    CaptionPromptTemplate, DetectPromptTemplate, OcrPromptTemplate, PromptProfile, QuestionPromptTemplate,
    resolve_prompt_profile,
};
pub use types::{
    AnalyzeRequest, CaptionRequest, CaptionStyle, DetectRequest, OcrMode, OcrRequest, OutputFormat, PointingResponse,
    QuestionRequest, TextResponse,
};
