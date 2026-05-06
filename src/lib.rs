mod api;
mod client;
mod error;
mod media;
mod models;
mod parsing;
mod pointing;
mod prompting;
mod types;

pub use client::{Perceptron, PerceptronClient};
pub use error::ApiErrorDetail;
pub use error::PerceptronError;
pub use media::{Image, ImageFormat, Media, Modality, Video, VideoFormat};
pub use models::{Model, SamplingParameter};
pub use pointing::{BoundingBox, Clip, ClipTimestamp, Point, Pointing, Polygon};
pub use types::{
    AnalyzeRequest, CaptionRequest, CaptionStyle, DetectRequest, OcrMode, OcrRequest, OutputFormat, PointingResponse,
    QuestionRequest, TextResponse,
};
