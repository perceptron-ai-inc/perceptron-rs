use serde::{Deserialize, Serialize};

/// Output format for model responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OutputFormat {
    /// Plain text response (default).
    #[default]
    Text,
    /// Return point coordinates as `<point>` tags for spatial annotation.
    Point,
    /// Return bounding box coordinates as `<point_box>` tags for spatial annotation.
    Box,
    /// Return polygon coordinates for spatial annotation.
    Polygon,
}

/// Style for caption requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum CaptionStyle {
    /// A brief, human-friendly caption (default).
    #[default]
    Concise,
    /// A thorough, detailed description.
    Detailed,
}

/// Output mode for OCR requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OcrMode {
    /// Plain text extraction (default).
    #[default]
    Plain,
    /// Markdown-formatted output.
    Markdown,
    /// HTML markup output.
    Html,
}

/// Generation parameters shared across all request types.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GenerationParams {
    /// Model to use for the request.
    pub model: String,
    /// Whether to enable chain-of-thought reasoning.
    pub reasoning: Option<bool>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Top-k sampling value.
    pub top_k: Option<u32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_completion_tokens: Option<u32>,
}

/// Generate delegation setter methods for [`GenerationParams`] on a request struct.
macro_rules! generation_param_setters {
    () => {
        /// Enable chain-of-thought reasoning.
        pub fn reasoning(mut self, enable: bool) -> Self {
            self.generation_params.reasoning = Some(enable);
            self
        }

        /// Set the sampling temperature.
        pub fn temperature(mut self, temperature: f32) -> Self {
            self.generation_params.temperature = Some(temperature);
            self
        }

        /// Set nucleus sampling probability.
        pub fn top_p(mut self, top_p: f32) -> Self {
            self.generation_params.top_p = Some(top_p);
            self
        }

        /// Set top-k sampling.
        pub fn top_k(mut self, top_k: u32) -> Self {
            self.generation_params.top_k = Some(top_k);
            self
        }

        /// Set frequency penalty.
        pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
            self.generation_params.frequency_penalty = Some(frequency_penalty);
            self
        }

        /// Set presence penalty.
        pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
            self.generation_params.presence_penalty = Some(presence_penalty);
            self
        }

        /// Set maximum completion tokens.
        pub fn max_completion_tokens(mut self, max_tokens: u32) -> Self {
            self.generation_params.max_completion_tokens = Some(max_tokens);
            self
        }
    };
}

/// Parameters for a media analysis request.
///
/// Use [`AnalyzeRequest::new`] to create a request with required fields,
/// then chain optional setters using the builder pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AnalyzeRequest {
    /// User message or prompt to send.
    pub message: String,
    /// URL of the media to analyze.
    pub image_url: String,
    /// Output format for the response.
    pub output_format: Option<OutputFormat>,
    /// Generation parameters.
    #[serde(flatten)]
    pub generation_params: GenerationParams,
}

impl AnalyzeRequest {
    /// Create a new analysis request with required fields.
    pub fn new(model: impl Into<String>, message: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            image_url: image_url.into(),
            output_format: None,
            generation_params: GenerationParams {
                model: model.into(),
                ..Default::default()
            },
        }
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    generation_param_setters!();
}

/// Parameters for generating a caption.
///
/// Use [`CaptionRequest::new`] to create a request with required fields,
/// then chain optional setters using the builder pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CaptionRequest {
    /// URL of the media to caption.
    pub image_url: String,
    /// Caption style.
    pub style: CaptionStyle,
    /// Output format for the response (defaults to Box).
    pub output_format: Option<OutputFormat>,
    /// Generation parameters.
    #[serde(flatten)]
    pub generation_params: GenerationParams,
}

impl CaptionRequest {
    /// Create a new caption request.
    pub fn new(model: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            image_url: image_url.into(),
            style: CaptionStyle::default(),
            output_format: None,
            generation_params: GenerationParams {
                model: model.into(),
                ..Default::default()
            },
        }
    }

    /// Set the caption style.
    pub fn style(mut self, style: CaptionStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    generation_param_setters!();
}

/// Parameters for an OCR request.
///
/// Use [`OcrRequest::new`] to create a request with required fields,
/// then chain optional setters using the builder pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OcrRequest {
    /// URL of the media to extract text from.
    pub image_url: String,
    /// OCR output mode.
    pub mode: OcrMode,
    /// Generation parameters.
    #[serde(flatten)]
    pub generation_params: GenerationParams,
}

impl OcrRequest {
    /// Create a new OCR request.
    pub fn new(model: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            image_url: image_url.into(),
            mode: OcrMode::default(),
            generation_params: GenerationParams {
                model: model.into(),
                ..Default::default()
            },
        }
    }

    /// Set the OCR output mode.
    pub fn mode(mut self, mode: OcrMode) -> Self {
        self.mode = mode;
        self
    }

    generation_param_setters!();
}

/// Parameters for an object detection request.
///
/// Use [`DetectRequest::new`] to create a request with required fields,
/// then chain optional setters using the builder pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetectRequest {
    /// URL of the media to detect objects in.
    pub image_url: String,
    /// Optional list of object categories to detect.
    pub classes: Option<Vec<String>>,
    /// Generation parameters.
    #[serde(flatten)]
    pub generation_params: GenerationParams,
}

impl DetectRequest {
    /// Create a new detection request.
    pub fn new(model: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            image_url: image_url.into(),
            classes: None,
            generation_params: GenerationParams {
                model: model.into(),
                ..Default::default()
            },
        }
    }

    /// Set the object categories to detect.
    pub fn classes(mut self, classes: Vec<String>) -> Self {
        self.classes = Some(classes);
        self
    }

    generation_param_setters!();
}

/// A point annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Point {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
    /// Optional label.
    pub mention: Option<String>,
}

/// A bounding box annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BoundingBox {
    /// Top-left X coordinate.
    pub x1: u32,
    /// Top-left Y coordinate.
    pub y1: u32,
    /// Bottom-right X coordinate.
    pub x2: u32,
    /// Bottom-right Y coordinate.
    pub y2: u32,
    /// Optional label.
    pub mention: Option<String>,
}

/// A polygon annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Polygon {
    /// Polygon hull as (x, y) pairs.
    pub hull: Vec<(u32, u32)>,
    /// Optional label.
    pub mention: Option<String>,
}

/// Response for text-only methods (ocr, question).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TextResponse {
    /// The main response content from the model.
    pub content: Option<String>,
    /// Chain-of-thought reasoning content (if reasoning was enabled).
    pub reasoning: Option<String>,
}

/// Pointing data extracted from model output — exactly one spatial type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Pointing {
    /// Point annotations.
    Points(Vec<Point>),
    /// Bounding box annotations.
    Boxes(Vec<BoundingBox>),
    /// Polygon annotations.
    Polygons(Vec<Polygon>),
}

/// Response for spatial methods (analyze, caption, detect).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PointingResponse {
    /// The main response content from the model.
    pub content: Option<String>,
    /// Chain-of-thought reasoning content (if reasoning was enabled).
    pub reasoning: Option<String>,
    /// Extracted spatial pointing data.
    pub pointing: Option<Pointing>,
}
