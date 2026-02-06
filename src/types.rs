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

impl OutputFormat {
    /// Generate the hint tag for the system prompt.
    pub(crate) fn to_hint(&self, enable_reasoning: Option<bool>) -> Option<String> {
        let mut components = Vec::new();

        match self {
            OutputFormat::Text => {}
            OutputFormat::Point => components.push("POINT"),
            OutputFormat::Box => components.push("BOX"),
            OutputFormat::Polygon => components.push("POLYGON"),
        }

        if enable_reasoning.unwrap_or(false) {
            components.push("THINK");
        }

        if components.is_empty() {
            None
        } else {
            Some(format!("<hint>{}</hint>", components.join(" ")))
        }
    }
}

/// Parameters for an image analysis request.
///
/// Use [`AnalyzeImageRequest::new`] to create a request with required fields,
/// then chain optional setters using the builder pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AnalyzeImageRequest {
    /// The model to use.
    pub model: String,
    /// The user message/prompt to send.
    pub message: String,
    /// Image URL for multimodal requests.
    pub image_url: String,
    /// Output format: Text (default), Point, Box, or Polygon.
    pub output_format: OutputFormat,
    /// Enable chain-of-thought reasoning.
    pub reasoning: Option<bool>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    /// Maximum number of completion tokens to generate.
    pub max_completion_tokens: Option<u32>,
}

impl AnalyzeImageRequest {
    /// Create a new image analysis request with required fields.
    pub fn new(model: impl Into<String>, message: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            message: message.into(),
            image_url: image_url.into(),
            output_format: OutputFormat::default(),
            reasoning: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            max_completion_tokens: None,
        }
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Enable chain-of-thought reasoning.
    pub fn reasoning(mut self, enable: bool) -> Self {
        self.reasoning = Some(enable);
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set nucleus sampling probability.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set maximum completion tokens.
    pub fn max_completion_tokens(mut self, max_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_tokens);
        self
    }
}

/// Result of an image analysis request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AnalyzeImageResponse {
    /// The main response content from the model.
    pub content: Option<String>,
    /// Chain-of-thought reasoning content (if reasoning was enabled).
    pub reasoning: Option<String>,
}
