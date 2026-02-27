use reqwest::Client;

use crate::chat_completions::*;
use crate::error::PerceptronError;
use crate::pointing;
use crate::types::*;

/// Client for the Perceptron SDK.
#[derive(Clone, Debug)]
pub struct PerceptronClient {
    chat_completions: ChatCompletionsClient,
}

impl PerceptronClient {
    /// Create a new client with default settings.
    pub fn new() -> Self {
        Self {
            chat_completions: ChatCompletionsClient::new(),
        }
    }

    /// Set the base URL for the model. Defaults to `https://api.perceptron.inc`.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.chat_completions.set_base_url(url.into());
        self
    }

    /// Set the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.chat_completions.set_api_key(key.into());
        self
    }

    /// Add a custom header to include on every request.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.chat_completions.set_header(name.into(), value.into());
        self
    }

    /// Set the HTTP client to use for requests.
    pub fn http_client(mut self, client: Client) -> Self {
        self.chat_completions.set_http_client(client);
        self
    }

    async fn send(&self, wire_request: CreateChatCompletionRequest) -> Result<TextResponse, PerceptronError> {
        let completion = self.chat_completions.complete(wire_request).await?;

        let response = match completion.choices.into_iter().next() {
            Some(choice) => TextResponse {
                content: choice.message.content,
                reasoning: choice.message.reasoning_content,
            },
            None => TextResponse {
                content: None,
                reasoning: None,
            },
        };

        Ok(response)
    }

    async fn send_and_extract(
        &self,
        wire_request: CreateChatCompletionRequest,
        output_format: &OutputFormat,
    ) -> Result<PointingResponse, PerceptronError> {
        let completion = self.chat_completions.complete(wire_request).await?;

        let response = match completion.choices.into_iter().next() {
            Some(choice) => {
                let pointing = choice
                    .message
                    .content
                    .as_deref()
                    .and_then(|text| pointing::extract(text, output_format));
                PointingResponse {
                    content: choice.message.content,
                    reasoning: choice.message.reasoning_content,
                    pointing,
                }
            }
            None => PointingResponse {
                content: None,
                reasoning: None,
                pointing: None,
            },
        };

        Ok(response)
    }
}

/// Trait for analyzing visual media with a Perceptron AI model.
pub trait Perceptron {
    /// Analyze visual media with a custom prompt.
    fn analyze(
        &self,
        request: AnalyzeRequest,
    ) -> impl Future<Output = Result<PointingResponse, PerceptronError>> + Send;

    /// Generate a caption for visual media.
    fn caption(
        &self,
        request: CaptionRequest,
    ) -> impl Future<Output = Result<PointingResponse, PerceptronError>> + Send;

    /// Extract text using OCR.
    fn ocr(&self, request: OcrRequest) -> impl Future<Output = Result<TextResponse, PerceptronError>> + Send;

    /// Detect and segment objects.
    fn detect(&self, request: DetectRequest) -> impl Future<Output = Result<PointingResponse, PerceptronError>> + Send;
}

impl Perceptron for PerceptronClient {
    async fn analyze(&self, request: AnalyzeRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Text);
        let desc = RequestDescriptor {
            image_url: request.image_url,
            system_prompts: system_hint(Some(&output_format), request.generation_params.reasoning)
                .into_iter()
                .collect(),
            user_text: Some(request.message),
            generation_params: request.generation_params,
        };
        self.send_and_extract(build_wire_request(desc), &output_format).await
    }

    async fn caption(&self, request: CaptionRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Box);
        let user_text = match request.style {
            CaptionStyle::Concise => "Provide a concise, human-friendly caption for the upcoming image.",
            CaptionStyle::Detailed => {
                "Provide a detailed caption describing key objects, relationships, and context in the upcoming image."
            }
        };
        let desc = RequestDescriptor {
            image_url: request.image_url,
            system_prompts: system_hint(Some(&output_format), request.generation_params.reasoning)
                .into_iter()
                .collect(),
            user_text: Some(user_text.to_string()),
            generation_params: request.generation_params,
        };
        self.send_and_extract(build_wire_request(desc), &output_format).await
    }

    async fn ocr(&self, request: OcrRequest) -> Result<TextResponse, PerceptronError> {
        let ocr_system = "You are an OCR (Optical Character Recognition) system. \
            Accurately detect, extract, and transcribe all readable text from the image.";
        let mut system_prompts: Vec<String> = system_hint(None, request.generation_params.reasoning)
            .into_iter()
            .collect();
        system_prompts.push(ocr_system.to_string());
        let user_text = match request.mode {
            OcrMode::Plain => None,
            OcrMode::Markdown => Some(
                "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate.".to_string(),
            ),
            OcrMode::Html => Some(
                "Transcribe every readable word in the image using HTML markup.".to_string(),
            ),
        };
        let desc = RequestDescriptor {
            image_url: request.image_url,
            system_prompts,
            user_text,
            generation_params: request.generation_params,
        };
        self.send(build_wire_request(desc)).await
    }

    async fn detect(&self, request: DetectRequest) -> Result<PointingResponse, PerceptronError> {
        let domain_system = match &request.classes {
            Some(classes) if !classes.is_empty() => {
                format!(
                    "Your goal is to segment out the following categories: {}",
                    classes.join(", ")
                )
            }
            _ => "Your goal is to segment out the objects in the scene".to_string(),
        };
        let mut system_prompts: Vec<String> =
            system_hint(Some(&OutputFormat::Box), request.generation_params.reasoning)
                .into_iter()
                .collect();
        system_prompts.push(domain_system);
        let desc = RequestDescriptor {
            image_url: request.image_url,
            system_prompts,
            user_text: None,
            generation_params: request.generation_params,
        };
        self.send_and_extract(build_wire_request(desc), &OutputFormat::Box)
            .await
    }
}

/// Generate the hint tag for the system prompt based on output format and reasoning.
fn system_hint(output_format: Option<&OutputFormat>, enable_reasoning: Option<bool>) -> Option<String> {
    let mut components = Vec::new();

    match output_format {
        Some(OutputFormat::Point) => components.push("POINT"),
        Some(OutputFormat::Box) => components.push("BOX"),
        Some(OutputFormat::Polygon) => components.push("POLYGON"),
        _ => {}
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

struct RequestDescriptor {
    image_url: String,
    system_prompts: Vec<String>,
    user_text: Option<String>,
    generation_params: GenerationParams,
}

fn build_wire_request(desc: RequestDescriptor) -> CreateChatCompletionRequest {
    let mut messages = Vec::new();

    for system in desc.system_prompts {
        messages.push(ChatCompletionMessage::System(ChatCompletionSystemMessage {
            content: ChatCompletionSystemMessageContent::Text(system),
        }));
    }

    let mut user_parts = vec![ChatCompletionContentPart::ImageUrl(ChatCompletionContentPartImage {
        image_url: ImageUrl { url: desc.image_url },
    })];

    if let Some(text) = desc.user_text {
        user_parts.push(ChatCompletionContentPart::Text(ChatCompletionContentPartText { text }));
    }

    messages.push(ChatCompletionMessage::User(ChatCompletionUserMessage {
        content: ChatCompletionUserMessageContent::Array(user_parts),
    }));

    CreateChatCompletionRequest {
        messages,
        model: desc.generation_params.model,
        max_completion_tokens: desc.generation_params.max_completion_tokens,
        temperature: desc.generation_params.temperature,
        top_p: desc.generation_params.top_p,
        top_k: desc.generation_params.top_k,
        frequency_penalty: desc.generation_params.frequency_penalty,
        presence_penalty: desc.generation_params.presence_penalty,
    }
}
