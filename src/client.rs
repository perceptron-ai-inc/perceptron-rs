use reqwest::Client;

use crate::api::ApiClient;
use crate::api::chat_completions::*;
use crate::error::PerceptronError;
use crate::media::*;
use crate::models::Model;
use crate::parsing;
use crate::prompting;
use crate::types::*;

/// Client for the Perceptron SDK.
#[derive(Clone, Debug)]
pub struct PerceptronClient {
    api: ApiClient,
}

impl PerceptronClient {
    /// Create a new client with default settings.
    pub fn new() -> Self {
        Self { api: ApiClient::new() }
    }

    /// Set the base URL for the model. Defaults to `https://api.perceptron.inc`.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.api.base_url = url.into();
        self
    }

    /// Set the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api.api_key = Some(key.into());
        self
    }

    /// Add a custom header to include on every request.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.api.headers.insert(name.into(), value.into());
        self
    }

    /// Set the HTTP client to use for requests.
    pub fn http_client(mut self, client: Client) -> Self {
        self.api.http = client;
        self
    }

    async fn send(&self, wire_request: CreateChatCompletionRequest) -> Result<TextResponse, PerceptronError> {
        let completion = self.api.chat_completions(wire_request).await?;

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
        let completion = self.api.chat_completions(wire_request).await?;

        let response = match completion.choices.into_iter().next() {
            Some(choice) => {
                let pointing = choice
                    .message
                    .content
                    .as_deref()
                    .and_then(|text| parsing::extract(text, output_format));
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
    /// List all available models.
    fn models(&self) -> impl Future<Output = Result<Vec<Model>, PerceptronError>> + Send;

    /// Get a single model by ID.
    fn model(&self, id: &str) -> impl Future<Output = Result<Model, PerceptronError>> + Send;

    /// Ask a question about visual media.
    fn question(
        &self,
        request: QuestionRequest,
    ) -> impl Future<Output = Result<PointingResponse, PerceptronError>> + Send;

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
    async fn models(&self) -> Result<Vec<Model>, PerceptronError> {
        let resp = self.api.models().await?;
        Ok(resp.data.into_iter().map(Model::from).collect())
    }

    async fn model(&self, id: &str) -> Result<Model, PerceptronError> {
        let resp = self.api.model(id).await?;
        Ok(resp.into())
    }

    async fn question(&self, request: QuestionRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Text);
        let profile = prompting::resolve_prompt_profile(&request.model);
        let mut system_prompts: Vec<String> = system_hint(Some(&output_format), request.reasoning)
            .into_iter()
            .collect();
        if let Some(system) = profile.question.system(&output_format) {
            system_prompts.push(system.to_string());
        }
        let desc = RequestDescriptor {
            media: request.media,
            system_prompts,
            user_text: Some(request.question),
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };
        self.send_and_extract(build_wire_request(desc), &output_format).await
    }

    async fn analyze(&self, request: AnalyzeRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Text);
        let desc = RequestDescriptor {
            media: request.media,
            system_prompts: system_hint(Some(&output_format), request.reasoning)
                .into_iter()
                .collect(),
            user_text: Some(request.message),
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };
        self.send_and_extract(build_wire_request(desc), &output_format).await
    }

    async fn caption(&self, request: CaptionRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Box);
        let profile = prompting::resolve_prompt_profile(&request.model);
        let mut system_prompts: Vec<String> = system_hint(Some(&output_format), request.reasoning)
            .into_iter()
            .collect();
        if let Some(system) = profile.caption.system {
            system_prompts.push(system.to_string());
        }
        let user_text = Some(profile.caption.user_text(&request.style).to_string());
        let desc = RequestDescriptor {
            media: request.media,
            system_prompts,
            user_text,
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };
        self.send_and_extract(build_wire_request(desc), &output_format).await
    }

    async fn ocr(&self, request: OcrRequest) -> Result<TextResponse, PerceptronError> {
        let profile = prompting::resolve_prompt_profile(&request.model);
        let mut system_prompts: Vec<String> = system_hint(None, request.reasoning).into_iter().collect();
        if let Some(system) = profile.ocr.system {
            system_prompts.push(system.to_string());
        }
        let user_text = request
            .prompt
            .or_else(|| profile.ocr.user_text(&request.mode).map(|s| s.to_string()));
        let desc = RequestDescriptor {
            media: request.media,
            system_prompts,
            user_text,
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };
        self.send(build_wire_request(desc)).await
    }

    async fn detect(&self, request: DetectRequest) -> Result<PointingResponse, PerceptronError> {
        let profile = prompting::resolve_prompt_profile(&request.model);
        let mut system_prompts: Vec<String> = system_hint(Some(&OutputFormat::Box), request.reasoning)
            .into_iter()
            .collect();
        system_prompts.push(profile.detect.system_text(request.classes.as_deref()));
        let desc = RequestDescriptor {
            media: request.media,
            system_prompts,
            user_text: None,
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
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
    media: Media,
    system_prompts: Vec<String>,
    user_text: Option<String>,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
}

fn build_wire_request(desc: RequestDescriptor) -> CreateChatCompletionRequest {
    let mut messages = Vec::new();

    for system in desc.system_prompts {
        messages.push(ChatCompletionMessage::System(ChatCompletionSystemMessage {
            content: ChatCompletionSystemMessageContent::Text(system),
        }));
    }

    let media_url = desc.media.to_url();
    let media_part = match desc.media.modality() {
        Modality::Image => ChatCompletionContentPart::ImageUrl(ChatCompletionContentPartImage {
            image_url: ImageUrl { url: media_url },
        }),
        Modality::Video => ChatCompletionContentPart::VideoUrl(ChatCompletionContentPartVideo {
            video_url: VideoUrl { url: media_url },
        }),
    };
    let mut user_parts = vec![media_part];

    if let Some(text) = desc.user_text {
        user_parts.push(ChatCompletionContentPart::Text(ChatCompletionContentPartText { text }));
    }

    messages.push(ChatCompletionMessage::User(ChatCompletionUserMessage {
        content: ChatCompletionUserMessageContent::Array(user_parts),
    }));

    CreateChatCompletionRequest {
        messages,
        model: desc.model,
        max_completion_tokens: desc.max_tokens,
        temperature: desc.temperature,
        top_p: desc.top_p,
        top_k: desc.top_k,
        frequency_penalty: desc.frequency_penalty,
        presence_penalty: desc.presence_penalty,
    }
}
