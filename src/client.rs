use reqwest::Client;

use crate::api::ApiClient;
use crate::api::chat_completions::*;
use crate::error::PerceptronError;
use crate::media::{Audio, Media};
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
        output_format: Option<&OutputFormat>,
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
        let output_format = request.output_format.as_ref();
        let profile = &prompting::ISAAC;
        let system_prompts: Vec<String> = profile
            .question
            .resolve_system(output_format, &request.media)
            .map(str::to_string)
            .into_iter()
            .collect();
        let desc = RequestDescriptor {
            media: request.media,
            vision_config: vision_config(
                output_format,
                request.reasoning,
                request.focus,
                request.enable_audio_in_video,
            ),
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
        self.send_and_extract(build_wire_request(desc), output_format).await
    }

    async fn analyze(&self, request: AnalyzeRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.as_ref();
        let desc = RequestDescriptor {
            media: request.media,
            vision_config: vision_config(
                output_format,
                request.reasoning,
                request.focus,
                request.enable_audio_in_video,
            ),
            system_prompts: Vec::new(),
            user_text: Some(request.message),
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        };
        self.send_and_extract(build_wire_request(desc), output_format).await
    }

    async fn caption(&self, request: CaptionRequest) -> Result<PointingResponse, PerceptronError> {
        let output_format = request.output_format.unwrap_or(OutputFormat::Box);
        let profile = &prompting::ISAAC;
        let system_prompts: Vec<String> = profile
            .caption
            .resolve_system(&request.media)
            .map(str::to_string)
            .into_iter()
            .collect();
        let user_text = Some(profile.caption.resolve_user(&request.style, &request.media).to_string());
        let desc = RequestDescriptor {
            media: request.media,
            vision_config: vision_config(
                Some(&output_format),
                request.reasoning,
                request.focus,
                request.enable_audio_in_video,
            ),
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
        self.send_and_extract(build_wire_request(desc), Some(&output_format))
            .await
    }

    async fn ocr(&self, request: OcrRequest) -> Result<TextResponse, PerceptronError> {
        let profile = &prompting::ISAAC;
        let system_prompts: Vec<String> = profile.ocr.resolve_system().map(str::to_string).into_iter().collect();
        let user_text = request
            .prompt
            .or_else(|| profile.ocr.resolve_user(&request.mode).map(str::to_string));
        let desc = RequestDescriptor {
            media: request.image.into(),
            vision_config: vision_config(None, request.reasoning, request.focus, None),
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
        let profile = &prompting::ISAAC;
        let system_prompts = vec![
            profile
                .detect
                .resolve_system(request.classes.as_deref(), &request.media),
        ];
        let desc = RequestDescriptor {
            media: request.media,
            vision_config: vision_config(Some(&OutputFormat::Box), request.reasoning, request.focus, None),
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
        self.send_and_extract(build_wire_request(desc), Some(&OutputFormat::Box))
            .await
    }
}

/// Build the `vision_config` request field, or `None` when nothing is set so the field is omitted.
fn vision_config(
    output_format: Option<&OutputFormat>,
    reasoning: Option<bool>,
    focus: Option<bool>,
    enable_audio_in_video: Option<bool>,
) -> Option<VisionConfig> {
    let annotation_format = output_format.and_then(|format| match format {
        OutputFormat::Text => None,
        OutputFormat::Point => Some(AnnotationFormat::Point),
        OutputFormat::Box => Some(AnnotationFormat::Box),
        OutputFormat::Polygon => Some(AnnotationFormat::Polygon),
        OutputFormat::Clip => Some(AnnotationFormat::Clip),
    });
    let config = VisionConfig {
        enable_thinking: reasoning,
        annotation_format,
        internal_tools: focus.map(|focus| InternalTools { focus: Some(focus) }),
        enable_audio_in_video,
    };
    (!config.is_empty()).then_some(config)
}

struct RequestDescriptor {
    media: Media,
    vision_config: Option<VisionConfig>,
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

    let media_part = match desc.media {
        Media::Image(image) => ChatCompletionContentPart::ImageUrl(ChatCompletionContentPartImage {
            image_url: ImageUrl { url: image.to_url() },
        }),
        Media::Video(video) => ChatCompletionContentPart::VideoUrl(ChatCompletionContentPartVideo {
            video_url: VideoUrl { url: video.to_url() },
        }),
        Media::Audio(Audio::Url { src }) => ChatCompletionContentPart::AudioUrl(ChatCompletionContentPartAudio {
            audio_url: AudioUrl { url: src },
        }),
        Media::Audio(Audio::Base64 { format, data }) => {
            ChatCompletionContentPart::InputAudio(ChatCompletionContentPartInputAudio {
                input_audio: InputAudio {
                    data,
                    format: format.to_string(),
                },
            })
        }
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
        vision_config: desc.vision_config,
    }
}
