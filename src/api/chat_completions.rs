use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionContentPartText {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionContentPartImage {
    pub image_url: ImageUrl,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionContentPartVideo {
    pub video_url: VideoUrl,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionContentPartAudio {
    pub audio_url: AudioUrl,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionContentPartInputAudio {
    pub input_audio: InputAudio,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionContentPart {
    Text(ChatCompletionContentPartText),
    ImageUrl(ChatCompletionContentPartImage),
    VideoUrl(ChatCompletionContentPartVideo),
    AudioUrl(ChatCompletionContentPartAudio),
    InputAudio(ChatCompletionContentPartInputAudio),
}

/// Annotation format the model should emit alongside text output.
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationFormat {
    Point,
    Box,
    Polygon,
    Clip,
}

/// Internal-tool toggles for Perceptron vision models.
#[derive(Debug, Serialize, Clone, Default)]
pub struct InternalTools {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<bool>,
}

/// Perceptron vision-model controls, sent as the `vision_config` request field.
#[derive(Debug, Serialize, Clone, Default)]
pub struct VisionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation_format: Option<AnnotationFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_tools: Option<InternalTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_audio_in_video: Option<bool>,
}

impl VisionConfig {
    /// True when no field is set, so the whole object can be omitted from the request.
    pub fn is_empty(&self) -> bool {
        self.enable_thinking.is_none()
            && self.annotation_format.is_none()
            && self.internal_tools.is_none()
            && self.enable_audio_in_video.is_none()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ChatCompletionSystemMessageContent {
    Text(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ChatCompletionUserMessageContent {
    Text(String),
    Array(Vec<ChatCompletionContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionSystemMessage {
    pub content: ChatCompletionSystemMessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionUserMessage {
    pub content: ChatCompletionUserMessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionMessage {
    System(ChatCompletionSystemMessage),
    User(ChatCompletionUserMessage),
}

#[derive(Debug, Serialize, Clone)]
pub struct CreateChatCompletionRequest {
    pub messages: Vec<ChatCompletionMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_config: Option<VisionConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatCompletionChoice {
    pub message: ChatCompletionResponseMessage,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CreateChatCompletionResponse {
    pub choices: Vec<ChatCompletionChoice>,
}
