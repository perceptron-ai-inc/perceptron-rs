use serde::{Deserialize, Serialize};

// ============================================================================
// Request Types
// ============================================================================

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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionContentPart {
    Text(ChatCompletionContentPartText),
    ImageUrl(ChatCompletionContentPartImage),
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
}

// ============================================================================
// Response Types
// ============================================================================

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
