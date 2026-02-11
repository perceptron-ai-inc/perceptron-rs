use reqwest::Client;

use crate::chat_completions::*;
use crate::error::PerceptronError;
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
}

/// Trait for analyzing images with a Perceptron AI model.
pub trait Perceptron {
    /// Analyze an image using a Perceptron AI model.
    fn analyze_image(
        &self,
        request: AnalyzeImageRequest,
    ) -> impl Future<Output = Result<AnalyzeImageResponse, PerceptronError>> + Send;
}

impl Perceptron for PerceptronClient {
    async fn analyze_image(&self, request: AnalyzeImageRequest) -> Result<AnalyzeImageResponse, PerceptronError> {
        let wire_request = build_chat_completion_request(&request);

        let completion = self.chat_completions.complete(wire_request).await?;

        let response = match completion.choices.into_iter().next() {
            Some(choice) => AnalyzeImageResponse {
                content: choice.message.content,
                reasoning: choice.message.reasoning_content,
            },
            None => AnalyzeImageResponse {
                content: None,
                reasoning: None,
            },
        };

        Ok(response)
    }
}

fn build_chat_completion_request(request: &AnalyzeImageRequest) -> CreateChatCompletionRequest {
    let mut messages = Vec::new();

    if let Some(hint) = request.output_format.to_hint(request.reasoning) {
        messages.push(ChatCompletionMessage::System(ChatCompletionSystemMessage {
            content: ChatCompletionSystemMessageContent::Text(hint),
        }));
    }

    let user_content = ChatCompletionUserMessageContent::Array(vec![
        ChatCompletionContentPart::ImageUrl(ChatCompletionContentPartImage {
            image_url: ImageUrl {
                url: request.image_url.clone(),
            },
        }),
        ChatCompletionContentPart::Text(ChatCompletionContentPartText {
            text: request.message.clone(),
        }),
    ]);

    messages.push(ChatCompletionMessage::User(ChatCompletionUserMessage {
        content: user_content,
    }));

    CreateChatCompletionRequest {
        messages,
        model: request.model.clone(),
        max_completion_tokens: request.max_completion_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
    }
}
