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

    #[cfg(test)]
    fn with_mock(mock: ChatCompletionsClient) -> Self {
        Self { chat_completions: mock }
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

    /// Analyze an image using a Perceptron AI model.
    ///
    /// Sends a multimodal chat completion request with the specified image and prompt,
    /// and returns the model's response. Supports configurable sampling parameters.
    pub async fn analyze_image(&self, request: AnalyzeImageRequest) -> Result<AnalyzeImageResponse, PerceptronError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_completions::ChatCompletionError;

    fn test_request() -> AnalyzeImageRequest {
        AnalyzeImageRequest::new("test-model", "Describe this", "https://example.com/img.jpg")
    }

    fn user_message() -> ChatCompletionMessage {
        ChatCompletionMessage::User(ChatCompletionUserMessage {
            content: ChatCompletionUserMessageContent::Array(vec![
                ChatCompletionContentPart::ImageUrl(ChatCompletionContentPartImage {
                    image_url: ImageUrl {
                        url: "https://example.com/img.jpg".to_string(),
                    },
                }),
                ChatCompletionContentPart::Text(ChatCompletionContentPartText {
                    text: "Describe this".to_string(),
                }),
            ]),
        })
    }

    fn match_request() -> impl Fn(&CreateChatCompletionRequest) -> bool {
        |req| {
            req.model == "test-model"
                && req.messages == vec![user_message()]
                && req.max_completion_tokens.is_none()
                && req.temperature.is_none()
                && req.top_p.is_none()
                && req.top_k.is_none()
                && req.frequency_penalty.is_none()
                && req.presence_penalty.is_none()
        }
    }

    #[tokio::test]
    async fn analyze_image_complete_fails() {
        let mut mock = ChatCompletionsClient::faux();
        faux::when!(mock.complete(_ = faux::from_fn!(match_request())))
            .once()
            .then(|_| Err(ChatCompletionError::RequestFailed("timeout".to_string())));

        let client = PerceptronClient::with_mock(mock);
        let result = client.analyze_image(test_request()).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn analyze_image_empty_choices() {
        let mut mock = ChatCompletionsClient::faux();
        faux::when!(mock.complete(_ = faux::from_fn!(match_request())))
            .once()
            .then(|_| Ok(CreateChatCompletionResponse { choices: vec![] }));

        let client = PerceptronClient::with_mock(mock);
        let response = client.analyze_image(test_request()).await.unwrap();

        assert_eq!(response.content, None);
        assert_eq!(response.reasoning, None);
    }

    #[tokio::test]
    async fn analyze_image_success() {
        let mut mock = ChatCompletionsClient::faux();
        faux::when!(mock.complete(_ = faux::from_fn!(match_request())))
            .once()
            .then(|_| {
                Ok(CreateChatCompletionResponse {
                    choices: vec![ChatCompletionChoice {
                        message: ChatCompletionResponseMessage {
                            content: Some("a cat".to_string()),
                            reasoning_content: Some("I see fur".to_string()),
                        },
                    }],
                })
            });

        let client = PerceptronClient::with_mock(mock);
        let response = client.analyze_image(test_request()).await.unwrap();

        assert_eq!(response.content, Some("a cat".to_string()));
        assert_eq!(response.reasoning, Some("I see fur".to_string()));
    }

    #[tokio::test]
    async fn analyze_image_all_fields() {
        let mut mock = ChatCompletionsClient::faux();
        faux::when!(mock.complete(
            _ = faux::from_fn!(|req: &CreateChatCompletionRequest| {
                req.model == "test-model"
                    && req.messages
                        == vec![
                            ChatCompletionMessage::System(ChatCompletionSystemMessage {
                                content: ChatCompletionSystemMessageContent::Text(
                                    "<hint>POINT THINK</hint>".to_string(),
                                ),
                            }),
                            user_message(),
                        ]
                    && req.max_completion_tokens == Some(100)
                    && req.temperature == Some(0.7)
                    && req.top_p == Some(0.9)
                    && req.top_k == Some(50)
                    && req.frequency_penalty == Some(0.5)
                    && req.presence_penalty == Some(0.3)
            })
        ))
        .once()
        .then(|_| {
            Ok(CreateChatCompletionResponse {
                choices: vec![ChatCompletionChoice {
                    message: ChatCompletionResponseMessage {
                        content: Some("a cat".to_string()),
                        reasoning_content: Some("I see fur".to_string()),
                    },
                }],
            })
        });

        let client = PerceptronClient::with_mock(mock);
        let request = AnalyzeImageRequest::new("test-model", "Describe this", "https://example.com/img.jpg")
            .output_format(OutputFormat::Point)
            .reasoning(true)
            .temperature(0.7)
            .top_p(0.9)
            .top_k(50)
            .frequency_penalty(0.5)
            .presence_penalty(0.3)
            .max_completion_tokens(100);
        let response = client.analyze_image(request).await.unwrap();

        assert_eq!(response.content, Some("a cat".to_string()));
        assert_eq!(response.reasoning, Some("I see fur".to_string()));
    }
}
