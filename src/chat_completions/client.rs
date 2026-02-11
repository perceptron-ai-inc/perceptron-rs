use std::collections::HashMap;

use reqwest::Client;

use super::error::{ApiErrorDetail, ApiErrorResponse, ChatCompletionError};
use super::types::*;

/// Low-level client for the chat completions API.
#[cfg_attr(test, faux::create)]
#[derive(Clone, Debug)]
pub struct ChatCompletionsClient {
    http: Client,
    base_url: String,
    api_key: Option<String>,
    headers: HashMap<String, String>,
}

#[cfg_attr(test, faux::methods)]
impl ChatCompletionsClient {
    pub fn new() -> Self {
        Self {
            http: Client::new(),
            base_url: "https://api.perceptron.inc".to_string(),
            api_key: None,
            headers: HashMap::new(),
        }
    }

    pub fn set_base_url(&mut self, url: String) {
        self.base_url = url;
    }

    pub fn set_api_key(&mut self, key: String) {
        self.api_key = Some(key);
    }

    pub fn set_header(&mut self, name: String, value: String) {
        self.headers.insert(name, value);
    }

    pub fn set_http_client(&mut self, client: Client) {
        self.http = client;
    }

    /// Send a chat completion request and return the raw response.
    pub async fn complete(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, ChatCompletionError> {
        let mut req_builder = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request);

        if let Some(key) = &self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        for (name, value) in &self.headers {
            req_builder = req_builder.header(name, value);
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| ChatCompletionError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .map_err(|e| ChatCompletionError::ParseFailed(e.to_string()))?;
            let detail = serde_json::from_str::<ApiErrorResponse>(&body)
                .map(|r| r.error)
                .unwrap_or(ApiErrorDetail {
                    message: format!("Failed to parse error response body: {body}"),
                    error_type: None,
                    param: None,
                    code: None,
                });
            return Err(ChatCompletionError::ApiError { status, detail });
        }

        response
            .json()
            .await
            .map_err(|e| ChatCompletionError::ParseFailed(e.to_string()))
    }
}
