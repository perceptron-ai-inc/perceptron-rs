use std::collections::HashMap;

use reqwest::Client;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::chat_completions::*;
use super::models::*;
use crate::error::{ApiErrorDetail, ApiErrorResponse, PerceptronError};

/// Low-level HTTP client for the Perceptron API.
#[derive(Clone, Debug)]
pub struct ApiClient {
    pub http: Client,
    pub base_url: String,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
}

impl ApiClient {
    pub fn new() -> Self {
        Self {
            http: Client::new(),
            base_url: "https://api.perceptron.inc".to_string(),
            api_key: None,
            headers: HashMap::new(),
        }
    }

    /// List all available models.
    pub async fn models(&self) -> Result<ModelsResponse, PerceptronError> {
        self.get("/v1/models?extended=true").await
    }

    /// Get a single model by ID.
    pub async fn model(&self, id: &str) -> Result<ModelResponse, PerceptronError> {
        self.get(&format!("/v1/models/{}?extended=true", id)).await
    }

    /// Send a chat completion request.
    pub async fn chat_completions(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, PerceptronError> {
        self.post("/v1/chat/completions", &request).await
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, PerceptronError> {
        self.send(self.http.get(format!("{}{}", self.base_url, path))).await
    }

    async fn post<T: DeserializeOwned>(&self, path: &str, body: &impl Serialize) -> Result<T, PerceptronError> {
        self.send(self.http.post(format!("{}{}", self.base_url, path)).json(body))
            .await
    }

    async fn send<T: DeserializeOwned>(&self, mut req_builder: reqwest::RequestBuilder) -> Result<T, PerceptronError> {
        if let Some(key) = &self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        for (name, value) in &self.headers {
            req_builder = req_builder.header(name, value);
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| PerceptronError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(Self::error_from_response(response).await);
        }

        response
            .json()
            .await
            .map_err(|e| PerceptronError::ParseFailed(e.to_string()))
    }

    async fn error_from_response(response: reqwest::Response) -> PerceptronError {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<ApiErrorResponse>(&body)
            .map(|r| r.error)
            .unwrap_or(ApiErrorDetail {
                message: body,
                error_type: None,
                param: None,
                code: None,
            });
        PerceptronError::ApiError { status, detail }
    }
}
