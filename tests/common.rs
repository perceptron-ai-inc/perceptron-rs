#![allow(dead_code)]

use perceptron_ai::PerceptronClient;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub async fn setup() -> (MockServer, PerceptronClient) {
    let server = MockServer::start().await;
    let client = PerceptronClient::new().base_url(server.uri());
    (server, client)
}

pub fn response(content: &str, reasoning: Option<&str>) -> Value {
    json!({
        "choices": [{
            "message": {
                "content": content,
                "reasoning_content": reasoning
            }
        }]
    })
}

pub async fn mock_response(server: &MockServer, matcher: impl wiremock::Match + 'static, body: Value) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(matcher)
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(server)
        .await;
}
