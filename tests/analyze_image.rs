use perceptron_ai::{AnalyzeImageRequest, OutputFormat, Perceptron, PerceptronClient};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_request(model: &str) -> AnalyzeImageRequest {
    AnalyzeImageRequest::new(model, "Describe this", "https://example.com/img.jpg")
}

fn success_response() -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "content": "a cat",
                "reasoning_content": "I see fur"
            }
        }]
    })
}

#[tokio::test]
async fn complete_fails() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "test-model"})))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": {
                "message": "internal server error",
                "type": "server_error"
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let result = client.analyze_image(test_request("test-model")).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("internal server error"));
}

#[tokio::test]
async fn empty_choices() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "test-model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": []
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let response = client.analyze_image(test_request("test-model")).await.unwrap();

    assert_eq!(response.content, None);
    assert_eq!(response.reasoning, None);
}

#[tokio::test]
async fn success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "test-model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let response = client.analyze_image(test_request("test-model")).await.unwrap();

    assert_eq!(response.content, Some("a cat".to_string()));
    assert_eq!(response.reasoning, Some("I see fur".to_string()));
}

#[tokio::test]
async fn all_fields() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "test-model",
            "max_completion_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
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
