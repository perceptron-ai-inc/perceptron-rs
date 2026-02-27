use perceptron_ai::{OcrMode, OcrRequest, Perceptron, PerceptronClient};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn success_response(content: &str) -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "content": content,
                "reasoning_content": null
            }
        }]
    })
}

#[tokio::test]
async fn plain_default() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/doc.jpg"}}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response("Hello World")))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = OcrRequest::new("test-model", "https://example.com/doc.jpg");
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("Hello World".to_string()));
}

#[tokio::test]
async fn markdown_mode() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "system"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url"},
                        {"type": "text", "text": "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate."}
                    ]
                }
            ]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(success_response("# Hello\n\nWorld")),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = OcrRequest::new("test-model", "https://example.com/doc.jpg").mode(OcrMode::Markdown);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("# Hello\n\nWorld".to_string()));
}

#[tokio::test]
async fn html_mode() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "system"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url"},
                        {"type": "text", "text": "Transcribe every readable word in the image using HTML markup."}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response("<p>Hello World</p>")))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = OcrRequest::new("test-model", "https://example.com/doc.jpg").mode(OcrMode::Html);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("<p>Hello World</p>".to_string()));
}

#[tokio::test]
async fn with_reasoning() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "system",
                    "content": "<hint>THINK</hint>"
                },
                {
                    "role": "system",
                    "content": "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": "Hello",
                    "reasoning_content": "I can see text"
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = OcrRequest::new("test-model", "https://example.com/doc.jpg").reasoning(true);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("Hello".to_string()));
    assert_eq!(response.reasoning, Some("I can see text".to_string()));
}
