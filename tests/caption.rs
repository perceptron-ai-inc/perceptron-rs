use perceptron_ai::{
    BoundingBox, CaptionRequest, CaptionStyle, OutputFormat, Perceptron, PerceptronClient, Point, Pointing,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn box_response() -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "content": r#"A cat on a windowsill <point_box mention="cat"> (10,20) (300,400) </point_box>"#,
                "reasoning_content": null
            }
        }]
    })
}

#[tokio::test]
async fn concise_default() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "<hint>BOX</hint>"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                        {"type": "text", "text": "Provide a concise, human-friendly caption for the upcoming image."}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(box_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = CaptionRequest::new("test-model", "https://example.com/img.jpg");
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![BoundingBox {
            x1: 10,
            y1: 20,
            x2: 300,
            y2: 400,
            mention: Some("cat".to_string()),
        }]))
    );
}

#[tokio::test]
async fn detailed_style() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "system",
                    "content": "<hint>BOX</hint>"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url"},
                        {"type": "text", "text": "Provide a detailed caption describing key objects, relationships, and context in the upcoming image."}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(box_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = CaptionRequest::new("test-model", "https://example.com/img.jpg").style(CaptionStyle::Detailed);
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![BoundingBox {
            x1: 10,
            y1: 20,
            x2: 300,
            y2: 400,
            mention: Some("cat".to_string()),
        }]))
    );
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
                    "content": "<hint>BOX THINK</hint>"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(box_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = CaptionRequest::new("test-model", "https://example.com/img.jpg").reasoning(true);
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert!(response.pointing.is_some());
}

#[tokio::test]
async fn multiple_boxes() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"A cat and dog <point_box mention="cat"> (10,20) (100,200) </point_box><point_box mention="dog"> (300,50) (500,400) </point_box>"#,
                    "reasoning_content": null
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = CaptionRequest::new("test-model", "https://example.com/img.jpg");
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![
            BoundingBox {
                x1: 10,
                y1: 20,
                x2: 100,
                y2: 200,
                mention: Some("cat".to_string())
            },
            BoundingBox {
                x1: 300,
                y1: 50,
                x2: 500,
                y2: 400,
                mention: Some("dog".to_string())
            },
        ]))
    );
}

#[tokio::test]
async fn point_format() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [
                {
                    "role": "system",
                    "content": "<hint>POINT</hint>"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"A cat <point mention="cat"> (150,250) </point>"#,
                    "reasoning_content": null
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = CaptionRequest::new("test-model", "https://example.com/img.jpg").output_format(OutputFormat::Point);
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Points(vec![Point {
            x: 150,
            y: 250,
            mention: Some("cat".to_string()),
        }]))
    );
}
