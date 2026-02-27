use perceptron_ai::{BoundingBox, DetectRequest, Perceptron, PerceptronClient, Pointing};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn success_response() -> serde_json::Value {
    json!({
        "choices": [{
            "message": {
                "content": r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#,
                "reasoning_content": null
            }
        }]
    })
}

#[tokio::test]
async fn general_detection() {
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
                    "role": "system",
                    "content": "Your goal is to segment out the objects in the scene"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ]
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = DetectRequest::new("test-model", "https://example.com/img.jpg");
    let response = client.detect(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![BoundingBox {
            x1: 10,
            y1: 20,
            x2: 100,
            y2: 200,
            mention: Some("cat".to_string()),
        }]))
    );
}

#[tokio::test]
async fn with_classes() {
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
                    "role": "system",
                    "content": "Your goal is to segment out the following categories: cat, dog"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"<point_box mention="cat"> (10,20) (100,200) </point_box><point_box mention="dog"> (300,400) (500,600) </point_box>"#,
                    "reasoning_content": null
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = DetectRequest::new("test-model", "https://example.com/img.jpg")
        .classes(vec!["cat".to_string(), "dog".to_string()]);
    let response = client.detect(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![
            BoundingBox {
                x1: 10,
                y1: 20,
                x2: 100,
                y2: 200,
                mention: Some("cat".to_string()),
            },
            BoundingBox {
                x1: 300,
                y1: 400,
                x2: 500,
                y2: 600,
                mention: Some("dog".to_string()),
            },
        ]))
    );
}

#[tokio::test]
async fn multiple_detections() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"<point_box mention="person"> (50,30) (200,500) </point_box><point_box mention="car"> (400,200) (700,450) </point_box><point_box mention="tree"> (750,50) (900,500) </point_box>"#,
                    "reasoning_content": null
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = DetectRequest::new("test-model", "https://example.com/img.jpg");
    let response = client.detect(request).await.unwrap();

    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![
            BoundingBox {
                x1: 50,
                y1: 30,
                x2: 200,
                y2: 500,
                mention: Some("person".to_string())
            },
            BoundingBox {
                x1: 400,
                y1: 200,
                x2: 700,
                y2: 450,
                mention: Some("car".to_string())
            },
            BoundingBox {
                x1: 750,
                y1: 50,
                x2: 900,
                y2: 500,
                mention: Some("tree".to_string())
            },
        ]))
    );
}

#[tokio::test]
async fn collection() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"<collection mention="cat"><point_box> (10,20) (100,200) </point_box><point_box> (300,50) (500,400) </point_box></collection>"#,
                    "reasoning_content": null
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = DetectRequest::new("test-model", "https://example.com/img.jpg");
    let response = client.detect(request).await.unwrap();

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
                mention: Some("cat".to_string())
            },
        ]))
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
                },
                {
                    "role": "system",
                    "content": "Your goal is to segment out the objects in the scene"
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "content": r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#,
                    "reasoning_content": "I see a cat in the image"
                }
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = PerceptronClient::new().base_url(server.uri());
    let request = DetectRequest::new("test-model", "https://example.com/img.jpg").reasoning(true);
    let response = client.detect(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(response.reasoning, Some("I see a cat in the image".to_string()));
    assert_eq!(
        response.pointing,
        Some(Pointing::Boxes(vec![BoundingBox {
            x1: 10,
            y1: 20,
            x2: 100,
            y2: 200,
            mention: Some("cat".to_string()),
        }]))
    );
}
