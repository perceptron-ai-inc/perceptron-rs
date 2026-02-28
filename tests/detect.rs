use perceptron_ai::{BoundingBox, DetectRequest, Media, MediaFormat, Perceptron, Pointing};
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

fn single_box_content() -> &'static str {
    r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#
}

fn assert_single_cat_box(response: &perceptron_ai::PointingResponse) {
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
async fn general_detection() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "system", "content": "Your goal is to segment out the objects in the scene"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]}
            ]
        })),
        common::response(single_box_content(), None),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::image_url("https://example.com/img.jpg"));
    let response = client.detect(request).await.unwrap();

    assert_single_cat_box(&response);
}

#[tokio::test]
async fn with_classes() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "system", "content": "Your goal is to segment out the following categories: cat, dog"}
            ]
        })),
        common::response(r#"<point_box mention="cat"> (10,20) (100,200) </point_box><point_box mention="dog"> (300,400) (500,600) </point_box>"#, None),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::image_url("https://example.com/img.jpg"))
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
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>BOX</hint>"}]})),
        common::response(r#"<point_box mention="person"> (50,30) (200,500) </point_box><point_box mention="car"> (400,200) (700,450) </point_box><point_box mention="tree"> (750,50) (900,500) </point_box>"#, None),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::image_url("https://example.com/img.jpg"));
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
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>BOX</hint>"}]})),
        common::response(r#"<collection mention="cat"><point_box> (10,20) (100,200) </point_box><point_box> (300,50) (500,400) </point_box></collection>"#, None),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::image_url("https://example.com/img.jpg"));
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
async fn base64_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system"},
                {"role": "system"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,imgdata"}}
                ]}
            ]
        })),
        common::response(single_box_content(), None),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::base64(MediaFormat::Png, "imgdata"));
    let response = client.detect(request).await.unwrap();

    assert_single_cat_box(&response);
}

#[tokio::test]
async fn with_reasoning() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX THINK</hint>"},
                {"role": "system", "content": "Your goal is to segment out the objects in the scene"}
            ]
        })),
        common::response(single_box_content(), Some("I see a cat in the image")),
    )
    .await;

    let request = DetectRequest::new("test-model", Media::image_url("https://example.com/img.jpg")).reasoning(true);
    let response = client.detect(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(response.reasoning, Some("I see a cat in the image".to_string()));
    assert_single_cat_box(&response);
}
