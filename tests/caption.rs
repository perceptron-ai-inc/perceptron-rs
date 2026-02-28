use perceptron_ai::{
    BoundingBox, CaptionRequest, CaptionStyle, Media, MediaFormat, OutputFormat, Perceptron, Point, Pointing,
};
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

fn box_content() -> &'static str {
    r#"A cat on a windowsill <point_box mention="cat"> (10,20) (300,400) </point_box>"#
}

fn assert_single_cat_box(response: &perceptron_ai::PointingResponse) {
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
async fn concise_default() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    {"type": "text", "text": "Provide a concise, human-friendly caption for the upcoming image."}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::image_url("https://example.com/img.jpg"));
    let response = client.caption(request).await.unwrap();

    assert_single_cat_box(&response);
}

#[tokio::test]
async fn detailed_style() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Provide a detailed caption describing key objects, relationships, and context in the upcoming image."}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::image_url("https://example.com/img.jpg"))
        .style(CaptionStyle::Detailed);
    let response = client.caption(request).await.unwrap();

    assert_single_cat_box(&response);
}

#[tokio::test]
async fn with_reasoning() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{"role": "system", "content": "<hint>BOX THINK</hint>"}]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::image_url("https://example.com/img.jpg")).reasoning(true);
    let response = client.caption(request).await.unwrap();

    assert!(response.content.is_some());
    assert!(response.pointing.is_some());
}

#[tokio::test]
async fn multiple_boxes() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>BOX</hint>"}]})),
        common::response(r#"A cat and dog <point_box mention="cat"> (10,20) (100,200) </point_box><point_box mention="dog"> (300,50) (500,400) </point_box>"#, None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::image_url("https://example.com/img.jpg"));
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
async fn base64_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,imgdata"}},
                    {"type": "text"}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::base64(MediaFormat::Jpeg, "imgdata"));
    let response = client.caption(request).await.unwrap();

    assert_single_cat_box(&response);
}

#[tokio::test]
async fn point_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{"role": "system", "content": "<hint>POINT</hint>"}]
        })),
        common::response(r#"A cat <point mention="cat"> (150,250) </point>"#, None),
    )
    .await;

    let request = CaptionRequest::new("test-model", Media::image_url("https://example.com/img.jpg"))
        .output_format(OutputFormat::Point);
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
