use perceptron_ai::{CaptionRequest, CaptionStyle, DetectRequest, Media, OcrMode, OcrRequest, Perceptron};
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

const QWEN_MODEL: &str = "qwen3-vl-72b";

fn box_content() -> &'static str {
    r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#
}

#[tokio::test]
async fn qwen_caption_concise() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Describe the primary subjects, their actions, and visible context in one vivid sentence."}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = CaptionRequest::new(QWEN_MODEL, Media::image_url("https://example.com/img.jpg"));
    let response = client.caption(request).await.unwrap();
    assert!(response.content.is_some());
}

#[tokio::test]
async fn qwen_caption_detailed() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Provide a multi-sentence caption that calls out subjects, relationships, scene intent, and any text embedded in the image."}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request =
        CaptionRequest::new(QWEN_MODEL, Media::image_url("https://example.com/img.jpg")).style(CaptionStyle::Detailed);
    let response = client.caption(request).await.unwrap();
    assert!(response.content.is_some());
}

#[tokio::test]
async fn qwen_ocr_plain() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": QWEN_MODEL,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Read all the text in the image."}
                ]}
            ]
        })),
        common::response("Hello World", None),
    )
    .await;

    let request = OcrRequest::new(QWEN_MODEL, Media::image_url("https://example.com/doc.jpg"));
    let response = client.ocr(request).await.unwrap();
    assert_eq!(response.content, Some("Hello World".to_string()));
}

#[tokio::test]
async fn qwen_ocr_markdown() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "qwenvl markdown"}
                ]}
            ]
        })),
        common::response("# Hello", None),
    )
    .await;

    let request = OcrRequest::new(QWEN_MODEL, Media::image_url("https://example.com/doc.jpg")).mode(OcrMode::Markdown);
    let response = client.ocr(request).await.unwrap();
    assert_eq!(response.content, Some("# Hello".to_string()));
}

#[tokio::test]
async fn qwen_detect_general() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "system", "content": "Locate every object of interest and report bounding box coordinates in JSON format."},
                {"role": "user", "content": [
                    {"type": "image_url"}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = DetectRequest::new(QWEN_MODEL, Media::image_url("https://example.com/img.jpg"));
    let response = client.detect(request).await.unwrap();
    assert!(response.content.is_some());
}

#[tokio::test]
async fn qwen_detect_with_classes() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "system", "content": "Locate every instance that belongs to the following categories: \"cat, dog\". Report bbox coordinates in JSON format."},
                {"role": "user", "content": [
                    {"type": "image_url"}
                ]}
            ]
        })),
        common::response(box_content(), None),
    )
    .await;

    let request = DetectRequest::new(QWEN_MODEL, Media::image_url("https://example.com/img.jpg"))
        .classes(vec!["cat".to_string(), "dog".to_string()]);
    let response = client.detect(request).await.unwrap();
    assert!(response.content.is_some());
}
