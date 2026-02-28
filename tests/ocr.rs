use perceptron_ai::{Media, MediaFormat, OcrMode, OcrRequest, Perceptron};
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

#[tokio::test]
async fn plain_default() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
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
        })),
        common::response("Hello World", None),
    )
    .await;

    let request = OcrRequest::new("test-model", Media::image_url("https://example.com/doc.jpg"));
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("Hello World".to_string()));
}

#[tokio::test]
async fn markdown_mode() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate."}
                ]}
            ]
        })),
        common::response("# Hello\n\nWorld", None),
    )
    .await;

    let request =
        OcrRequest::new("test-model", Media::image_url("https://example.com/doc.jpg")).mode(OcrMode::Markdown);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("# Hello\n\nWorld".to_string()));
}

#[tokio::test]
async fn html_mode() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Transcribe every readable word in the image using HTML markup."}
                ]}
            ]
        })),
        common::response("<p>Hello World</p>", None),
    )
    .await;

    let request = OcrRequest::new("test-model", Media::image_url("https://example.com/doc.jpg")).mode(OcrMode::Html);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("<p>Hello World</p>".to_string()));
}

#[tokio::test]
async fn base64_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/webp;base64,docdata"}}
                ]}
            ]
        })),
        common::response("Hello World", None),
    )
    .await;

    let request = OcrRequest::new("test-model", Media::base64(MediaFormat::Webp, "docdata"));
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("Hello World".to_string()));
}

#[tokio::test]
async fn with_reasoning() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>THINK</hint>"},
                {"role": "system", "content": "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."}
            ]
        })),
        common::response("Hello", Some("I can see text")),
    )
    .await;

    let request = OcrRequest::new("test-model", Media::image_url("https://example.com/doc.jpg")).reasoning(true);
    let response = client.ocr(request).await.unwrap();

    assert_eq!(response.content, Some("Hello".to_string()));
    assert_eq!(response.reasoning, Some("I can see text".to_string()));
}
