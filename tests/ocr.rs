use perceptron_ai::{Media, MediaFormat, OcrMode, OcrRequest, Perceptron};
use rstest::rstest;
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

#[rstest]
#[case::isaac(
    "isaac-test",
    Some(
        "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
    ),
    None
)]
#[case::qwen("qwen3-vl-72b", None, Some("Read all the text in the image."))]
#[case::unknown_defaults_to_isaac(
    "unknown-model",
    Some(
        "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
    ),
    None
)]
#[tokio::test]
async fn plain(#[case] model: &str, #[case] expected_system: Option<&str>, #[case] expected_user_text: Option<&str>) {
    let (server, client) = common::setup().await;

    let mut messages = vec![];
    if let Some(system) = expected_system {
        messages.push(json!({"role": "system", "content": system}));
    }
    let mut user_content = vec![json!({"type": "image_url", "image_url": {"url": "https://example.com/doc.jpg"}})];
    if let Some(text) = expected_user_text {
        user_content.push(json!({"type": "text", "text": text}));
    }
    messages.push(json!({"role": "user", "content": user_content}));

    common::mock_response(
        &server,
        body_partial_json(json!({"model": model, "messages": messages})),
        common::response("Hello World", None),
    )
    .await;

    let request = OcrRequest::new(model, Media::image_url("https://example.com/doc.jpg"));
    let response = client.ocr(request).await.unwrap();
    assert_eq!(response.content, Some("Hello World".to_string()));
}

#[rstest]
#[case::isaac(
    "isaac-test",
    Some(
        "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
    ),
    "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate."
)]
#[case::qwen("qwen3-vl-72b", None, "qwenvl markdown")]
#[tokio::test]
async fn markdown_mode(#[case] model: &str, #[case] expected_system: Option<&str>, #[case] expected_text: &str) {
    let (server, client) = common::setup().await;

    let mut messages = vec![];
    if let Some(system) = expected_system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": [
        {"type": "image_url"},
        {"type": "text", "text": expected_text}
    ]}));

    common::mock_response(
        &server,
        body_partial_json(json!({"messages": messages})),
        common::response("# Hello\n\nWorld", None),
    )
    .await;

    let request = OcrRequest::new(model, Media::image_url("https://example.com/doc.jpg")).mode(OcrMode::Markdown);
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

    let request = OcrRequest::new("isaac-test", Media::image_url("https://example.com/doc.jpg")).mode(OcrMode::Html);
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

    let request = OcrRequest::new("isaac-test", Media::base64(MediaFormat::Webp, "docdata"));
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

    let request = OcrRequest::new("isaac-test", Media::image_url("https://example.com/doc.jpg")).reasoning(true);
    let response = client.ocr(request).await.unwrap();
    assert_eq!(response.content, Some("Hello".to_string()));
    assert_eq!(response.reasoning, Some("I can see text".to_string()));
}

#[tokio::test]
async fn custom_prompt() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Extract only the dates"}
                ]}
            ]
        })),
        common::response("2024-01-15", None),
    )
    .await;

    let request = OcrRequest::new("isaac-test", Media::image_url("https://example.com/doc.jpg"))
        .prompt("Extract only the dates");
    let response = client.ocr(request).await.unwrap();
    assert_eq!(response.content, Some("2024-01-15".to_string()));
}
