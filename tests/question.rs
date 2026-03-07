use perceptron_ai::{Media, MediaFormat, OutputFormat, Perceptron, QuestionRequest};
use rstest::rstest;
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

#[rstest]
#[case::isaac("isaac-test", None)]
#[case::qwen("qwen3-vl-72b", None)]
#[case::unknown_defaults_to_isaac("unknown-model", None)]
#[tokio::test]
async fn plain(#[case] model: &str, #[case] expected_system: Option<&str>) {
    let (server, client) = common::setup().await;

    let mut messages = vec![];
    if let Some(system) = expected_system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        {"type": "text", "text": "What color is the cat?"}
    ]}));

    common::mock_response(
        &server,
        body_partial_json(json!({"model": model, "messages": messages})),
        common::response("The cat is orange", None),
    )
    .await;

    let request = QuestionRequest::new(
        model,
        "What color is the cat?",
        Media::image_url("https://example.com/img.jpg"),
    );
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("The cat is orange".to_string()));
    assert_eq!(response.pointing, None);
}

#[tokio::test]
async fn with_grounded_output() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": "qwen3-vl-72b",
            "messages": [
                {"role": "system", "content": "<hint>BOX</hint>"},
                {"role": "system", "content": "You are Qwen3-VL performing grounded reasoning. Give the answer and reference the relevant regions using structured tags when available. Report bbox coordinates in JSON format."},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Where is the cat?"}
                ]}
            ]
        })),
        common::response(r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#, None),
    )
    .await;

    let request = QuestionRequest::new(
        "qwen3-vl-72b",
        "Where is the cat?",
        Media::image_url("https://example.com/img.jpg"),
    )
    .output_format(OutputFormat::Box);
    let response = client.question(request).await.unwrap();
    assert!(response.content.is_some());
    assert!(response.pointing.is_some());
}

#[tokio::test]
async fn isaac_grounded_no_system() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>POINT</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "Where is the cat?"}
                ]}
            ]
        })),
        common::response(r#"<point mention="cat"> (50,60) </point>"#, None),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "Where is the cat?",
        Media::image_url("https://example.com/img.jpg"),
    )
    .output_format(OutputFormat::Point);
    let response = client.question(request).await.unwrap();
    assert!(response.pointing.is_some());
}

#[tokio::test]
async fn base64_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    {"type": "text", "text": "What is this?"}
                ]
            }]
        })),
        common::response("a cat", None),
    )
    .await;

    let request = QuestionRequest::new("isaac-test", "What is this?", Media::base64(MediaFormat::Png, "abc123"));
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("a cat".to_string()));
}

#[tokio::test]
async fn with_reasoning() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [
                {"role": "system", "content": "<hint>THINK</hint>"},
                {"role": "user", "content": [
                    {"type": "image_url"},
                    {"type": "text", "text": "How many cats?"}
                ]}
            ]
        })),
        common::response("Three cats", Some("I count the cats")),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "How many cats?",
        Media::image_url("https://example.com/img.jpg"),
    )
    .reasoning(true);
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("Three cats".to_string()));
    assert_eq!(response.reasoning, Some("I count the cats".to_string()));
}
