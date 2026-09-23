use perceptron_ai::{Audio, AudioFormat, Image, ImageFormat, OutputFormat, Perceptron, QuestionRequest, Video};
use rstest::rstest;
use serde_json::json;
use wiremock::matchers::body_partial_json;

mod common;

#[rstest]
#[case::isaac("isaac-test", None)]
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
        Image::url("https://example.com/img.jpg"),
    );
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("The cat is orange".to_string()));
    assert_eq!(response.pointing, None);
}

#[tokio::test]
async fn grounded_sends_annotation_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "vision_config": {"annotation_format": "point"},
            "messages": [
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
        Image::url("https://example.com/img.jpg"),
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

    let request = QuestionRequest::new("isaac-test", "What is this?", Image::base64(ImageFormat::Png, "abc123"));
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("a cat".to_string()));
}

#[tokio::test]
async fn with_reasoning() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "vision_config": {"enable_thinking": true},
            "messages": [
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
        Image::url("https://example.com/img.jpg"),
    )
    .reasoning(true);
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("Three cats".to_string()));
    assert_eq!(response.reasoning, Some("I count the cats".to_string()));
}

#[tokio::test]
async fn audio_url_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": "https://example.com/clip.wav"}},
                {"type": "text", "text": "What is said?"}
            ]}]
        })),
        common::response("Hello there", None),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "What is said?",
        Audio::url("https://example.com/clip.wav"),
    );
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("Hello there".to_string()));
}

#[tokio::test]
async fn base64_audio_is_sent_as_input_audio() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}},
                {"type": "text", "text": "Transcribe."}
            ]}]
        })),
        common::response("hello", None),
    )
    .await;

    let request = QuestionRequest::new("isaac-test", "Transcribe.", Audio::base64(AudioFormat::Mp3, "AAAA"));
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("hello".to_string()));
}

#[tokio::test]
async fn enable_audio_in_video_is_sent_as_vision_config() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "vision_config": {"enable_audio_in_video": true},
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                {"type": "text", "text": "What is said in the clip?"}
            ]}]
        })),
        common::response("Someone says hi", None),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "What is said in the clip?",
        Video::url("https://example.com/vid.mp4"),
    )
    .enable_audio_in_video(true);
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("Someone says hi".to_string()));
}

#[tokio::test]
async fn reasoning_false_is_sent_as_enable_thinking_false() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"vision_config": {"enable_thinking": false}})),
        common::response("Three cats", None),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "How many cats?",
        Image::url("https://example.com/img.jpg"),
    )
    .reasoning(false);
    let response = client.question(request).await.unwrap();
    assert_eq!(response.content, Some("Three cats".to_string()));
    assert_eq!(response.reasoning, None);
}

#[tokio::test]
async fn all_vision_config_fields_are_sent_together() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "vision_config": {
                "enable_thinking": true,
                "annotation_format": "clip",
                "enable_audio_in_video": true
            }
        })),
        common::response(r#"<clip mention="dog barks" t="1 2"/>"#, Some("Listening")),
    )
    .await;

    let request = QuestionRequest::new(
        "isaac-test",
        "When does the dog bark?",
        Video::url("https://example.com/vid.mp4"),
    )
    .output_format(OutputFormat::Clip)
    .reasoning(true)
    .enable_audio_in_video(true);
    let response = client.question(request).await.unwrap();
    assert_eq!(response.reasoning, Some("Listening".to_string()));
    assert!(response.pointing.is_some());
}
