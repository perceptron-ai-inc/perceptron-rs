use perceptron_ai::{
    AnalyzeRequest, CaptionRequest, CaptionStyle, DetectRequest, Media, MediaFormat, Modality, Model, OcrMode,
    OcrRequest, OutputFormat, Point, Pointing, PointingResponse, SamplingParameter, TextResponse,
};
use serde_json::json;

fn roundtrip<T>(value: &T, expected: serde_json::Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let serialized = serde_json::to_value(value).expect("serialize failed");
    assert_eq!(serialized, expected, "serialized JSON mismatch");
    let deserialized: T = serde_json::from_value(expected).expect("deserialize failed");
    assert_eq!(&deserialized, value, "round-trip mismatch");
}

// --- Requests ---

#[test]
fn analyze_request_all_fields() {
    roundtrip(
        &AnalyzeRequest::new(
            "model-v1",
            "Describe this",
            Media::image_url("https://example.com/img.jpg"),
        )
        .output_format(OutputFormat::Point)
        .reasoning(true)
        .temperature(0.5)
        .top_p(0.25)
        .top_k(50)
        .frequency_penalty(0.5)
        .presence_penalty(0.125)
        .max_completion_tokens(100),
        json!({
            "message": "Describe this",
            "media": {"type": "url", "modality": "image", "src": "https://example.com/img.jpg"},
            "output_format": "point",
            "model": "model-v1",
            "reasoning": true,
            "temperature": 0.5,
            "top_p": 0.25,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.125,
            "max_completion_tokens": 100
        }),
    );
}

#[test]
fn caption_request_all_fields() {
    roundtrip(
        &CaptionRequest::new("model-v1", Media::base64(MediaFormat::Jpeg, "data"))
            .style(CaptionStyle::Detailed)
            .output_format(OutputFormat::Box)
            .reasoning(true)
            .temperature(0.5)
            .top_p(0.25)
            .top_k(50)
            .frequency_penalty(0.5)
            .presence_penalty(0.125)
            .max_completion_tokens(100),
        json!({
            "media": {"type": "base64", "format": "jpeg", "data": "data"},
            "style": "detailed",
            "output_format": "box",
            "model": "model-v1",
            "reasoning": true,
            "temperature": 0.5,
            "top_p": 0.25,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.125,
            "max_completion_tokens": 100
        }),
    );
}

#[test]
fn ocr_request_all_fields() {
    roundtrip(
        &OcrRequest::new("model-v1", Media::image_url("https://example.com/doc.jpg"))
            .mode(OcrMode::Markdown)
            .reasoning(true)
            .temperature(0.5)
            .top_p(0.25)
            .top_k(50)
            .frequency_penalty(0.5)
            .presence_penalty(0.125)
            .max_completion_tokens(100),
        json!({
            "media": {"type": "url", "modality": "image", "src": "https://example.com/doc.jpg"},
            "mode": "markdown",
            "model": "model-v1",
            "reasoning": true,
            "temperature": 0.5,
            "top_p": 0.25,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.125,
            "max_completion_tokens": 100
        }),
    );
}

#[test]
fn detect_request_all_fields() {
    roundtrip(
        &DetectRequest::new("model-v1", Media::video_url("https://example.com/vid.mp4"))
            .classes(vec!["cat".to_string(), "dog".to_string()])
            .reasoning(true)
            .temperature(0.5)
            .top_p(0.25)
            .top_k(50)
            .frequency_penalty(0.5)
            .presence_penalty(0.125)
            .max_completion_tokens(100),
        json!({
            "media": {"type": "url", "modality": "video", "src": "https://example.com/vid.mp4"},
            "classes": ["cat", "dog"],
            "model": "model-v1",
            "reasoning": true,
            "temperature": 0.5,
            "top_p": 0.25,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.125,
            "max_completion_tokens": 100
        }),
    );
}

// --- Models ---

#[test]
fn model_all_fields() {
    roundtrip(
        &Model {
            id: "isaac-0.1".to_string(),
            name: "Isaac".to_string(),
            modalities: vec![Modality::Image, Modality::Video],
            output_formats: vec![
                OutputFormat::Text,
                OutputFormat::Point,
                OutputFormat::Box,
                OutputFormat::Polygon,
            ],
            sampling_parameters: vec![SamplingParameter::Temperature, SamplingParameter::TopP],
            max_context_tokens: 128000,
            max_output_tokens: 4096,
        },
        json!({
            "id": "isaac-0.1",
            "name": "Isaac",
            "modalities": ["image", "video"],
            "output_formats": ["text", "point", "box", "polygon"],
            "sampling_parameters": ["temperature", "top_p"],
            "max_context_tokens": 128000,
            "max_output_tokens": 4096
        }),
    );
}

// --- Responses ---

#[test]
fn text_response() {
    roundtrip(
        &TextResponse {
            content: Some("hello".to_string()),
            reasoning: Some("thinking".to_string()),
        },
        json!({"content": "hello", "reasoning": "thinking"}),
    );
}

#[test]
fn pointing_response() {
    roundtrip(
        &PointingResponse {
            content: Some("a cat".to_string()),
            reasoning: Some("I see fur".to_string()),
            pointing: Some(Pointing {
                points: vec![Point {
                    x: 50,
                    y: 60,
                    mention: Some("cat".to_string()),
                }],
                ..Default::default()
            }),
        },
        json!({
            "content": "a cat",
            "reasoning": "I see fur",
            "pointing": {"points": [{"x": 50, "y": 60, "mention": "cat"}]}
        }),
    );
}
