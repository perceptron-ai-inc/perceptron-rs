mod common;

use perceptron_ai::{Modality, Model, OutputFormat, Perceptron, SamplingParameter};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

fn sample_model(id: &str, name: &str) -> serde_json::Value {
    json!({
        "id": id,
        "name": name,
        "modalities": ["image", "video"],
        "output_formats": ["text", "point", "box", "polygon"],
        "sampling_parameters": ["temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"],
        "max_context_tokens": 128000,
        "max_output_tokens": 4096
    })
}

fn assert_model(model: &Model, expected_id: &str, expected_name: &str) {
    assert_eq!(model.id, expected_id);
    assert_eq!(model.name, expected_name);
    assert_eq!(model.modalities, vec![Modality::Image, Modality::Video]);
    assert_eq!(
        model.output_formats,
        vec![
            OutputFormat::Text,
            OutputFormat::Point,
            OutputFormat::Box,
            OutputFormat::Polygon
        ]
    );
    assert_eq!(
        model.sampling_parameters,
        vec![
            SamplingParameter::Temperature,
            SamplingParameter::TopP,
            SamplingParameter::TopK,
            SamplingParameter::FrequencyPenalty,
            SamplingParameter::PresencePenalty,
        ]
    );
    assert_eq!(model.max_context_tokens, 128000);
    assert_eq!(model.max_output_tokens, 4096);
}

#[tokio::test]
async fn list_models() {
    let (server, client) = common::setup().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [sample_model("isaac-0.1", "Isaac"), sample_model("isaac-0.2", "Isaac 2")]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let models = client.models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_model(&models[0], "isaac-0.1", "Isaac");
    assert_model(&models[1], "isaac-0.2", "Isaac 2");
}

#[tokio::test]
async fn get_single_model() {
    let (server, client) = common::setup().await;

    Mock::given(method("GET"))
        .and(path("/v1/models/isaac-0.1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sample_model("isaac-0.1", "Isaac")))
        .expect(1)
        .mount(&server)
        .await;

    let model = client.model("isaac-0.1").await.unwrap();
    assert_model(&model, "isaac-0.1", "Isaac");
}

#[tokio::test]
async fn model_not_found() {
    let (server, client) = common::setup().await;

    Mock::given(method("GET"))
        .and(path("/v1/models/nonexistent"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {
                "message": "Model not found: nonexistent",
                "type": "not_found_error"
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let err = client.model("nonexistent").await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("404"), "expected 404 in error: {msg}");
    assert!(msg.contains("Model not found"), "expected detail in error: {msg}");
}
