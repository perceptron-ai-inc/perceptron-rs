use perceptron_ai::{
    AnalyzeRequest, BoundingBox, Media, MediaFormat, OutputFormat, Perceptron, Point, Pointing, Polygon,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, ResponseTemplate};

mod common;

fn test_request(model: &str) -> AnalyzeRequest {
    AnalyzeRequest::new(model, "Describe this", Media::image_url("https://example.com/img.jpg"))
}

#[tokio::test]
async fn complete_fails() {
    let (server, client) = common::setup().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": {
                "message": "internal server error",
                "type": "server_error"
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let result = client.analyze(test_request("test-model")).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("internal server error"));
}

#[tokio::test]
async fn empty_choices() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"model": "test-model"})),
        json!({"choices": []}),
    )
    .await;

    let response = client.analyze(test_request("test-model")).await.unwrap();

    assert_eq!(response.content, None);
    assert_eq!(response.reasoning, None);
    assert_eq!(response.pointing, None);
}

#[tokio::test]
async fn text_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"model": "test-model"})),
        common::response("a cat", Some("I see fur")),
    )
    .await;

    let response = client.analyze(test_request("test-model")).await.unwrap();

    assert_eq!(response.content, Some("a cat".to_string()));
    assert_eq!(response.reasoning, Some("I see fur".to_string()));
    assert_eq!(response.pointing, None);
}

#[tokio::test]
async fn point_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>POINT</hint>"}]})),
        common::response(r#"<point mention="cat"> (100,200) </point>"#, None),
    )
    .await;
    let request = test_request("test-model").output_format(OutputFormat::Point);
    let response = client.analyze(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Points(vec![Point {
            x: 100,
            y: 200,
            mention: Some("cat".to_string()),
        }]))
    );
}

#[tokio::test]
async fn box_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>BOX</hint>"}]})),
        common::response(r#"<point_box mention="cat"> (10,20) (100,200) </point_box>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Box);
    let response = client.analyze(request).await.unwrap();

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
async fn polygon_format() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>POLYGON</hint>"}]})),
        common::response(r#"<polygon mention="cat"> (0,0) (100,0) (100,100) </polygon>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Polygon);
    let response = client.analyze(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(
        response.pointing,
        Some(Pointing::Polygons(vec![Polygon {
            hull: vec![(0, 0), (100, 0), (100, 100)],
            mention: Some("cat".to_string()),
        }]))
    );
}

#[tokio::test]
async fn multiple_points() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>POINT</hint>"}]})),
        common::response(r#"<point mention="left eye"> (150,200) </point><point mention="right eye"> (250,200) </point><point mention="nose"> (200,280) </point>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Point);
    let response = client.analyze(request).await.unwrap();

    assert_eq!(
        response.pointing,
        Some(Pointing::Points(vec![
            Point {
                x: 150,
                y: 200,
                mention: Some("left eye".to_string())
            },
            Point {
                x: 250,
                y: 200,
                mention: Some("right eye".to_string())
            },
            Point {
                x: 200,
                y: 280,
                mention: Some("nose".to_string())
            },
        ]))
    );
}

#[tokio::test]
async fn multiple_boxes() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>BOX</hint>"}]})),
        common::response(r#"<point_box mention="cat"> (10,20) (100,200) </point_box><point_box mention="dog"> (300,50) (500,400) </point_box><point_box mention="bird"> (600,10) (700,80) </point_box>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Box);
    let response = client.analyze(request).await.unwrap();

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
            BoundingBox {
                x1: 600,
                y1: 10,
                x2: 700,
                y2: 80,
                mention: Some("bird".to_string())
            },
        ]))
    );
}

#[tokio::test]
async fn multiple_polygons() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>POLYGON</hint>"}]})),
        common::response(r#"<polygon mention="roof"> (100,50) (200,10) (300,50) </polygon><polygon mention="wall"> (100,50) (300,50) (300,200) (100,200) </polygon>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Polygon);
    let response = client.analyze(request).await.unwrap();

    assert_eq!(
        response.pointing,
        Some(Pointing::Polygons(vec![
            Polygon {
                hull: vec![(100, 50), (200, 10), (300, 50)],
                mention: Some("roof".to_string())
            },
            Polygon {
                hull: vec![(100, 50), (300, 50), (300, 200), (100, 200)],
                mention: Some("wall".to_string())
            },
        ]))
    );
}

#[tokio::test]
async fn collection_with_inheritance() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({"messages": [{"role": "system", "content": "<hint>POINT</hint>"}]})),
        common::response(r#"<collection mention="person"><point> (150,200) </point><point> (250,200) </point></collection><point mention="ball"> (500,400) </point>"#, None),
    )
    .await;

    let request = test_request("test-model").output_format(OutputFormat::Point);
    let response = client.analyze(request).await.unwrap();

    assert_eq!(
        response.pointing,
        Some(Pointing::Points(vec![
            Point {
                x: 150,
                y: 200,
                mention: Some("person".to_string())
            },
            Point {
                x: 250,
                y: 200,
                mention: Some("person".to_string())
            },
            Point {
                x: 500,
                y: 400,
                mention: Some("ball".to_string())
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
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
                ]
            }]
        })),
        common::response("a cat", None),
    )
    .await;

    let request = AnalyzeRequest::new("test-model", "Describe this", Media::base64(MediaFormat::Png, "abc123"));
    let response = client.analyze(request).await.unwrap();

    assert_eq!(response.content, Some("a cat".to_string()));
}

#[tokio::test]
async fn video_url_media() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}}
                ]
            }]
        })),
        common::response("a video of a cat", None),
    )
    .await;

    let request = AnalyzeRequest::new(
        "test-model",
        "Describe this",
        Media::video_url("https://example.com/vid.mp4"),
    );
    let response = client.analyze(request).await.unwrap();

    assert_eq!(response.content, Some("a video of a cat".to_string()));
}

#[tokio::test]
async fn all_generation_params() {
    let (server, client) = common::setup().await;
    common::mock_response(
        &server,
        body_partial_json(json!({
            "model": "test-model",
            "max_completion_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3
        })),
        common::response(r#"<point mention="cat"> (50,60) </point>"#, Some("I see fur")),
    )
    .await;

    let request = test_request("test-model")
        .output_format(OutputFormat::Point)
        .reasoning(true)
        .temperature(0.7)
        .top_p(0.9)
        .top_k(50)
        .frequency_penalty(0.5)
        .presence_penalty(0.3)
        .max_completion_tokens(100);
    let response = client.analyze(request).await.unwrap();

    assert!(response.content.is_some());
    assert_eq!(response.reasoning, Some("I see fur".to_string()));
    assert_eq!(
        response.pointing,
        Some(Pointing::Points(vec![Point {
            x: 50,
            y: 60,
            mention: Some("cat".to_string()),
        }]))
    );
}
