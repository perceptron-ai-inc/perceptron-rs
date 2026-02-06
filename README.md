# Perceptron Rust SDK

Rust SDK for Perceptron AI models.

Get an API key at https://platform.perceptron.inc

## Usage

```rust
use perceptron::{PerceptronClient, AnalyzeImageRequest, OutputFormat};

#[tokio::main]
async fn main() -> Result<(), perceptron::PerceptronError> {
    let client = PerceptronClient::new()
        .api_key("my-api-key");

    let request = AnalyzeImageRequest::new(
        "model-name",
        "Describe this image",
        "https://example.com/image.jpg",
    )
    .output_format(OutputFormat::Point)
    .reasoning(true)
    .temperature(0.7);

    let response = client.analyze_image(request).await?;

    if let Some(content) = response.content {
        println!("{content}");
    }

    Ok(())
}
```

## On-device deployment

For models running locally, set a custom base URL:

```rust
let client = PerceptronClient::new()
    .base_url("http://localhost:8080");
```