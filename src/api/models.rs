use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};

use crate::media::Modality;
use crate::models::{Model, SamplingParameter};
use crate::types::OutputFormat;

#[derive(Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelResponse>,
}

#[derive(Deserialize)]
pub struct ModelResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    #[serde(deserialize_with = "known_values")]
    pub modalities: Vec<Modality>,
    #[serde(deserialize_with = "known_values")]
    pub output_formats: Vec<OutputFormat>,
    #[serde(deserialize_with = "known_values")]
    pub sampling_parameters: Vec<SamplingParameter>,
    pub max_context_tokens: u64,
    pub max_output_tokens: u64,
}

/// Keeps the values this crate version names and skips the rest, so a value the API adds later does
/// not fail the whole listing.
fn known_values<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let values = Vec::<serde_json::Value>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .filter_map(|value| serde_json::from_value(value).ok())
        .collect())
}

impl From<ModelResponse> for Model {
    fn from(response: ModelResponse) -> Self {
        Self {
            id: response.id,
            name: response.name,
            description: response.description,
            modalities: response.modalities,
            output_formats: response.output_formats,
            sampling_parameters: response.sampling_parameters,
            max_context_tokens: response.max_context_tokens,
            max_output_tokens: response.max_output_tokens,
        }
    }
}
