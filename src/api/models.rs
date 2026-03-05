use serde::Deserialize;

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
    pub modalities: Vec<Modality>,
    pub output_formats: Vec<OutputFormat>,
    pub sampling_parameters: Vec<SamplingParameter>,
    pub max_context_tokens: u64,
    pub max_output_tokens: u64,
}

impl From<ModelResponse> for Model {
    fn from(response: ModelResponse) -> Self {
        Self {
            id: response.id,
            name: response.name,
            modalities: response.modalities,
            output_formats: response.output_formats,
            sampling_parameters: response.sampling_parameters,
            max_context_tokens: response.max_context_tokens,
            max_output_tokens: response.max_output_tokens,
        }
    }
}
