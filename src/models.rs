use serde::{Deserialize, Serialize};

use crate::media::Modality;
use crate::types::OutputFormat;

/// A Perceptron model with metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Model {
    /// Model identifier (e.g. `"isaac-0.1"`)
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Description of the model.
    pub description: Option<String>,
    /// Input modalities the model accepts.
    pub modalities: Vec<Modality>,
    /// Output formats the model supports.
    pub output_formats: Vec<OutputFormat>,
    /// Sampling parameters the model accepts.
    pub sampling_parameters: Vec<SamplingParameter>,
    /// Maximum context window size in tokens.
    pub max_context_tokens: u64,
    /// Maximum output tokens the model can generate.
    pub max_output_tokens: u64,
}

/// Sampling parameter a model accepts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SamplingParameter {
    /// Temperature sampling.
    Temperature,
    /// Nucleus sampling probability.
    TopP,
    /// Top-k sampling.
    TopK,
    /// Frequency penalty.
    FrequencyPenalty,
    /// Presence penalty.
    PresencePenalty,
}
