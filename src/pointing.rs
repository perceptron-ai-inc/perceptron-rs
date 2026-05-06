use serde::{Deserialize, Serialize};

/// A point annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Point {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
    /// Optional label.
    pub mention: Option<String>,
    /// Optional timestamp in seconds (for video annotations).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timestamp: Option<f32>,
}

/// A bounding box annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BoundingBox {
    /// Top-left X coordinate.
    pub x1: u32,
    /// Top-left Y coordinate.
    pub y1: u32,
    /// Bottom-right X coordinate.
    pub x2: u32,
    /// Bottom-right Y coordinate.
    pub y2: u32,
    /// Optional label.
    pub mention: Option<String>,
    /// Optional timestamp in seconds (for video annotations).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timestamp: Option<f32>,
}

/// A polygon annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Polygon {
    /// Polygon hull as (x, y) pairs.
    pub hull: Vec<(u32, u32)>,
    /// Optional label.
    pub mention: Option<String>,
    /// Optional timestamp in seconds (for video annotations).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timestamp: Option<f32>,
}

/// A video clip annotation from the model. Either a single moment or a time range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ClipTimestamp {
    /// A single instant in seconds.
    Moment(f32),
    /// A time range in seconds (start exclusive, end inclusive).
    Range {
        /// Start of the range, in seconds.
        start: f32,
        /// End of the range, in seconds.
        end: f32,
    },
}

/// A video clip annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Clip {
    /// Optional label.
    pub mention: Option<String>,
    /// Clip timestamp — either a single moment or a time range.
    pub timestamp: ClipTimestamp,
}

/// Pointing data extracted from model output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Pointing {
    /// Point annotations.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub points: Vec<Point>,
    /// Bounding box annotations.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub boxes: Vec<BoundingBox>,
    /// Polygon annotations.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub polygons: Vec<Polygon>,
    /// Clip annotations.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub clips: Vec<Clip>,
}
