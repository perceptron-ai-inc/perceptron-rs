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
}

/// A polygon annotation from the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Polygon {
    /// Polygon hull as (x, y) pairs.
    pub hull: Vec<(u32, u32)>,
    /// Optional label.
    pub mention: Option<String>,
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
}
