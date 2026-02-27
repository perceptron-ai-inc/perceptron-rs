use regex::Regex;
use std::sync::LazyLock;

use crate::types::{BoundingBox, OutputFormat, Point, Pointing, Polygon};

const REGEX_EXPECT: &str = "regex creation should never fail here";

/// Build a regex that matches a specific XML-like tag by name.
fn tag_regex(tag_name: &str) -> Regex {
    Regex::new(&format!(r"(?i)<{tag_name}([^>]*)>([\s\S]*?)</{tag_name}>")).expect(REGEX_EXPECT)
}

// Compile regexes once on first use to avoid recompilation on every call.
static POINT_REGEX: LazyLock<Regex> = LazyLock::new(|| tag_regex("point"));
static BOX_REGEX: LazyLock<Regex> = LazyLock::new(|| tag_regex("point_box"));
static POLYGON_REGEX: LazyLock<Regex> = LazyLock::new(|| tag_regex("polygon"));
static COLLECTION_REGEX: LazyLock<Regex> = LazyLock::new(|| tag_regex("collection"));

static COORD_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)").expect(REGEX_EXPECT));

static MENTION_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"mention="([^"]*)""#).expect(REGEX_EXPECT));

/// Return the compiled regex for the target tag type.
fn item_regex(format: &OutputFormat) -> &'static Regex {
    match format {
        OutputFormat::Point => &POINT_REGEX,
        OutputFormat::Box => &BOX_REGEX,
        OutputFormat::Polygon => &POLYGON_REGEX,
        OutputFormat::Text => unreachable!(),
    }
}

fn parse_mention(attr_str: &str) -> Option<String> {
    MENTION_REGEX.captures(attr_str).map(|c| c[1].to_string())
}

fn parse_coords(body: &str) -> Vec<(u32, u32)> {
    COORD_REGEX
        .captures_iter(body)
        .filter_map(|c| {
            let x = c[1].parse::<u32>().ok()?;
            let y = c[2].parse::<u32>().ok()?;
            Some((x, y))
        })
        .collect()
}

/// Extract annotations from model output text based on the output format.
pub(crate) fn extract(text: &str, format: &OutputFormat) -> Option<Pointing> {
    match format {
        OutputFormat::Point => {
            let points = extract_items(text, item_regex(format), parse_point);
            if points.is_empty() {
                None
            } else {
                Some(Pointing::Points(points))
            }
        }
        OutputFormat::Box => {
            let boxes = extract_items(text, item_regex(format), parse_box);
            if boxes.is_empty() {
                None
            } else {
                Some(Pointing::Boxes(boxes))
            }
        }
        OutputFormat::Polygon => {
            let polygons = extract_items(text, item_regex(format), parse_polygon);
            if polygons.is_empty() {
                None
            } else {
                Some(Pointing::Polygons(polygons))
            }
        }
        OutputFormat::Text => None,
    }
}

fn parse_point(coords: &[(u32, u32)], mention: Option<String>) -> Option<Point> {
    coords.first().map(|&(x, y)| Point { x, y, mention })
}

fn parse_box(coords: &[(u32, u32)], mention: Option<String>) -> Option<BoundingBox> {
    if coords.len() >= 2 {
        Some(BoundingBox {
            x1: coords[0].0,
            y1: coords[0].1,
            x2: coords[1].0,
            y2: coords[1].1,
            mention,
        })
    } else {
        None
    }
}

fn parse_polygon(coords: &[(u32, u32)], mention: Option<String>) -> Option<Polygon> {
    if coords.len() >= 3 {
        Some(Polygon {
            hull: coords.to_vec(),
            mention,
        })
    } else {
        None
    }
}

/// Extract items of the target tag type, flattening collections.
fn extract_items<T>(
    text: &str,
    target_regex: &Regex,
    parse_fn: fn(&[(u32, u32)], Option<String>) -> Option<T>,
) -> Vec<T> {
    let mut results = Vec::new();

    // Process collections and strip them from the text
    let remaining = COLLECTION_REGEX.replace_all(text, |cap: &regex::Captures| {
        let parent_mention = parse_mention(&cap[1]);
        for inner_cap in target_regex.captures_iter(&cap[2]) {
            let child_mention = parse_mention(&inner_cap[1]);
            let mention = child_mention.or(parent_mention.clone());
            if let Some(item) = parse_fn(&parse_coords(&inner_cap[2]), mention) {
                results.push(item);
            }
        }
        "" // strip collection from text
    });

    // Find standalone items in the remaining text
    for cap in target_regex.captures_iter(&remaining) {
        if let Some(item) = parse_fn(&parse_coords(&cap[2]), parse_mention(&cap[1])) {
            results.push(item);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_single_point() {
        let text = r#"<point mention="target"> (100,200) </point>"#;
        let result = extract(text, &OutputFormat::Point);
        assert_eq!(
            result,
            Some(Pointing::Points(vec![Point {
                x: 100,
                y: 200,
                mention: Some("target".to_string()),
            }]))
        );
    }

    #[test]
    fn extract_single_box() {
        let text = r#"<point_box mention="cat" t=0.95> (10,20) (100,200) </point_box>"#;
        let result = extract(text, &OutputFormat::Box);
        assert_eq!(
            result,
            Some(Pointing::Boxes(vec![BoundingBox {
                x1: 10,
                y1: 20,
                x2: 100,
                y2: 200,
                mention: Some("cat".to_string()),
            }]))
        );
    }

    #[test]
    fn extract_single_polygon() {
        let text = r#"<polygon mention="triangle"> (0,0) (100,0) (100,100) </polygon>"#;
        let result = extract(text, &OutputFormat::Polygon);
        assert_eq!(
            result,
            Some(Pointing::Polygons(vec![Polygon {
                hull: vec![(0, 0), (100, 0), (100, 100)],
                mention: Some("triangle".to_string()),
            }]))
        );
    }

    #[test]
    fn extract_collection_with_inheritance() {
        let text = r#"<collection mention="cat" t=0.85>
            <point_box> (10,20) (30,40) </point_box>
            <point_box mention="explicit"> (50,60) (70,80) </point_box>
        </collection>"#;
        let result = extract(text, &OutputFormat::Box);
        let boxes = match result {
            Some(Pointing::Boxes(b)) => b,
            other => panic!("Expected Pointing::Boxes, got {:?}", other),
        };
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].mention, Some("cat".to_string()));
        // Child with explicit mention keeps its own
        assert_eq!(boxes[1].mention, Some("explicit".to_string()));
    }

    #[test]
    fn extract_mixed_standalone_and_collection() {
        let text = r#"
            <point_box mention="dog"> (1,2) (3,4) </point_box>
            <collection mention="cat">
                <point_box> (10,20) (30,40) </point_box>
            </collection>
            <point_box mention="bird"> (50,60) (70,80) </point_box>
        "#;
        let result = extract(text, &OutputFormat::Box);
        let boxes = match result {
            Some(Pointing::Boxes(b)) => b,
            other => panic!("Expected Pointing::Boxes, got {:?}", other),
        };
        assert_eq!(boxes.len(), 3);
        // Collections are processed first, then standalone items
        assert_eq!(boxes[0].mention, Some("cat".to_string()));
        assert_eq!(boxes[1].mention, Some("dog".to_string()));
        assert_eq!(boxes[2].mention, Some("bird".to_string()));
    }

    #[test]
    fn extract_text_format_returns_none() {
        let text = r#"<point_box> (10,20) (30,40) </point_box>"#;
        assert_eq!(extract(text, &OutputFormat::Text), None);
    }

    #[test]
    fn extract_invalid_coords_skipped() {
        let text = r#"<point> no coords here </point>"#;
        assert_eq!(extract(text, &OutputFormat::Point), None);
    }

    #[test]
    fn extract_multiple_points() {
        let text = r#"
            <point> (10,20) </point>
            <point mention="a"> (30,40) </point>
            <point t=0.5> (50,60) </point>
        "#;
        let result = extract(text, &OutputFormat::Point);
        let points = match result {
            Some(Pointing::Points(p)) => p,
            other => panic!("Expected Pointing::Points, got {:?}", other),
        };
        assert_eq!(points.len(), 3);
        assert_eq!(
            points[0],
            Point {
                x: 10,
                y: 20,
                mention: None
            }
        );
        assert_eq!(
            points[1],
            Point {
                x: 30,
                y: 40,
                mention: Some("a".to_string())
            }
        );
        assert_eq!(
            points[2],
            Point {
                x: 50,
                y: 60,
                mention: None
            }
        );
    }
}
