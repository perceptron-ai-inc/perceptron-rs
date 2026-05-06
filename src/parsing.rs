use regex::Regex;
use std::sync::LazyLock;

use crate::pointing::{BoundingBox, Clip, ClipTimestamp, Point, Pointing, Polygon};
use crate::types::OutputFormat;

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
static CLIP_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<clip\b\s*([^>]*?)\s*/>").expect(REGEX_EXPECT));

static COORD_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)").expect(REGEX_EXPECT));

static MENTION_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"mention="([^"]*)""#).expect(REGEX_EXPECT));

static T_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\bt=(?:"([^"]*)"|(\S+))"#).expect(REGEX_EXPECT));

fn parse_mention(attr_str: &str) -> Option<String> {
    MENTION_REGEX.captures(attr_str).map(|c| c[1].to_string())
}

fn parse_t(attr_str: &str) -> Option<f32> {
    T_REGEX.captures(attr_str).and_then(|c| {
        let value = c.get(1).or(c.get(2))?.as_str();
        value.split_whitespace().next()?.parse::<f32>().ok()
    })
}

/// Parse the `t` attribute on a `<clip />` tag, which may be a single moment or a range.
/// Accepts: `t=1.5`, `t="1.5"`, `t="1.5 seconds"`, `t="1.5 2.0"`, `t="1.5 seconds 2.0 seconds"`.
fn parse_clip_t(attr_str: &str) -> Option<ClipTimestamp> {
    let value = T_REGEX
        .captures(attr_str)
        .and_then(|c| c.get(1).or(c.get(2)))?
        .as_str();
    let nums: Vec<f32> = value
        .split_whitespace()
        .filter_map(|s| s.parse::<f32>().ok())
        .collect();
    match nums.as_slice() {
        [start] => Some(ClipTimestamp::Moment(*start)),
        [start, end] => Some(ClipTimestamp::Range {
            start: *start,
            end: *end,
        }),
        _ => None,
    }
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
/// Returns `None` when no format is requested (text-only response).
pub(crate) fn extract(text: &str, format: Option<&OutputFormat>) -> Option<Pointing> {
    let format = format?;
    let mut pointing = Pointing::default();
    match format {
        OutputFormat::Point => pointing.points = extract_items(text, &POINT_REGEX, parse_point),
        OutputFormat::Box => pointing.boxes = extract_items(text, &BOX_REGEX, parse_box),
        OutputFormat::Polygon => pointing.polygons = extract_items(text, &POLYGON_REGEX, parse_polygon),
        OutputFormat::Clip => pointing.clips = extract_clips(text),
    }
    // Omit pointing entirely when nothing was extracted, following the API
    // convention of absent fields rather than empty arrays.
    if pointing == Pointing::default() {
        None
    } else {
        Some(pointing)
    }
}

fn parse_point(coords: &[(u32, u32)], mention: Option<String>, timestamp: Option<f32>) -> Option<Point> {
    coords.first().map(|&(x, y)| Point { x, y, mention, timestamp })
}

fn parse_box(coords: &[(u32, u32)], mention: Option<String>, timestamp: Option<f32>) -> Option<BoundingBox> {
    if coords.len() >= 2 {
        Some(BoundingBox {
            x1: coords[0].0,
            y1: coords[0].1,
            x2: coords[1].0,
            y2: coords[1].1,
            mention,
            timestamp,
        })
    } else {
        None
    }
}

fn parse_polygon(coords: &[(u32, u32)], mention: Option<String>, timestamp: Option<f32>) -> Option<Polygon> {
    if coords.len() >= 3 {
        Some(Polygon {
            hull: coords.to_vec(),
            mention,
            timestamp,
        })
    } else {
        None
    }
}

/// Extract self-closing `<clip />` tags. Clips have no coordinates, just `mention` and `t` attrs.
fn extract_clips(text: &str) -> Vec<Clip> {
    let mut results = Vec::new();

    let remaining = COLLECTION_REGEX.replace_all(text, |cap: &regex::Captures| {
        let parent_mention = parse_mention(&cap[1]);
        for inner_cap in CLIP_REGEX.captures_iter(&cap[2]) {
            let mention = parse_mention(&inner_cap[1]).or(parent_mention.clone());
            if let Some(timestamp) = parse_clip_t(&inner_cap[1]) {
                results.push(Clip { mention, timestamp });
            }
        }
        ""
    });

    for cap in CLIP_REGEX.captures_iter(&remaining) {
        let mention = parse_mention(&cap[1]);
        if let Some(timestamp) = parse_clip_t(&cap[1]) {
            results.push(Clip { mention, timestamp });
        }
    }

    results
}

/// Extract items of the target tag type, flattening collections.
fn extract_items<T>(
    text: &str,
    target_regex: &Regex,
    parse_fn: fn(&[(u32, u32)], Option<String>, Option<f32>) -> Option<T>,
) -> Vec<T> {
    let mut results = Vec::new();

    // Process collections and strip them from the text
    let remaining = COLLECTION_REGEX.replace_all(text, |cap: &regex::Captures| {
        let parent_mention = parse_mention(&cap[1]);
        for inner_cap in target_regex.captures_iter(&cap[2]) {
            let child_mention = parse_mention(&inner_cap[1]);
            let mention = child_mention.or(parent_mention.clone());
            let timestamp = parse_t(&inner_cap[1]);
            if let Some(item) = parse_fn(&parse_coords(&inner_cap[2]), mention, timestamp) {
                results.push(item);
            }
        }
        "" // strip collection from text
    });

    // Find standalone items in the remaining text
    for cap in target_regex.captures_iter(&remaining) {
        if let Some(item) = parse_fn(&parse_coords(&cap[2]), parse_mention(&cap[1]), parse_t(&cap[1])) {
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
        let result = extract(text, Some(&OutputFormat::Point));
        assert_eq!(
            result,
            Some(Pointing {
                points: vec![Point {
                    x: 100,
                    y: 200,
                    mention: Some("target".to_string()),
                    timestamp: None,
                }],
                ..Default::default()
            })
        );
    }

    #[test]
    fn extract_single_box() {
        let text = r#"<point_box mention="cat" t=0.95> (10,20) (100,200) </point_box>"#;
        let result = extract(text, Some(&OutputFormat::Box));
        assert_eq!(
            result,
            Some(Pointing {
                boxes: vec![BoundingBox {
                    x1: 10,
                    y1: 20,
                    x2: 100,
                    y2: 200,
                    mention: Some("cat".to_string()),
                    timestamp: Some(0.95),
                }],
                ..Default::default()
            })
        );
    }

    #[test]
    fn extract_single_polygon() {
        let text = r#"<polygon mention="triangle"> (0,0) (100,0) (100,100) </polygon>"#;
        let result = extract(text, Some(&OutputFormat::Polygon));
        assert_eq!(
            result,
            Some(Pointing {
                polygons: vec![Polygon {
                    hull: vec![(0, 0), (100, 0), (100, 100)],
                    mention: Some("triangle".to_string()),
                    timestamp: None,
                }],
                ..Default::default()
            })
        );
    }

    #[test]
    fn extract_collection_with_inheritance() {
        let text = r#"<collection mention="cat" t=0.85>
            <point_box> (10,20) (30,40) </point_box>
            <point_box mention="explicit"> (50,60) (70,80) </point_box>
        </collection>"#;
        let result = extract(text, Some(&OutputFormat::Box));
        let boxes = &result.expect("expected Some(Pointing)").boxes;
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
        let result = extract(text, Some(&OutputFormat::Box));
        let boxes = &result.expect("expected Some(Pointing)").boxes;
        assert_eq!(boxes.len(), 3);
        // Collections are processed first, then standalone items
        assert_eq!(boxes[0].mention, Some("cat".to_string()));
        assert_eq!(boxes[1].mention, Some("dog".to_string()));
        assert_eq!(boxes[2].mention, Some("bird".to_string()));
    }

    #[test]
    fn extract_no_format_returns_none() {
        let text = r#"<point_box> (10,20) (30,40) </point_box>"#;
        assert_eq!(extract(text, None), None);
    }

    #[test]
    fn extract_invalid_coords_skipped() {
        let text = r#"<point> no coords here </point>"#;
        assert_eq!(extract(text, Some(&OutputFormat::Point)), None);
    }

    #[test]
    fn extract_clip_all_timestamp_forms() {
        let text = r#"
            <clip mention="intro" t=1.5/>
            <clip mention="outro" t="2.5 seconds"/>
            <clip mention="action" t="10 20"/>
            <clip mention="scene" t="30 seconds 45 seconds"/>
        "#;
        let result = extract(text, Some(&OutputFormat::Clip));
        let clips = &result.expect("expected Some(Pointing)").clips;
        assert_eq!(clips.len(), 4);
        assert_eq!(clips[0].mention.as_deref(), Some("intro"));
        assert_eq!(clips[0].timestamp, ClipTimestamp::Moment(1.5));
        assert_eq!(clips[1].timestamp, ClipTimestamp::Moment(2.5));
        assert_eq!(
            clips[2].timestamp,
            ClipTimestamp::Range { start: 10.0, end: 20.0 }
        );
        assert_eq!(
            clips[3].timestamp,
            ClipTimestamp::Range { start: 30.0, end: 45.0 }
        );
    }

    #[test]
    fn extract_multiple_points() {
        let text = r#"
            <point> (10,20) </point>
            <point mention="a"> (30,40) </point>
            <point t=0.5> (50,60) </point>
        "#;
        let result = extract(text, Some(&OutputFormat::Point));
        let points = &result.expect("expected Some(Pointing)").points;
        assert_eq!(points.len(), 3);
        assert_eq!(
            points[0],
            Point {
                x: 10,
                y: 20,
                mention: None,
                timestamp: None,
            }
        );
        assert_eq!(
            points[1],
            Point {
                x: 30,
                y: 40,
                mention: Some("a".to_string()),
                timestamp: None,
            }
        );
        assert_eq!(
            points[2],
            Point {
                x: 50,
                y: 60,
                mention: None,
                timestamp: Some(0.5),
            }
        );
    }
}
