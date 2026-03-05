use std::sync::LazyLock;

use crate::types::{CaptionStyle, OcrMode};

/// Prompt template for caption requests.
#[derive(Debug, Clone)]
pub struct CaptionPromptTemplate {
    /// Optional system instruction for the caption endpoint.
    pub system: Option<&'static str>,
    /// User text for concise captions.
    pub concise: &'static str,
    /// User text for detailed captions.
    pub detailed: &'static str,
}

impl CaptionPromptTemplate {
    /// Return the user text for the given caption style.
    pub fn user_text(&self, style: &CaptionStyle) -> &'static str {
        match style {
            CaptionStyle::Concise => self.concise,
            CaptionStyle::Detailed => self.detailed,
        }
    }
}

/// Prompt template for OCR requests.
#[derive(Debug, Clone)]
pub struct OcrPromptTemplate {
    /// Optional system instruction for the OCR endpoint.
    pub system: Option<&'static str>,
    /// User text for plain mode (None means no user text).
    pub plain: Option<&'static str>,
    /// User text for markdown mode.
    pub markdown: &'static str,
    /// User text for HTML mode.
    pub html: &'static str,
}

impl OcrPromptTemplate {
    /// Return the user text for the given OCR mode, or `None` for plain when the profile omits it.
    pub fn user_text(&self, mode: &OcrMode) -> Option<&'static str> {
        match mode {
            OcrMode::Plain => self.plain,
            OcrMode::Markdown => Some(self.markdown),
            OcrMode::Html => Some(self.html),
        }
    }
}

/// Prompt template for detect requests.
#[derive(Debug, Clone)]
pub struct DetectPromptTemplate {
    /// System text when no categories are specified.
    pub general: &'static str,
    /// Template with `{categories}` placeholder for category-specific detection.
    pub category_template: &'static str,
}

impl DetectPromptTemplate {
    /// Return the system text, substituting categories if provided.
    pub fn system_text(&self, categories: Option<&[String]>) -> String {
        match categories {
            Some(cats) if !cats.is_empty() => self.category_template.replace("{categories}", &cats.join(", ")),
            _ => self.general.to_string(),
        }
    }
}

/// A collection of prompt templates for a specific model family.
#[derive(Debug, Clone)]
pub struct PromptProfile {
    /// Identifier key for this profile.
    pub key: &'static str,
    /// Caption prompt template.
    pub caption: CaptionPromptTemplate,
    /// OCR prompt template.
    pub ocr: OcrPromptTemplate,
    /// Detect prompt template.
    pub detect: DetectPromptTemplate,
}

// ---------------------------------------------------------------------------
// Built-in profiles
// ---------------------------------------------------------------------------

const ISAAC: PromptProfile = PromptProfile {
    key: "isaac-default",
    caption: CaptionPromptTemplate {
        system: None,
        concise: "Provide a concise, human-friendly caption for the upcoming image.",
        detailed: "Provide a detailed caption describing key objects, relationships, and context in the upcoming image.",
    },
    ocr: OcrPromptTemplate {
        system: Some(
            "You are an OCR (Optical Character Recognition) system. \
            Accurately detect, extract, and transcribe all readable text from the image.",
        ),
        plain: None,
        markdown: "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate.",
        html: "Transcribe every readable word in the image using HTML markup.",
    },
    detect: DetectPromptTemplate {
        general: "Your goal is to segment out the objects in the scene",
        category_template: "Your goal is to segment out the following categories: {categories}",
    },
};

const QWEN: PromptProfile = PromptProfile {
    key: "qwen3-vl-235b-a22b-thinking",
    caption: CaptionPromptTemplate {
        system: None,
        concise: "Describe the primary subjects, their actions, and visible context in one vivid sentence.",
        detailed: "Provide a multi-sentence caption that calls out subjects, relationships, scene intent, and any text embedded in the image.",
    },
    ocr: OcrPromptTemplate {
        system: None,
        plain: Some("Read all the text in the image."),
        markdown: "qwenvl markdown",
        html: "qwenvl html",
    },
    detect: DetectPromptTemplate {
        general: "Locate every object of interest and report bounding box coordinates in JSON format.",
        category_template: "Locate every instance that belongs to the following categories: \"{categories}\". Report bbox coordinates in JSON format.",
    },
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

struct PromptProfileRegistry {
    profiles: Vec<&'static PromptProfile>,
    aliases: Vec<(&'static str, &'static str)>,
    prefixes: Vec<(&'static str, &'static str)>,
}

impl PromptProfileRegistry {
    fn resolve(&self, model: &str) -> &'static PromptProfile {
        let model_lower = model.to_lowercase();

        // Exact key match
        for profile in &self.profiles {
            if profile.key == model_lower {
                return profile;
            }
        }

        // Alias match
        for &(alias, key) in &self.aliases {
            if alias == model_lower {
                for profile in &self.profiles {
                    if profile.key == key {
                        return profile;
                    }
                }
            }
        }

        // Prefix match
        for &(prefix, key) in &self.prefixes {
            if model_lower.starts_with(prefix) {
                for profile in &self.profiles {
                    if profile.key == key {
                        return profile;
                    }
                }
            }
        }

        // Default: first registered profile
        self.profiles[0]
    }
}

static REGISTRY: LazyLock<PromptProfileRegistry> = LazyLock::new(|| PromptProfileRegistry {
    profiles: vec![&ISAAC, &QWEN],
    aliases: vec![
        ("default", "isaac-default"),
        ("isaac", "isaac-default"),
        ("perceptron", "isaac-default"),
        ("isaac-0.1", "isaac-default"),
        ("qwen", "qwen3-vl-235b-a22b-thinking"),
        ("qwen3", "qwen3-vl-235b-a22b-thinking"),
        ("qwen3-vl", "qwen3-vl-235b-a22b-thinking"),
        ("qwen3-vl-235b", "qwen3-vl-235b-a22b-thinking"),
    ],
    prefixes: vec![("isaac-", "isaac-default"), ("qwen3-", "qwen3-vl-235b-a22b-thinking")],
});

/// Resolve the prompt profile for a given model name.
///
/// Resolution order: exact key → alias → prefix → default (Isaac).
pub fn resolve_prompt_profile(model: &str) -> &'static PromptProfile {
    REGISTRY.resolve(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_exact_key() {
        let profile = resolve_prompt_profile("isaac-default");
        assert_eq!(profile.key, "isaac-default");

        let profile = resolve_prompt_profile("qwen3-vl-235b-a22b-thinking");
        assert_eq!(profile.key, "qwen3-vl-235b-a22b-thinking");
    }

    #[test]
    fn resolve_alias() {
        let profile = resolve_prompt_profile("isaac");
        assert_eq!(profile.key, "isaac-default");

        let profile = resolve_prompt_profile("default");
        assert_eq!(profile.key, "isaac-default");

        let profile = resolve_prompt_profile("qwen");
        assert_eq!(profile.key, "qwen3-vl-235b-a22b-thinking");

        let profile = resolve_prompt_profile("qwen3-vl");
        assert_eq!(profile.key, "qwen3-vl-235b-a22b-thinking");
    }

    #[test]
    fn resolve_prefix_match() {
        let profile = resolve_prompt_profile("qwen3-vl-72b");
        assert_eq!(profile.key, "qwen3-vl-235b-a22b-thinking");

        let profile = resolve_prompt_profile("isaac-2.0");
        assert_eq!(profile.key, "isaac-default");
    }

    #[test]
    fn resolve_default_fallback() {
        let profile = resolve_prompt_profile("unknown-model");
        assert_eq!(profile.key, "isaac-default");
    }

    #[test]
    fn caption_user_text() {
        let profile = resolve_prompt_profile("isaac");
        assert_eq!(
            profile.caption.user_text(&CaptionStyle::Concise),
            "Provide a concise, human-friendly caption for the upcoming image."
        );
        assert_eq!(
            profile.caption.user_text(&CaptionStyle::Detailed),
            "Provide a detailed caption describing key objects, relationships, and context in the upcoming image."
        );
    }

    #[test]
    fn ocr_user_text() {
        // Isaac: plain → None, markdown/html → Some
        let isaac = resolve_prompt_profile("isaac");
        assert_eq!(isaac.ocr.user_text(&OcrMode::Plain), None);
        assert!(isaac.ocr.user_text(&OcrMode::Markdown).is_some());
        assert!(isaac.ocr.user_text(&OcrMode::Html).is_some());

        // Qwen: plain → Some, markdown/html → Some
        let qwen = resolve_prompt_profile("qwen");
        assert_eq!(
            qwen.ocr.user_text(&OcrMode::Plain),
            Some("Read all the text in the image.")
        );
        assert_eq!(qwen.ocr.user_text(&OcrMode::Markdown), Some("qwenvl markdown"));
        assert_eq!(qwen.ocr.user_text(&OcrMode::Html), Some("qwenvl html"));
    }

    #[test]
    fn detect_system_text_general() {
        let profile = resolve_prompt_profile("isaac");
        assert_eq!(
            profile.detect.system_text(None),
            "Your goal is to segment out the objects in the scene"
        );
    }

    #[test]
    fn detect_system_text_with_categories() {
        let profile = resolve_prompt_profile("isaac");
        let cats = vec!["cat".to_string(), "dog".to_string()];
        assert_eq!(
            profile.detect.system_text(Some(&cats)),
            "Your goal is to segment out the following categories: cat, dog"
        );
    }

    #[test]
    fn detect_system_text_empty_categories() {
        let profile = resolve_prompt_profile("isaac");
        let cats: Vec<String> = vec![];
        assert_eq!(
            profile.detect.system_text(Some(&cats)),
            "Your goal is to segment out the objects in the scene"
        );
    }
}
