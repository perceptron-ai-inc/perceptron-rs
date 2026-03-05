use crate::types::{CaptionStyle, OcrMode};

/// Prompt template for caption requests.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct PromptProfile {
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

/// Resolve the prompt profile for a given model name based on its prefix.
///
/// Returns `None` if the model doesn't match any known profile.
pub fn resolve_prompt_profile(model: &str) -> Option<&'static PromptProfile> {
    let m = model.to_lowercase();
    if m.starts_with("qwen") {
        Some(&QWEN)
    } else if m.starts_with("isaac") {
        Some(&ISAAC)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_isaac_models() {
        assert_eq!(resolve_prompt_profile("isaac").unwrap(), &ISAAC);
        assert_eq!(resolve_prompt_profile("isaac-default").unwrap(), &ISAAC);
        assert_eq!(resolve_prompt_profile("isaac-2.0").unwrap(), &ISAAC);
    }

    #[test]
    fn resolve_qwen_models() {
        assert_eq!(resolve_prompt_profile("qwen").unwrap(), &QWEN);
        assert_eq!(resolve_prompt_profile("qwen3-vl").unwrap(), &QWEN);
        assert_eq!(resolve_prompt_profile("qwen3-vl-72b").unwrap(), &QWEN);
        assert_eq!(resolve_prompt_profile("qwen3-vl-235b-a22b-thinking").unwrap(), &QWEN);
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve_prompt_profile("unknown-model").is_none());
    }

    #[test]
    fn caption_user_text() {
        let profile = resolve_prompt_profile("isaac").unwrap();
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
        let isaac = resolve_prompt_profile("isaac").unwrap();
        assert_eq!(isaac.ocr.user_text(&OcrMode::Plain), None);
        assert!(isaac.ocr.user_text(&OcrMode::Markdown).is_some());
        assert!(isaac.ocr.user_text(&OcrMode::Html).is_some());

        // Qwen: plain → Some, markdown/html → Some
        let qwen = resolve_prompt_profile("qwen").unwrap();
        assert_eq!(
            qwen.ocr.user_text(&OcrMode::Plain),
            Some("Read all the text in the image.")
        );
        assert_eq!(qwen.ocr.user_text(&OcrMode::Markdown), Some("qwenvl markdown"));
        assert_eq!(qwen.ocr.user_text(&OcrMode::Html), Some("qwenvl html"));
    }

    #[test]
    fn detect_system_text_general() {
        let profile = resolve_prompt_profile("isaac").unwrap();
        assert_eq!(
            profile.detect.system_text(None),
            "Your goal is to segment out the objects in the scene"
        );
    }

    #[test]
    fn detect_system_text_with_categories() {
        let profile = resolve_prompt_profile("isaac").unwrap();
        let cats = vec!["cat".to_string(), "dog".to_string()];
        assert_eq!(
            profile.detect.system_text(Some(&cats)),
            "Your goal is to segment out the following categories: cat, dog"
        );
    }

    #[test]
    fn detect_system_text_empty_categories() {
        let profile = resolve_prompt_profile("isaac").unwrap();
        let cats: Vec<String> = vec![];
        assert_eq!(
            profile.detect.system_text(Some(&cats)),
            "Your goal is to segment out the objects in the scene"
        );
    }
}
