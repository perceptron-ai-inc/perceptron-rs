use crate::media::Media;
use crate::types::{CaptionStyle, OcrMode, OutputFormat};

/// A prompt whose text varies by media modality.
#[derive(Debug, Clone, PartialEq)]
pub struct ModalityPrompt {
    /// Text used when the request media is an image.
    image: &'static str,
    /// Text used when the request media is a video.
    video: &'static str,
}

impl ModalityPrompt {
    /// Return the prompt text for the given media.
    pub fn get(&self, media: &Media) -> &'static str {
        match media {
            Media::Image(_) => self.image,
            Media::Video(_) => self.video,
        }
    }
}

/// Prompt template for question requests.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionPromptTemplate {
    /// System instruction for open-ended (text) questions.
    open_instruction: Option<ModalityPrompt>,
    /// System instruction for grounded (spatial) questions.
    grounded_instruction: Option<ModalityPrompt>,
}

impl QuestionPromptTemplate {
    /// Resolve the system instruction for the given output format and media.
    /// `None` for `output_format` selects the open (text) instruction.
    pub fn resolve_system(&self, output_format: Option<&OutputFormat>, media: &Media) -> Option<&'static str> {
        let prompt = match output_format {
            None => self.open_instruction.as_ref(),
            Some(_) => self.grounded_instruction.as_ref(),
        }?;
        Some(prompt.get(media))
    }
}

/// Prompt template for caption requests.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionPromptTemplate {
    /// Optional system instruction for the caption endpoint.
    system: Option<ModalityPrompt>,
    /// User text for concise captions.
    concise: ModalityPrompt,
    /// User text for detailed captions.
    detailed: ModalityPrompt,
}

impl CaptionPromptTemplate {
    /// Resolve the system instruction for the given media, if any.
    pub fn resolve_system(&self, media: &Media) -> Option<&'static str> {
        self.system.as_ref().map(|p| p.get(media))
    }

    /// Resolve the user text for the given caption style and media.
    pub fn resolve_user(&self, style: &CaptionStyle, media: &Media) -> &'static str {
        match style {
            CaptionStyle::Concise => self.concise.get(media),
            CaptionStyle::Detailed => self.detailed.get(media),
        }
    }
}

/// Prompt template for OCR requests.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrPromptTemplate {
    /// Optional system instruction for the OCR endpoint.
    system: Option<&'static str>,
    /// User text for plain mode (None means no user text).
    plain: Option<&'static str>,
    /// User text for markdown mode.
    markdown: &'static str,
    /// User text for HTML mode.
    html: &'static str,
}

impl OcrPromptTemplate {
    /// Resolve the system instruction, if any.
    pub fn resolve_system(&self) -> Option<&'static str> {
        self.system
    }

    /// Resolve the user text for the given OCR mode, or `None` for plain when omitted.
    pub fn resolve_user(&self, mode: &OcrMode) -> Option<&'static str> {
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
    general: ModalityPrompt,
    /// Template with `{categories}` placeholder for category-specific detection.
    category_template: ModalityPrompt,
}

impl DetectPromptTemplate {
    /// Resolve the system text for the given categories and media, substituting `{categories}` if provided.
    pub fn resolve_system(&self, categories: Option<&[String]>, media: &Media) -> String {
        match categories {
            Some(cats) if !cats.is_empty() => self
                .category_template
                .get(media)
                .replace("{categories}", &cats.join(", ")),
            _ => self.general.get(media).to_string(),
        }
    }
}

/// A collection of prompt templates for a specific model family.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptProfile {
    /// Question prompt template.
    pub question: QuestionPromptTemplate,
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
    question: QuestionPromptTemplate {
        open_instruction: None,
        grounded_instruction: None,
    },
    caption: CaptionPromptTemplate {
        system: None,
        concise: ModalityPrompt {
            image: "Provide a concise, human-friendly caption for the upcoming image.",
            video: "Provide a concise, human-friendly caption for the upcoming video.",
        },
        detailed: ModalityPrompt {
            image: "Provide a detailed caption describing key objects, relationships, and context in the upcoming image.",
            video: "Provide a detailed caption describing key objects, relationships, and context in the upcoming video.",
        },
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
        general: ModalityPrompt {
            image: "Your goal is to segment out the objects in the scene",
            video: "Your goal is to segment out the objects in the scene",
        },
        category_template: ModalityPrompt {
            image: "Your goal is to segment out the following categories: {categories}",
            video: "Your goal is to segment out the following categories: {categories}",
        },
    },
};

const QWEN: PromptProfile = PromptProfile {
    question: QuestionPromptTemplate {
        open_instruction: None,
        grounded_instruction: Some(ModalityPrompt {
            image: "You are Qwen3-VL performing grounded reasoning. Give the answer and reference the relevant regions using structured tags when available. Report bbox coordinates in JSON format.",
            video: "You are Qwen3-VL performing grounded reasoning. Give the answer and reference the relevant regions using structured tags when available. Report bbox coordinates in JSON format.",
        }),
    },
    caption: CaptionPromptTemplate {
        system: None,
        concise: ModalityPrompt {
            image: "Describe the primary subjects, their actions, and visible context in one vivid sentence.",
            video: "Describe the primary subjects, their actions, and visible context in one vivid sentence.",
        },
        detailed: ModalityPrompt {
            image: "Provide a multi-sentence caption that calls out subjects, relationships, scene intent, and any text embedded in the image.",
            video: "Provide a multi-sentence caption that calls out subjects, relationships, scene intent, and any text embedded in the video.",
        },
    },
    ocr: OcrPromptTemplate {
        system: None,
        plain: Some("Read all the text in the image."),
        markdown: "qwenvl markdown",
        html: "qwenvl html",
    },
    detect: DetectPromptTemplate {
        general: ModalityPrompt {
            image: "Locate every object of interest and report bounding box coordinates in JSON format.",
            video: "Locate every object of interest and report bounding box coordinates in JSON format.",
        },
        category_template: ModalityPrompt {
            image: "Locate every instance that belongs to the following categories: \"{categories}\". Report bbox coordinates in JSON format.",
            video: "Locate every instance that belongs to the following categories: \"{categories}\". Report bbox coordinates in JSON format.",
        },
    },
};

/// Resolve the prompt profile for a given model name based on its prefix.
///
/// Defaults to Isaac if no specific profile matches.
pub fn resolve_prompt_profile(model: &str) -> &'static PromptProfile {
    let m = model.to_lowercase();
    if m.starts_with("qwen") { &QWEN } else { &ISAAC }
}
