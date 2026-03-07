use serde::{Deserialize, Serialize};

/// The modality of media being processed.
#[derive(Debug, Clone, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Modality {
    /// Image media.
    Image,
}

/// Media encoding format.
#[derive(Debug, Clone, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum MediaFormat {
    /// PNG image.
    Png,
    /// JPEG image.
    Jpeg,
    /// WebP image.
    Webp,
}

impl MediaFormat {
    /// Returns the [`Modality`] for this format.
    pub fn modality(&self) -> &Modality {
        match self {
            MediaFormat::Png | MediaFormat::Jpeg | MediaFormat::Webp => &Modality::Image,
        }
    }

    /// Returns the MIME type string (e.g. `"image/png"`).
    pub fn mime(&self) -> String {
        format!("{}/{}", self.modality(), self)
    }
}

/// Media for a request — either a URL or base64-encoded data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Media {
    /// A URL pointing to media.
    Url {
        /// The modality of the media.
        modality: Modality,
        /// The source URL.
        src: String,
    },
    /// Base64-encoded media data.
    Base64 {
        /// The media format.
        format: MediaFormat,
        /// The base64-encoded data.
        data: String,
    },
}

impl Media {
    /// Create from an image URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Media::Url {
            modality: Modality::Image,
            src: url.into(),
        }
    }

    /// Create from base64-encoded data.
    pub fn base64(format: MediaFormat, data: impl Into<String>) -> Self {
        Media::Base64 {
            format,
            data: data.into(),
        }
    }

    /// Returns the [`Modality`] for this media.
    pub fn modality(&self) -> &Modality {
        match self {
            Media::Url { modality, .. } => modality,
            Media::Base64 { format, .. } => format.modality(),
        }
    }

    /// Returns the URL for use in API requests.
    ///
    /// For `Url` variants, returns the URL as-is.
    /// For `Base64` variants, constructs a `data:{mime};base64,{data}` URL.
    pub fn to_url(&self) -> String {
        match self {
            Media::Url { src, .. } => src.clone(),
            Media::Base64 { format, data } => {
                format!("data:{};base64,{}", format.mime(), data)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_format_modality() {
        assert!(matches!(MediaFormat::Png.modality(), Modality::Image));
        assert!(matches!(MediaFormat::Jpeg.modality(), Modality::Image));
        assert!(matches!(MediaFormat::Webp.modality(), Modality::Image));
    }

    #[test]
    fn media_format_mime() {
        assert_eq!(MediaFormat::Png.mime(), "image/png");
        assert_eq!(MediaFormat::Jpeg.mime(), "image/jpeg");
        assert_eq!(MediaFormat::Webp.mime(), "image/webp");
    }

    #[test]
    fn media_image_url() {
        let media = Media::image_url("https://example.com/img.png");
        assert!(matches!(media.modality(), Modality::Image));
        assert_eq!(media.to_url(), "https://example.com/img.png");
    }

    #[test]
    fn media_base64_image() {
        let media = Media::base64(MediaFormat::Png, "abc123");
        assert!(matches!(media.modality(), Modality::Image));
        assert_eq!(media.to_url(), "data:image/png;base64,abc123");
    }
}
