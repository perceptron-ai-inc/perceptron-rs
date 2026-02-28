use serde::{Deserialize, Serialize};

/// The type of media being processed.
#[derive(Debug, Clone, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum MediaType {
    /// Image media.
    Image,
    /// Video media.
    Video,
}

/// Media encoding format.
#[derive(Debug, Clone, strum::Display, strum::EnumString, Serialize, Deserialize)]
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
    /// MP4 video.
    Mp4,
    /// WebM video.
    Webm,
}

impl MediaFormat {
    /// Returns the [`MediaType`] for this format.
    pub fn media_type(&self) -> &MediaType {
        match self {
            MediaFormat::Png | MediaFormat::Jpeg | MediaFormat::Webp => &MediaType::Image,
            MediaFormat::Mp4 | MediaFormat::Webm => &MediaType::Video,
        }
    }

    /// Returns the MIME type string (e.g. `"image/png"`, `"video/mp4"`).
    pub fn mime(&self) -> String {
        format!("{}/{}", self.media_type(), self)
    }
}

/// Media for a request — either a URL or base64-encoded data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Media {
    /// A URL pointing to media.
    Url {
        /// The type of media.
        media_type: MediaType,
        /// The URL string.
        url: String,
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
            media_type: MediaType::Image,
            url: url.into(),
        }
    }

    /// Create from a video URL.
    pub fn video_url(url: impl Into<String>) -> Self {
        Media::Url {
            media_type: MediaType::Video,
            url: url.into(),
        }
    }

    /// Create from base64-encoded data.
    pub fn base64(format: MediaFormat, data: impl Into<String>) -> Self {
        Media::Base64 {
            format,
            data: data.into(),
        }
    }

    /// Returns the [`MediaType`] for this media.
    pub fn media_type(&self) -> &MediaType {
        match self {
            Media::Url { media_type, .. } => media_type,
            Media::Base64 { format, .. } => format.media_type(),
        }
    }

    /// Returns the URL for use in API requests.
    ///
    /// For `Url` variants, returns the URL as-is.
    /// For `Base64` variants, constructs a `data:{mime};base64,{data}` URL.
    pub fn to_url(&self) -> String {
        match self {
            Media::Url { url, .. } => url.clone(),
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
    fn media_format_media_type() {
        assert!(matches!(MediaFormat::Png.media_type(), MediaType::Image));
        assert!(matches!(MediaFormat::Jpeg.media_type(), MediaType::Image));
        assert!(matches!(MediaFormat::Webp.media_type(), MediaType::Image));
        assert!(matches!(MediaFormat::Mp4.media_type(), MediaType::Video));
        assert!(matches!(MediaFormat::Webm.media_type(), MediaType::Video));
    }

    #[test]
    fn media_format_mime() {
        assert_eq!(MediaFormat::Png.mime(), "image/png");
        assert_eq!(MediaFormat::Jpeg.mime(), "image/jpeg");
        assert_eq!(MediaFormat::Webp.mime(), "image/webp");
        assert_eq!(MediaFormat::Mp4.mime(), "video/mp4");
        assert_eq!(MediaFormat::Webm.mime(), "video/webm");
    }

    #[test]
    fn media_image_url() {
        let media = Media::image_url("https://example.com/img.png");
        assert!(matches!(media.media_type(), MediaType::Image));
        assert_eq!(media.to_url(), "https://example.com/img.png");
    }

    #[test]
    fn media_video_url() {
        let media = Media::video_url("https://example.com/vid.mp4");
        assert!(matches!(media.media_type(), MediaType::Video));
        assert_eq!(media.to_url(), "https://example.com/vid.mp4");
    }

    #[test]
    fn media_base64_image() {
        let media = Media::base64(MediaFormat::Png, "abc123");
        assert!(matches!(media.media_type(), MediaType::Image));
        assert_eq!(media.to_url(), "data:image/png;base64,abc123");
    }

    #[test]
    fn media_base64_video() {
        let media = Media::base64(MediaFormat::Mp4, "xyz789");
        assert!(matches!(media.media_type(), MediaType::Video));
        assert_eq!(media.to_url(), "data:video/mp4;base64,xyz789");
    }
}
