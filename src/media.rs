use serde::{Deserialize, Serialize};

/// The modality supported by a model.
#[derive(Debug, Clone, Copy, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Modality {
    /// Image input.
    Image,
    /// Video input.
    Video,
}

/// Image encoding format.
#[derive(Debug, Clone, Copy, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ImageFormat {
    /// PNG image.
    Png,
    /// JPEG image.
    Jpeg,
    /// WebP image.
    Webp,
}

impl ImageFormat {
    /// Returns the MIME type string (e.g. `"image/png"`).
    pub fn mime(&self) -> String {
        format!("image/{}", self)
    }
}

/// Video encoding format.
#[derive(Debug, Clone, Copy, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum VideoFormat {
    /// MP4 video.
    Mp4,
    /// WebM video.
    Webm,
}

impl VideoFormat {
    /// Returns the MIME type string (e.g. `"video/mp4"`).
    pub fn mime(&self) -> String {
        format!("video/{}", self)
    }
}

/// Image input — either a URL or base64-encoded data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Image {
    /// A URL pointing to an image.
    Url {
        /// The source URL.
        src: String,
    },
    /// Base64-encoded image data.
    Base64 {
        /// The image format.
        format: ImageFormat,
        /// The base64-encoded data.
        data: String,
    },
}

impl Image {
    /// Create from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Image::Url { src: url.into() }
    }

    /// Create from base64-encoded data.
    ///
    /// For data larger than ~1MB, prefer [`Self::url`]; large base64 payloads can hit
    /// request-size limits and increase request latency.
    pub fn base64(format: ImageFormat, data: impl Into<String>) -> Self {
        Image::Base64 {
            format,
            data: data.into(),
        }
    }

    /// Returns the URL for use in API requests.
    ///
    /// For `Url` variants, returns the URL as-is.
    /// For `Base64` variants, constructs a `data:{mime};base64,{data}` URL.
    pub fn to_url(&self) -> String {
        match self {
            Image::Url { src } => src.clone(),
            Image::Base64 { format, data } => format!("data:{};base64,{}", format.mime(), data),
        }
    }
}

/// Video input — either a URL or base64-encoded data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Video {
    /// A URL pointing to a video.
    Url {
        /// The source URL.
        src: String,
    },
    /// Base64-encoded video data.
    Base64 {
        /// The video format.
        format: VideoFormat,
        /// The base64-encoded data.
        data: String,
    },
}

impl Video {
    /// Create from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Video::Url { src: url.into() }
    }

    /// Create from base64-encoded data.
    ///
    /// For data larger than ~1MB, prefer [`Self::url`]; large base64 payloads can hit
    /// request-size limits and increase request latency.
    pub fn base64(format: VideoFormat, data: impl Into<String>) -> Self {
        Video::Base64 {
            format,
            data: data.into(),
        }
    }

    /// Returns the URL for use in API requests.
    ///
    /// For `Url` variants, returns the URL as-is.
    /// For `Base64` variants, constructs a `data:{mime};base64,{data}` URL.
    pub fn to_url(&self) -> String {
        match self {
            Video::Url { src } => src.clone(),
            Video::Base64 { format, data } => format!("data:{};base64,{}", format.mime(), data),
        }
    }
}

/// Media for endpoints that accept either an image or a video.
///
/// The SDK does not validate that the media's modality matches the target model's
/// supported modalities; mismatches surface as a server-side error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Media {
    /// Image input.
    Image(Image),
    /// Video input.
    Video(Video),
}

impl From<Image> for Media {
    fn from(image: Image) -> Self {
        Media::Image(image)
    }
}

impl From<Video> for Media {
    fn from(video: Video) -> Self {
        Media::Video(video)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_format_mime() {
        assert_eq!(ImageFormat::Png.mime(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime(), "image/jpeg");
        assert_eq!(ImageFormat::Webp.mime(), "image/webp");
    }

    #[test]
    fn video_format_mime() {
        assert_eq!(VideoFormat::Mp4.mime(), "video/mp4");
        assert_eq!(VideoFormat::Webm.mime(), "video/webm");
    }

    #[test]
    fn image_url() {
        let img = Image::url("https://example.com/img.png");
        assert_eq!(img.to_url(), "https://example.com/img.png");
    }

    #[test]
    fn image_base64() {
        let img = Image::base64(ImageFormat::Png, "abc123");
        assert_eq!(img.to_url(), "data:image/png;base64,abc123");
    }

    #[test]
    fn video_url() {
        let vid = Video::url("https://example.com/vid.mp4");
        assert_eq!(vid.to_url(), "https://example.com/vid.mp4");
    }

    #[test]
    fn video_base64() {
        let vid = Video::base64(VideoFormat::Mp4, "xyz789");
        assert_eq!(vid.to_url(), "data:video/mp4;base64,xyz789");
    }

    #[test]
    fn media_from_image() {
        let media: Media = Image::url("https://example.com/img.png").into();
        assert!(matches!(media, Media::Image(_)));
    }

    #[test]
    fn media_from_video() {
        let media: Media = Video::url("https://example.com/vid.mp4").into();
        assert!(matches!(media, Media::Video(_)));
    }
}
