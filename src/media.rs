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
    /// Audio input.
    Audio,
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

/// Audio encoding format.
#[derive(Debug, Clone, Copy, PartialEq, strum::Display, strum::EnumString, Serialize, Deserialize)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum AudioFormat {
    /// WAV audio.
    Wav,
    /// MP3 audio.
    Mp3,
    /// FLAC audio.
    Flac,
}

impl AudioFormat {
    /// Returns the MIME type string (e.g. `"audio/wav"`).
    pub fn mime(&self) -> String {
        match self {
            AudioFormat::Wav => "audio/wav".to_string(),
            AudioFormat::Mp3 => "audio/mpeg".to_string(),
            AudioFormat::Flac => "audio/flac".to_string(),
        }
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

/// Audio input — either a URL or base64-encoded data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Audio {
    /// A URL pointing to an audio file.
    Url {
        /// The source URL.
        src: String,
    },
    /// Base64-encoded audio data.
    Base64 {
        /// The audio format.
        format: AudioFormat,
        /// The base64-encoded data.
        data: String,
    },
}

impl Audio {
    /// Create from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Audio::Url { src: url.into() }
    }

    /// Create from base64-encoded data.
    ///
    /// For data larger than ~1MB, prefer [`Self::url`]; large base64 payloads can hit
    /// request-size limits and increase request latency.
    pub fn base64(format: AudioFormat, data: impl Into<String>) -> Self {
        Audio::Base64 {
            format,
            data: data.into(),
        }
    }

    /// Convert to a URL string for API requests.
    ///
    /// For `Base64` variants, constructs a `data:{mime};base64,{data}` URL.
    pub fn to_url(&self) -> String {
        match self {
            Audio::Url { src } => src.clone(),
            Audio::Base64 { format, data } => format!("data:{};base64,{}", format.mime(), data),
        }
    }
}

/// Media for endpoints that accept an image, a video, or an audio clip.
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
    /// Audio input.
    Audio(Audio),
}

impl From<Audio> for Media {
    fn from(audio: Audio) -> Self {
        Media::Audio(audio)
    }
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

    #[test]
    fn audio_format_mime() {
        assert_eq!(AudioFormat::Wav.mime(), "audio/wav");
        assert_eq!(AudioFormat::Mp3.mime(), "audio/mpeg");
        assert_eq!(AudioFormat::Flac.mime(), "audio/flac");
    }

    #[test]
    fn audio_url() {
        let clip = Audio::url("https://example.com/clip.wav");
        assert_eq!(clip.to_url(), "https://example.com/clip.wav");
    }

    #[test]
    fn audio_base64() {
        let clip = Audio::base64(AudioFormat::Wav, "abc123");
        assert_eq!(clip.to_url(), "data:audio/wav;base64,abc123");
    }

    #[test]
    fn media_from_audio() {
        let media: Media = Audio::url("https://example.com/clip.wav").into();
        assert!(matches!(media, Media::Audio(Audio::Url { .. })));
        let media: Media = Audio::base64(AudioFormat::Flac, "abc").into();
        assert!(matches!(
            media,
            Media::Audio(Audio::Base64 {
                format: AudioFormat::Flac,
                ..
            })
        ));
    }
}
