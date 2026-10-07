use std::fmt;

use serde::de::{IntoDeserializer, value::Error as ValueError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A value from a service listing: one this crate names, or one the service added after this crate
/// version was published. Listing enums are open by contract, so a new value never fails the
/// response; it arrives as [`OpenEnum::Unknown`] carrying the wire string.
///
/// Compares equal to the inner value, so `model.modalities == vec![Modality::Image]` reads as before.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenEnum<T> {
    /// A value this crate names.
    Known(T),
    /// A value this crate does not name, as the service wrote it.
    Unknown(String),
}

impl<T> OpenEnum<T> {
    /// The value when this crate names it.
    pub fn known(&self) -> Option<&T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown(_) => None,
        }
    }
}

impl<T: PartialEq> PartialEq<T> for OpenEnum<T> {
    fn eq(&self, other: &T) -> bool {
        matches!(self, Self::Known(value) if value == other)
    }
}

impl<T> From<T> for OpenEnum<T> {
    fn from(value: T) -> Self {
        Self::Known(value)
    }
}

impl<T: fmt::Display> fmt::Display for OpenEnum<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known(value) => value.fmt(f),
            Self::Unknown(raw) => f.write_str(raw),
        }
    }
}

impl<T: Serialize> Serialize for OpenEnum<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Known(value) => value.serialize(serializer),
            Self::Unknown(raw) => raw.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for OpenEnum<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        let known: Result<T, ValueError> = T::deserialize(raw.as_str().into_deserializer());
        Ok(match known {
            Ok(value) => Self::Known(value),
            Err(_) => Self::Unknown(raw),
        })
    }
}

#[cfg(feature = "schema")]
impl<T: schemars::JsonSchema> schemars::JsonSchema for OpenEnum<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Open{}", T::schema_name()).into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // Any string is valid on the wire; the named values are documentation, not a constraint.
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, strum::Display)]
    #[serde(rename_all = "snake_case")]
    #[strum(serialize_all = "snake_case")]
    enum Colour {
        Red,
        Blue,
    }

    #[test]
    fn a_named_value_is_known_and_any_other_string_is_kept() {
        let values: Vec<OpenEnum<Colour>> = serde_json::from_str(r#"["red", "chartreuse", "blue"]"#).unwrap();
        assert_eq!(
            values,
            vec![
                OpenEnum::Known(Colour::Red),
                OpenEnum::Unknown("chartreuse".to_string()),
                OpenEnum::Known(Colour::Blue),
            ]
        );
        assert_eq!(values[0], Colour::Red);
        assert_ne!(values[1], Colour::Red);
        assert_eq!(values[0].known(), Some(&Colour::Red));
        assert_eq!(values[1].known(), None);
    }

    #[test]
    fn it_round_trips_and_displays_as_the_wire_string() {
        let values = vec![
            OpenEnum::Known(Colour::Blue),
            OpenEnum::Unknown("chartreuse".to_string()),
        ];
        assert_eq!(serde_json::to_string(&values).unwrap(), r#"["blue","chartreuse"]"#);
        assert_eq!(
            values.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["blue", "chartreuse"]
        );
    }

    #[test]
    fn a_non_string_is_still_an_error() {
        assert!(serde_json::from_str::<OpenEnum<Colour>>("3").is_err());
    }
}
