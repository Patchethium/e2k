use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Zip(zip::result::ZipError),
    Json(serde_json::Error),
    /// The model file is malformed or uses an unsupported layout.
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Zip(e) => write!(f, "zip error: {e}"),
            Error::Json(e) => write!(f, "json error: {e}"),
            Error::Format(msg) => write!(f, "invalid model file: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Zip(e) => Some(e),
            Error::Json(e) => Some(e),
            Error::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Self {
        Error::Zip(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

pub(crate) fn format_err(msg: impl Into<String>) -> Error {
    Error::Format(msg.into())
}
