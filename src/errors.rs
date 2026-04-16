//! Noos error types — structured errors with context (P5: fail-open ≠ swallow).
//!
//! Every fallible function returns `Result<T, NoosError>`. Callers can
//! `.unwrap_or_default()` for fail-open, or `.inspect_err()` for monitoring.

use thiserror::Error;

/// Top-level error type for all Noos operations.
#[derive(Debug, Error)]
pub enum NoosError {
    /// AI provider returned an error (HTTP, timeout, auth).
    #[error("AI provider error: {provider} — {message}")]
    Provider {
        provider: String,
        message: String,
        status: Option<u16>,
    },

    /// Plugin lifecycle error (init, destroy, enrich).
    #[error("Plugin error: {plugin_id} — {message}")]
    Plugin { plugin_id: String, message: String },

    /// Pipeline execution error.
    #[error("Pipeline error: {composition_id} step {step} — {message}")]
    Pipeline {
        composition_id: String,
        step: usize,
        message: String,
    },

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Memory/persistence error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Operation cancelled (via abort signal).
    #[error("Operation cancelled")]
    Cancelled,

    /// Intervention not supported at the required depth.
    /// The model doesn't support the requested intervention level.
    #[error("Unsupported intervention: {0}")]
    UnsupportedIntervention(String),

    /// Generic internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for NoosError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// Convenience type alias.
pub type NoosResult<T> = Result<T, NoosError>;
