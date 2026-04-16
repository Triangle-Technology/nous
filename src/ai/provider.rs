//! AI Provider trait — pluggable LLM backend abstraction (P4 trait boundary).
//!
//! The cognitive engine NEVER imports concrete provider implementations.
//! All AI calls go through this trait, enabling:
//! - Swap between Anthropic/OpenAI/Google/local models
//! - Mock for testing
//! - Future: on-device inference via WASM

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::NousResult;

/// AI provider identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiProviderType {
    Anthropic,
    OpenAi,
    Google,
    Local,
}

/// A message in the conversation (provider-agnostic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Request for AI completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<ProviderMessage>,
    pub system_prompt: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
}

/// Response from AI completion (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub usage: TokenUsage,
    pub model: String,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A chunk from streaming response.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Incremental text content.
    TextDelta(String),
    /// Token usage (typically at end of stream).
    Usage(TokenUsage),
    /// Stream complete.
    Done,
    /// Error during streaming.
    Error(String),
}

/// The core AI provider trait.
///
/// Implement this for each LLM backend (Anthropic, OpenAI, etc.).
/// The cognitive engine only knows this trait — never the concrete type.
#[async_trait]
pub trait AiProvider: Send + Sync {
    /// Provider identifier.
    fn provider_type(&self) -> AiProviderType;

    /// Non-streaming completion.
    async fn complete(&self, request: CompletionRequest) -> NousResult<CompletionResponse>;

    /// Streaming completion — returns chunks via channel.
    async fn stream(
        &self,
        request: CompletionRequest,
        sender: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> NousResult<()>;
}

/// Embedding provider trait — separate from completion.
///
/// Nous may use different providers for completion vs embedding
/// (e.g., Claude for completion, OpenAI for embeddings).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding vector for text.
    async fn embed(&self, text: &str) -> NousResult<Vec<f32>>;

    /// Embedding dimension size.
    fn dimension(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_builds() {
        let req = CompletionRequest {
            model: "claude-3-5-sonnet".into(),
            messages: vec![ProviderMessage {
                role: MessageRole::User,
                content: "Hello".into(),
            }],
            system_prompt: Some("You are helpful".into()),
            max_tokens: 1024,
            temperature: 0.7,
            stream: false,
        };
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }
}
