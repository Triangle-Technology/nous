//! Response parser — parse streaming (SSE) and full responses from AI providers.
//!
//! P3: single parser used by all providers. Handles the three different
//! response formats (Anthropic, OpenAI, Google) with a unified output type.
//!
//! Pure functions, $0 LLM cost (only parses, doesn't call).

use serde_json::Value;

use crate::ai::provider::{AiProviderType, StreamChunk, TokenUsage};

/// Parse a single SSE line from a streaming response.
///
/// Returns `None` for non-data lines (comments, empty lines, etc.).
pub fn parse_sse_line(provider: AiProviderType, line: &str) -> Option<StreamChunk> {
    let line = line.trim();

    // SSE format: "data: {json}"
    let data = if let Some(rest) = line.strip_prefix("data: ") {
        rest.trim()
    } else {
        return None; // Not a data line
    };

    // End-of-stream markers
    if data == "[DONE]" {
        return Some(StreamChunk::Done);
    }

    // Parse JSON
    let json: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return None, // Malformed JSON — skip (P6 fail-open)
    };

    parse_stream_chunk(provider, &json)
}

/// Parse a JSON chunk from a streaming response.
pub fn parse_stream_chunk(provider: AiProviderType, data: &Value) -> Option<StreamChunk> {
    match provider {
        AiProviderType::Anthropic => parse_anthropic_chunk(data),
        AiProviderType::OpenAi | AiProviderType::Local => parse_openai_chunk(data),
        AiProviderType::Google => parse_google_chunk(data),
    }
}

/// Parse a complete (non-streaming) response.
pub fn parse_full_response(provider: AiProviderType, data: &Value) -> Option<(String, TokenUsage)> {
    match provider {
        AiProviderType::Anthropic => {
            let text = data
                .get("content")?
                .as_array()?
                .iter()
                .filter_map(|block| {
                    if block.get("type")?.as_str()? == "text" {
                        block.get("text")?.as_str().map(String::from)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            let usage = parse_anthropic_usage(data);
            Some((text, usage))
        }
        AiProviderType::OpenAi | AiProviderType::Local => {
            let text = data
                .get("choices")?
                .get(0)?
                .get("message")?
                .get("content")?
                .as_str()?
                .to_string();

            let usage = parse_openai_usage(data);
            Some((text, usage))
        }
        AiProviderType::Google => {
            let text = data
                .get("candidates")?
                .get(0)?
                .get("content")?
                .get("parts")?
                .get(0)?
                .get("text")?
                .as_str()?
                .to_string();

            let usage = parse_google_usage(data);
            Some((text, usage))
        }
    }
}

// ── Anthropic Parsing ──────────────────────────────────────────────────

fn parse_anthropic_chunk(data: &Value) -> Option<StreamChunk> {
    let event_type = data.get("type")?.as_str()?;

    match event_type {
        "content_block_delta" => {
            let delta = data.get("delta")?;
            if delta.get("type")?.as_str()? == "text_delta" {
                let text = delta.get("text")?.as_str()?.to_string();
                Some(StreamChunk::TextDelta(text))
            } else {
                None
            }
        }
        "message_delta" => {
            // Usage comes with message_delta at end
            let usage = data.get("usage").map(|u| TokenUsage {
                input_tokens: 0, // Not in delta
                output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            });
            usage.map(StreamChunk::Usage)
        }
        "message_stop" => Some(StreamChunk::Done),
        "error" => {
            let msg = data
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            Some(StreamChunk::Error(msg.to_string()))
        }
        _ => None,
    }
}

fn parse_anthropic_usage(data: &Value) -> TokenUsage {
    data.get("usage")
        .map(|u| TokenUsage {
            input_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
        .unwrap_or_default()
}

// ── OpenAI Parsing ─────────────────────────────────────────────────────

fn parse_openai_chunk(data: &Value) -> Option<StreamChunk> {
    let choices = data.get("choices")?.as_array()?;
    let choice = choices.first()?;

    let delta = choice.get("delta")?;
    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
        if !content.is_empty() {
            return Some(StreamChunk::TextDelta(content.to_string()));
        }
    }

    // Check finish_reason
    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
        if reason == "stop" {
            return Some(StreamChunk::Done);
        }
    }

    None
}

fn parse_openai_usage(data: &Value) -> TokenUsage {
    data.get("usage")
        .map(|u| TokenUsage {
            input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
        .unwrap_or_default()
}

// ── Google Parsing ─────────────────────────────────────────────────────

fn parse_google_chunk(data: &Value) -> Option<StreamChunk> {
    let text = data
        .get("candidates")?
        .get(0)?
        .get("content")?
        .get("parts")?
        .get(0)?
        .get("text")?
        .as_str()?;

    if text.is_empty() {
        None
    } else {
        Some(StreamChunk::TextDelta(text.to_string()))
    }
}

fn parse_google_usage(data: &Value) -> TokenUsage {
    data.get("usageMetadata")
        .map(|u| TokenUsage {
            input_tokens: u.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_anthropic_text_delta() {
        let data = json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let chunk = parse_stream_chunk(AiProviderType::Anthropic, &data);
        assert!(matches!(chunk, Some(StreamChunk::TextDelta(t)) if t == "Hello"));
    }

    #[test]
    fn parse_anthropic_done() {
        let data = json!({"type": "message_stop"});
        let chunk = parse_stream_chunk(AiProviderType::Anthropic, &data);
        assert!(matches!(chunk, Some(StreamChunk::Done)));
    }

    #[test]
    fn parse_openai_text_delta() {
        let data = json!({
            "choices": [{"delta": {"content": "World"}, "finish_reason": null}]
        });
        let chunk = parse_stream_chunk(AiProviderType::OpenAi, &data);
        assert!(matches!(chunk, Some(StreamChunk::TextDelta(t)) if t == "World"));
    }

    #[test]
    fn parse_openai_done() {
        let data = json!({
            "choices": [{"delta": {}, "finish_reason": "stop"}]
        });
        let chunk = parse_stream_chunk(AiProviderType::OpenAi, &data);
        assert!(matches!(chunk, Some(StreamChunk::Done)));
    }

    #[test]
    fn parse_google_text() {
        let data = json!({
            "candidates": [{"content": {"parts": [{"text": "Xin chào"}]}}]
        });
        let chunk = parse_stream_chunk(AiProviderType::Google, &data);
        assert!(matches!(chunk, Some(StreamChunk::TextDelta(t)) if t == "Xin chào"));
    }

    #[test]
    fn parse_sse_done_marker() {
        let chunk = parse_sse_line(AiProviderType::OpenAi, "data: [DONE]");
        assert!(matches!(chunk, Some(StreamChunk::Done)));
    }

    #[test]
    fn parse_sse_non_data_line() {
        assert!(parse_sse_line(AiProviderType::OpenAi, ": comment").is_none());
        assert!(parse_sse_line(AiProviderType::OpenAi, "").is_none());
        assert!(parse_sse_line(AiProviderType::OpenAi, "event: ping").is_none());
    }

    #[test]
    fn parse_full_anthropic() {
        let data = json!({
            "content": [{"type": "text", "text": "Response text"}],
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let (text, usage) = parse_full_response(AiProviderType::Anthropic, &data).unwrap();
        assert_eq!(text, "Response text");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
    }

    #[test]
    fn parse_full_openai() {
        let data = json!({
            "choices": [{"message": {"content": "OpenAI response"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 15}
        });
        let (text, usage) = parse_full_response(AiProviderType::OpenAi, &data).unwrap();
        assert_eq!(text, "OpenAI response");
        assert_eq!(usage.input_tokens, 5);
    }

    #[test]
    fn malformed_json_returns_none() {
        let chunk = parse_sse_line(AiProviderType::OpenAi, "data: not-json");
        assert!(chunk.is_none()); // P6 fail-open
    }
}
