//! Request builder — construct provider-specific HTTP request bodies.
//!
//! P3: single function used by all providers. When an API changes format,
//! update ONE file. Maps provider-agnostic CompletionRequest to
//! provider-specific JSON structures.
//!
//! Pure functions, $0 LLM cost (only builds the request, doesn't send it).

use serde_json::{json, Value};

use crate::ai::provider::{AiProviderType, CompletionRequest, MessageRole};

/// Build a provider-specific request body from a generic CompletionRequest.
///
/// Returns (url_path, headers, body) — the caller handles the actual HTTP call.
pub fn build_provider_request(
    provider: AiProviderType,
    request: &CompletionRequest,
    api_key: &str,
) -> ProviderRequest {
    match provider {
        AiProviderType::Anthropic => build_anthropic(request, api_key),
        AiProviderType::OpenAi => build_openai(request, api_key),
        AiProviderType::Google => build_google(request, api_key),
        AiProviderType::Local => build_openai(request, api_key), // Local models typically use OpenAI-compatible API
    }
}

/// Provider-specific HTTP request data.
#[derive(Debug, Clone)]
pub struct ProviderRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

fn build_anthropic(request: &CompletionRequest, api_key: &str) -> ProviderRequest {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .filter(|m| m.role != MessageRole::System) // Anthropic: system is separate
        .map(|m| {
            json!({
                "role": match m.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::System => "user", // Shouldn't reach here
                },
                "content": m.content,
            })
        })
        .collect();

    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
    });

    // Anthropic: system prompt is a top-level field, not a message
    if let Some(system) = &request.system_prompt {
        body["system"] = json!(system);
    }

    ProviderRequest {
        url: "https://api.anthropic.com/v1/messages".into(),
        headers: vec![
            ("x-api-key".into(), api_key.into()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body,
    }
}

fn build_openai(request: &CompletionRequest, api_key: &str) -> ProviderRequest {
    // OpenAI: system prompt is the first message
    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = &request.system_prompt {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    for m in &request.messages {
        messages.push(json!({
            "role": match m.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            },
            "content": m.content,
        }));
    }

    ProviderRequest {
        url: "https://api.openai.com/v1/chat/completions".into(),
        headers: vec![
            ("Authorization".into(), format!("Bearer {api_key}")),
            ("Content-Type".into(), "application/json".into()),
        ],
        body: json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }),
    }
}

fn build_google(request: &CompletionRequest, api_key: &str) -> ProviderRequest {
    // Google: contents/parts nesting, system_instruction separate
    let contents: Vec<Value> = request
        .messages
        .iter()
        .filter(|m| m.role != MessageRole::System)
        .map(|m| {
            json!({
                "role": match m.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "model",
                    MessageRole::System => "user",
                },
                "parts": [{"text": m.content}],
            })
        })
        .collect();

    let mut body = json!({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": request.max_tokens,
            "temperature": request.temperature,
        },
    });

    if let Some(system) = &request.system_prompt {
        body["system_instruction"] = json!({"parts": [{"text": system}]});
    }

    let model = &request.model;
    let method = if request.stream {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };

    ProviderRequest {
        // P5: API key in header, not URL params (security)
        url: format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"),
        headers: vec![
            ("x-goog-api-key".into(), api_key.into()),
            ("Content-Type".into(), "application/json".into()),
        ],
        body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::provider::ProviderMessage;

    fn make_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".into(),
            messages: vec![ProviderMessage {
                role: MessageRole::User,
                content: "Hello".into(),
            }],
            system_prompt: Some("Be helpful".into()),
            max_tokens: 1024,
            temperature: 0.7,
            stream: false,
        }
    }

    #[test]
    fn anthropic_system_separate() {
        let req = build_provider_request(AiProviderType::Anthropic, &make_request(), "key");
        assert!(req.body.get("system").is_some());
        // Messages should not contain system role
        let msgs = req.body["messages"].as_array().unwrap();
        for m in msgs {
            assert_ne!(m["role"].as_str().unwrap(), "system");
        }
    }

    #[test]
    fn openai_system_as_message() {
        let req = build_provider_request(AiProviderType::OpenAi, &make_request(), "key");
        let msgs = req.body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"].as_str().unwrap(), "system");
    }

    #[test]
    fn google_system_instruction() {
        let req = build_provider_request(AiProviderType::Google, &make_request(), "key");
        assert!(req.body.get("system_instruction").is_some());
    }

    #[test]
    fn anthropic_api_key_in_header() {
        let req = build_provider_request(AiProviderType::Anthropic, &make_request(), "sk-test");
        let has_key = req.headers.iter().any(|(k, v)| k == "x-api-key" && v == "sk-test");
        assert!(has_key);
        // P5: key NOT in URL
        assert!(!req.url.contains("sk-test"));
    }

    #[test]
    fn google_key_in_header_not_url() {
        let req = build_provider_request(AiProviderType::Google, &make_request(), "goog-key");
        let has_key = req.headers.iter().any(|(k, _)| k == "x-goog-api-key");
        assert!(has_key);
        assert!(!req.url.contains("goog-key")); // P5
    }
}
