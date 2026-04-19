"""Defensive extractors for LangChain callback payloads.

LangChain's callback payload shapes vary by provider, version, and
whether the call went through a chat model or a completion model. These
helpers try several canonical paths and return safe defaults on failure
so the adapter never raises inside a user's agent loop.

Kept separate from ``callback.py`` so shape drift can be patched in one
place.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def extract_user_message(inputs: Any) -> Optional[str]:
    """Pull a representative user-message string out of a chain's input dict.

    LangChain's ``on_chain_start`` receives the raw ``inputs`` passed to
    ``invoke``. Convention: the user's query lives under ``"input"``,
    ``"question"``, or ``"messages"``. Falls back to ``str(inputs)`` so
    scope-drift always has *something* to compare against.
    """
    if inputs is None:
        return None
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("input", "question", "query", "prompt"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value
        messages = inputs.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = _message_content(last)
            if content:
                return content
        try:
            return json.dumps(inputs, default=str)
        except (TypeError, ValueError):
            return str(inputs)
    if isinstance(inputs, list) and inputs:
        return _message_content(inputs[-1]) or str(inputs[-1])
    return str(inputs)


def extract_prompts_as_message(prompts: List[str]) -> Optional[str]:
    """Flatten an ``on_llm_start`` prompts list into one string."""
    if not prompts:
        return None
    return "\n\n".join(str(p) for p in prompts if p)


def extract_chat_messages(messages: List[List[Any]]) -> Optional[str]:
    """Flatten ``on_chat_model_start`` nested messages into one string.

    Takes the FINAL (most recent) message in the FIRST batch — the
    convention used by LangChain agents where each call contains one
    thread.
    """
    if not messages or not messages[0]:
        return None
    last = messages[0][-1]
    return _message_content(last)


def _message_content(msg: Any) -> Optional[str]:
    """Best-effort ``BaseMessage.content`` extraction."""
    if msg is None:
        return None
    if isinstance(msg, str):
        return msg
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts) if parts else None
    return None


def extract_response_text(response: Any) -> str:
    """Pull the assistant's full response text from an ``LLMResult``.

    Tries, in order: ``generations[0][0].text`` → ``generations[0][0].message.content``
    → ``str(response)``. Never raises.
    """
    try:
        gens = getattr(response, "generations", None)
        if gens and gens[0]:
            first = gens[0][0]
            text = getattr(first, "text", None)
            if isinstance(text, str) and text:
                return text
            msg = getattr(first, "message", None)
            if msg is not None:
                content = _message_content(msg)
                if content:
                    return content
    except (AttributeError, IndexError, TypeError):
        pass
    return str(response)


def extract_token_usage(response: Any) -> Tuple[int, int]:
    """Pull ``(tokens_in, tokens_out)`` from an ``LLMResult``.

    Tries, in order:
      1. ``response.llm_output["token_usage"]`` (OpenAI completion/chat style)
      2. ``response.llm_output["usage"]`` (Anthropic style)
      3. ``response.generations[0][0].message.usage_metadata`` (modern LC)
      4. ``response.generations[0][0].generation_info["usage"]``

    Returns ``(0, 0)`` if none match — downstream regulator tolerates
    zero-cost events.
    """
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        tok = llm_output.get("token_usage")
        if isinstance(tok, dict):
            tokens_in = _coerce_int(tok.get("prompt_tokens") or tok.get("input_tokens"))
            tokens_out = _coerce_int(tok.get("completion_tokens") or tok.get("output_tokens"))
            if tokens_in or tokens_out:
                return tokens_in, tokens_out
        usage = llm_output.get("usage")
        if isinstance(usage, dict):
            tokens_in = _coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
            tokens_out = _coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
            if tokens_in or tokens_out:
                return tokens_in, tokens_out

    gens = getattr(response, "generations", None)
    if gens and gens[0]:
        first = gens[0][0]
        msg = getattr(first, "message", None)
        if msg is not None:
            usage_meta = getattr(msg, "usage_metadata", None)
            if isinstance(usage_meta, dict):
                tokens_in = _coerce_int(usage_meta.get("input_tokens"))
                tokens_out = _coerce_int(usage_meta.get("output_tokens"))
                if tokens_in or tokens_out:
                    return tokens_in, tokens_out
        gen_info = getattr(first, "generation_info", None)
        if isinstance(gen_info, dict):
            usage = gen_info.get("usage") or gen_info.get("token_usage")
            if isinstance(usage, dict):
                tokens_in = _coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
                tokens_out = _coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
                if tokens_in or tokens_out:
                    return tokens_in, tokens_out

    return 0, 0


def tool_name_from_serialized(serialized: Optional[Dict[str, Any]]) -> str:
    """Pull the tool's name from ``on_tool_start``'s ``serialized`` dict.

    LangChain 0.3+: ``serialized["name"]`` is canonical. Older versions
    used ``serialized["id"][-1]`` as a last-resort. Returns ``"unknown"``
    if neither is available so tool-loop detection still has a key to
    count against.
    """
    if not isinstance(serialized, dict):
        return "unknown"
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    ident = serialized.get("id")
    if isinstance(ident, list) and ident:
        last = ident[-1]
        if isinstance(last, str) and last:
            return last
    return "unknown"


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion; returns 0 on failure."""
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
