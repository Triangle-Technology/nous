# Changelog

All notable changes to `noos-langchain` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-19

Initial release.

### Added

- `NoosCallbackHandler` — synchronous `langchain_core.callbacks.base.BaseCallbackHandler`
  subclass that maps LangChain's standard hooks (`on_chain_start`, `on_chat_model_start`,
  `on_llm_start`, `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_tool_start`,
  `on_tool_end`, `on_tool_error`, `on_chain_end`, `on_chain_error`) to
  `noos.LLMEvent` calls on a caller-owned `noos.Regulator`.
- `AsyncNoosCallbackHandler` — async variant inheriting from `AsyncCallbackHandler`.
  Same semantics, async-def hooks, for use with `ainvoke` / `astream` / LangGraph
  async flows.
- `CircuitBreakError` — propagates through LangChain's callback manager (class attr
  `raise_error = True`) when a `circuit_break` decision fires with
  `raise_on_circuit_break=True`. Carries `.decision` for inspection.
- Three consumption modes for `Decision` surfacing:
  1. Poll `handler.last_decision` after `invoke()` / `ainvoke()` returns.
  2. Pass `on_decision=callback` to react mid-run.
  3. `raise_on_circuit_break=True` to abort on halt decisions.
- Defensive payload extractors in `_compat.py`:
  - `extract_token_usage` handles four shapes: OpenAI `llm_output.token_usage`,
    Anthropic `llm_output.usage`, modern LangChain `message.usage_metadata`,
    streaming `generation_info.usage`. Fails open to `(0, 0)`.
  - `extract_user_message` tries `inputs["input"|"question"|"query"|"prompt"]`,
    last-message fallback, then `json.dumps` / `str` fallback.
  - `extract_response_text`, `extract_chat_messages`, `tool_name_from_serialized`
    with similar fail-open chains.
- PEP 561 `py.typed` marker — type hints propagate to mypy / pyright.
- Examples:
  - `basic_smoke.py` — four scenarios against fabricated LangChain payloads,
    no LLM or network required.
  - `openai_tools_agent.py` — full `AgentExecutor` demo with toy looping tools
    against OpenAI.
  - `anthropic_tools_agent.py` — same shape against Claude Haiku via
    `langchain-anthropic`, the first end-to-end demo hitting a real LLM
    provider.
  - `langgraph_agent.py` — LangGraph React agent demo.
  - `crewai_agent.py` — CrewAI agent via the LangChain LLM callback path.
    Demonstrates that `noos-langchain` covers CrewAI without a dedicated
    `noos-crewai` package.
- `docs/announcements.md` — post-publish outreach drafts for HN Show /
  LangChain Discord / r/LocalLLaMA.
- Tests — 30+ behavioural tests via fabricated `SimpleNamespace` payloads; no
  actual LangChain agent runtime needed. Includes async path coverage via
  `asyncio.run` (no `pytest-asyncio` dependency).

### Compatibility

- Python `>=3.9`
- `noos>=0.1.0` (the Rust-backed Python binding)
- `langchain-core>=0.3.0`
- Works with LangGraph via the standard `config={"callbacks": [handler]}` path.
- Works with CrewAI through its `BaseAgent` callbacks when supplied as an
  LC-compatible handler (coverage not yet in examples — PRs welcome).

### Known limitations

- Callback manager wraps each hook inside a `try/except`; exceptions raised
  inside user-supplied `on_decision` callbacks propagate through `raise_error =
  True`. Wrap your own callback in a try/except if you want best-effort
  delivery.
- `emit_tokens=True` has Python overhead proportional to stream length.
  LangChain does not surface per-token logprobs through callbacks, so the
  regulator falls back to its structural-confidence heuristic regardless.
- A `noos.Regulator` is single-threaded (`unsendable`). Spawn one handler +
  regulator per concurrent `ainvoke`.
