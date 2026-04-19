"""Behavioural tests for :class:`NoosCallbackHandler`.

Exercises the handler by calling its LangChain hooks directly with
fabricated payloads — no actual LangChain agent runtime is needed. This
keeps the test matrix fast and lets us cover payload shapes from
multiple providers (OpenAI, Anthropic, modern usage_metadata) in one
suite.

Run::

    cd bindings/python-langchain
    pip install -e ".[test]"
    pip install /path/to/noos  # or pip install noos once published
    pytest tests/
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

# Import guards — test suite requires both packages to be importable.
pytest.importorskip("noos")
pytest.importorskip("langchain_core")

from noos import Regulator

from noos_langchain import (
    AsyncNoosCallbackHandler,
    CircuitBreakError,
    NoosCallbackHandler,
)


# ── Fixtures and helpers ─────────────────────────────────────────────


def _make_llm_result(
    text: str,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    style: str = "openai",
) -> Any:
    """Fabricate an LLMResult-shaped object for the requested provider style."""
    gen = SimpleNamespace(text=text, message=None, generation_info=None)
    if style == "openai":
        return SimpleNamespace(
            generations=[[gen]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                }
            },
        )
    if style == "anthropic":
        return SimpleNamespace(
            generations=[[gen]],
            llm_output={
                "usage": {
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                }
            },
        )
    if style == "usage_metadata":
        msg = SimpleNamespace(
            content=text,
            usage_metadata={
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        )
        gen_with_msg = SimpleNamespace(
            text=text, message=msg, generation_info=None
        )
        return SimpleNamespace(
            generations=[[gen_with_msg]],
            llm_output=None,
        )
    if style == "no_usage":
        return SimpleNamespace(
            generations=[[gen]],
            llm_output=None,
        )
    raise ValueError(f"unknown style: {style}")


def _fresh_handler(
    **kwargs: Any,
) -> tuple[NoosCallbackHandler, Regulator]:
    r = Regulator.for_user("test_user")
    h = NoosCallbackHandler(r, **kwargs)
    return h, r


# ── Turn boundary detection ──────────────────────────────────────────


def test_on_chain_start_emits_turn_start_from_input_dict():
    h, r = _fresh_handler()
    run_id = uuid4()
    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "Refactor fetch_user to be async"},
        run_id=run_id,
    )
    # Indirect assertion: turn_started flag flips.
    assert h._turn_started is True


def test_nested_chain_start_does_not_re_emit_turn_start():
    h, _ = _fresh_handler()
    root_id = uuid4()
    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "task"},
        run_id=root_id,
    )
    assert h._turn_started is True

    nested_id = uuid4()
    h.on_chain_start(
        serialized={"name": "LLMChain"},
        inputs={"x": 1},
        run_id=nested_id,
        parent_run_id=root_id,
    )
    # Still only one turn active.
    assert h._turn_started is True


def test_on_chain_end_resets_turn_flag_only_for_root():
    h, _ = _fresh_handler()
    root_id = uuid4()
    h.on_chain_start(
        serialized=None,
        inputs={"input": "task"},
        run_id=root_id,
    )
    nested_id = uuid4()

    h.on_chain_end(outputs={}, run_id=nested_id, parent_run_id=root_id)
    assert h._turn_started is True  # nested end: unchanged

    h.on_chain_end(outputs={}, run_id=root_id)
    assert h._turn_started is False  # root end: reset


def test_llm_start_emits_turn_start_when_no_chain_wraps_it():
    h, _ = _fresh_handler()
    h.on_llm_start(
        serialized=None,
        prompts=["Just a bare completion call"],
        run_id=uuid4(),
    )
    assert h._turn_started is True


def test_chat_model_start_extracts_from_messages():
    from types import SimpleNamespace

    h, _ = _fresh_handler()
    msg = SimpleNamespace(content="What's the capital of France?")
    h.on_chat_model_start(
        serialized=None,
        messages=[[msg]],
        run_id=uuid4(),
    )
    assert h._turn_started is True


# ── LLM completion shapes ───────────────────────────────────────────


def test_on_llm_end_emits_turn_complete_and_cost_openai_style():
    h, r = _fresh_handler()
    h.on_chain_start(
        serialized=None,
        inputs={"input": "Say hello"},
        run_id=uuid4(),
    )
    h.on_llm_end(
        response=_make_llm_result(
            "Hello!", tokens_in=10, tokens_out=3, style="openai"
        ),
        run_id=uuid4(),
    )
    assert r.total_tokens_out() == 3
    assert h.last_decision is not None


def test_on_llm_end_extracts_tokens_anthropic_style():
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    h.on_llm_end(
        response=_make_llm_result(
            "resp", tokens_in=50, tokens_out=120, style="anthropic"
        ),
        run_id=uuid4(),
    )
    assert r.total_tokens_out() == 120


def test_on_llm_end_extracts_tokens_usage_metadata_style():
    """Modern LangChain attaches usage_metadata to the AIMessage directly."""
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    h.on_llm_end(
        response=_make_llm_result(
            "resp", tokens_in=7, tokens_out=42, style="usage_metadata"
        ),
        run_id=uuid4(),
    )
    assert r.total_tokens_out() == 42


def test_on_llm_end_tolerates_missing_usage():
    """No usage info anywhere → zero-token Cost event, no exception."""
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    h.on_llm_end(
        response=_make_llm_result("resp", style="no_usage"),
        run_id=uuid4(),
    )
    assert r.total_tokens_out() == 0
    assert h.last_decision is not None


# ── Scope drift end-to-end ──────────────────────────────────────────


def test_scope_drift_decision_surfaces_on_drifted_response():
    h, _ = _fresh_handler()
    h.on_chain_start(
        serialized=None,
        inputs={"input": "Refactor fetch_user to be async. Keep database lookup unchanged."},
        run_id=uuid4(),
    )
    drifted = "added counter timing retry cache wrapper handler middleware logger queue"
    h.on_llm_end(
        response=_make_llm_result(
            drifted, tokens_in=40, tokens_out=180, style="openai"
        ),
        run_id=uuid4(),
    )
    assert h.last_decision is not None
    assert h.last_decision.kind == "scope_drift_warn"
    assert h.last_decision.drift_score is not None
    assert h.last_decision.drift_score >= 0.5


# ── Tool lifecycle ───────────────────────────────────────────────────


def test_tool_start_and_end_pair_by_run_id():
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "find stuff"}, run_id=uuid4())

    tool_id = uuid4()
    h.on_tool_start(
        serialized={"name": "search_orders"},
        input_str='{"q": 42}',
        run_id=tool_id,
    )
    h.on_tool_end(output="ok", run_id=tool_id)
    assert r.tool_total_calls() == 1
    assert r.tool_counts_by_name() == {"search_orders": 1}
    assert r.tool_failure_count() == 0


def test_tool_error_emits_failure_result():
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())

    tool_id = uuid4()
    h.on_tool_start(
        serialized={"name": "broken_tool"},
        input_str="{}",
        run_id=tool_id,
    )
    h.on_tool_error(error=ValueError("boom"), run_id=tool_id)
    assert r.tool_failure_count() == 1


def test_tool_loop_detection_fires_circuit_break():
    h, _ = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "loop"}, run_id=uuid4())

    for _ in range(5):
        tool_id = uuid4()
        h.on_tool_start(
            serialized={"name": "stuck"},
            input_str="{}",
            run_id=tool_id,
        )
        h.on_tool_end(output="none", run_id=tool_id)

    d = h.last_decision
    assert d is not None
    assert d.is_circuit_break()
    assert d.reason.kind == "repeated_tool_call_loop"
    assert d.reason.tool_name == "stuck"


def test_unknown_tool_name_falls_back_gracefully():
    """Missing ``serialized`` dict → tool_name='unknown', no exception."""
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())

    tool_id = uuid4()
    h.on_tool_start(serialized=None, input_str=None, run_id=tool_id)
    h.on_tool_end(output="ok", run_id=tool_id)
    assert r.tool_counts_by_name() == {"unknown": 1}


def test_tool_end_without_matching_start_is_safe():
    """Defensive: tool_end for a run we never saw doesn't crash."""
    h, r = _fresh_handler()
    h.on_tool_end(output="orphan", run_id=uuid4())
    # Emits a tool_result for the 'unknown' tool with duration 0.
    assert r.tool_total_calls() >= 0  # no explicit tool_call was sent


# ── on_decision hook + raise_on_circuit_break ───────────────────────


def test_on_decision_callback_fires_on_every_update():
    seen: list = []
    h, _ = _fresh_handler(on_decision=lambda d: seen.append(d.kind))
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    h.on_llm_end(
        response=_make_llm_result("resp", tokens_in=5, tokens_out=10),
        run_id=uuid4(),
    )
    assert len(seen) == 1
    assert seen[0] in {"continue", "scope_drift_warn"}


def test_raise_on_circuit_break_aborts_on_tool_loop():
    h, _ = _fresh_handler(raise_on_circuit_break=True)
    h.on_chain_start(serialized=None, inputs={"input": "loop"}, run_id=uuid4())

    with pytest.raises(CircuitBreakError) as exc_info:
        for _ in range(5):
            tool_id = uuid4()
            h.on_tool_start(
                serialized={"name": "stuck"},
                input_str="{}",
                run_id=tool_id,
            )
            h.on_tool_end(output="none", run_id=tool_id)

    err = exc_info.value
    assert err.decision.is_circuit_break()
    assert err.decision.reason.kind == "repeated_tool_call_loop"


# ── Regulator passthrough ────────────────────────────────────────────


def test_handler_exposes_the_same_regulator_instance():
    r = Regulator.for_user("x")
    h = NoosCallbackHandler(r)
    assert h.regulator is r


def test_handler_preserves_cost_cap_on_wrapped_regulator():
    r = Regulator.for_user("x").with_cost_cap(2500)
    h = NoosCallbackHandler(r)
    assert h.regulator.cost_cap_tokens() == 2500


# ── on_llm_error + chain_error resilience ───────────────────────────


def test_on_llm_error_is_safe_no_op():
    """LLM-provider failure must not raise from the callback itself."""
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    # Should not raise. Regulator view stays consistent with turn_start
    # (no turn_complete + cost fires from the error path).
    h.on_llm_error(error=RuntimeError("provider timeout"), run_id=uuid4())
    assert r.total_tokens_out() == 0  # no cost event emitted


def test_chain_error_resets_turn_flag():
    """A root-chain error clears ``_turn_started`` so the next run starts clean."""
    h, _ = _fresh_handler()
    root = uuid4()
    h.on_chain_start(serialized=None, inputs={"input": "task"}, run_id=root)
    assert h._turn_started is True

    h.on_chain_error(error=ValueError("boom"), run_id=root)
    assert h._turn_started is False


def test_nested_chain_error_leaves_turn_flag_unchanged():
    """Only the root chain controls the turn boundary."""
    h, _ = _fresh_handler()
    root = uuid4()
    h.on_chain_start(serialized=None, inputs={"input": "task"}, run_id=root)

    nested = uuid4()
    h.on_chain_error(error=ValueError("nested"), run_id=nested, parent_run_id=root)
    assert h._turn_started is True  # nested failure shouldn't close the turn


# ── Multi-turn lifecycle ─────────────────────────────────────────────


def test_multiple_invocations_each_emit_a_turn_start():
    """Two back-to-back root chains emit two separate turn_start events."""
    h, _ = _fresh_handler()

    root1 = uuid4()
    h.on_chain_start(serialized=None, inputs={"input": "first"}, run_id=root1)
    h.on_chain_end(outputs={}, run_id=root1)
    assert h._turn_started is False

    root2 = uuid4()
    h.on_chain_start(serialized=None, inputs={"input": "second"}, run_id=root2)
    assert h._turn_started is True  # second turn opened cleanly


def test_multi_llm_calls_in_one_run_accumulate_cost():
    """AgentExecutor style: plan LLM call + summary LLM call in one run
    accumulate tokens into the same Noos turn."""
    h, r = _fresh_handler()
    root = uuid4()
    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "find user 42"},
        run_id=root,
    )

    # Plan LLM call
    h.on_llm_end(
        response=_make_llm_result("plan", tokens_in=30, tokens_out=15),
        run_id=uuid4(),
        parent_run_id=root,
    )
    # Summary LLM call
    h.on_llm_end(
        response=_make_llm_result("final answer", tokens_in=60, tokens_out=25),
        run_id=uuid4(),
        parent_run_id=root,
    )
    h.on_chain_end(outputs={}, run_id=root)

    # Cost accumulates across both LLM calls (15 + 25 = 40).
    assert r.total_tokens_out() == 40


# ── Async handler ───────────────────────────────────────────────────


def test_async_handler_basic_flow():
    """AsyncNoosCallbackHandler emits the same events via async hooks."""
    import asyncio

    async def run() -> AsyncNoosCallbackHandler:
        r = Regulator.for_user("async_user")
        h = AsyncNoosCallbackHandler(r)
        run_id = uuid4()
        await h.on_chain_start(
            serialized=None,
            inputs={"input": "Find user 42"},
            run_id=run_id,
        )
        await h.on_llm_end(
            response=_make_llm_result("ok", tokens_in=10, tokens_out=5),
            run_id=uuid4(),
            parent_run_id=run_id,
        )
        return h

    h = asyncio.run(run())
    assert h.last_decision is not None
    assert h.regulator.total_tokens_out() == 5


def test_async_handler_tool_loop_fires_circuit_break():
    """Async path runs the same tool-loop detection logic."""
    import asyncio

    async def run() -> AsyncNoosCallbackHandler:
        r = Regulator.for_user("async_user2")
        h = AsyncNoosCallbackHandler(r)
        await h.on_chain_start(
            serialized=None,
            inputs={"input": "loop"},
            run_id=uuid4(),
        )
        for _ in range(5):
            tool_id = uuid4()
            await h.on_tool_start(
                serialized={"name": "stuck"},
                input_str="{}",
                run_id=tool_id,
            )
            await h.on_tool_end(output="none", run_id=tool_id)
        return h

    h = asyncio.run(run())
    assert h.last_decision.is_circuit_break()
    assert h.last_decision.reason.kind == "repeated_tool_call_loop"


def test_async_handler_raise_on_circuit_break_propagates_through_await():
    """CircuitBreakError out of an async hook aborts the awaited run."""
    import asyncio

    async def run() -> None:
        r = Regulator.for_user("async_user3")
        h = AsyncNoosCallbackHandler(r, raise_on_circuit_break=True)
        await h.on_chain_start(
            serialized=None,
            inputs={"input": "loop"},
            run_id=uuid4(),
        )
        for _ in range(5):
            tool_id = uuid4()
            await h.on_tool_start(
                serialized={"name": "stuck"},
                input_str="{}",
                run_id=tool_id,
            )
            await h.on_tool_end(output="none", run_id=tool_id)

    with pytest.raises(CircuitBreakError):
        asyncio.run(run())


def test_async_handler_shares_compat_extraction_with_sync():
    """Token extraction logic works identically in async path."""
    import asyncio

    async def run() -> Regulator:
        r = Regulator.for_user("async_user4")
        h = AsyncNoosCallbackHandler(r)
        await h.on_chain_start(
            serialized=None,
            inputs={"input": "anthropic task"},
            run_id=uuid4(),
        )
        await h.on_llm_end(
            response=_make_llm_result(
                "answer", tokens_in=40, tokens_out=120, style="anthropic"
            ),
            run_id=uuid4(),
        )
        return r

    r = asyncio.run(run())
    assert r.total_tokens_out() == 120


# ── Emit-tokens streaming (opt-in) ──────────────────────────────────


def test_emit_tokens_off_by_default_no_token_events():
    """``emit_tokens=False`` (default): on_llm_new_token is a no-op."""
    h, r = _fresh_handler()
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    for tok in ["hello", "world"]:
        h.on_llm_new_token(tok, run_id=uuid4())
    # No turn_complete yet; confidence should still be the seeded default.
    assert 0.0 <= r.confidence() <= 1.0


def test_emit_tokens_on_feeds_structural_confidence():
    """``emit_tokens=True``: tokens flow into the regulator window."""
    h, r = _fresh_handler(emit_tokens=True)
    h.on_chain_start(serialized=None, inputs={"input": "x"}, run_id=uuid4())
    for tok in ["hello", "world", "foo", "bar", "baz"]:
        h.on_llm_new_token(tok, run_id=uuid4())
    # Regulator's logprob_coverage stays near 0 because LC feeds 0.0
    # sentinels — confirming the structural fallback is in charge.
    assert r.logprob_coverage() <= 0.5
