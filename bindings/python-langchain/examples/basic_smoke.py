"""Exercise the handler with fabricated LangChain payloads.

Four scenarios, no LLM or network required:

1. Scope drift — drifted response triggers ScopeDriftWarn.
2. Tool loop — 5 same-tool calls trigger CircuitBreak(RepeatedToolCallLoop).
3. Cost break — 3 declining-quality turns trip CircuitBreak(CostCapReached).
4. Procedural warning — 3 corrections on same cluster seed a pattern;
   next turn on that cluster fires ProceduralWarning pre-generation.

Run::

    cd bindings/python-langchain
    pip install -e .
    pip install /path/to/noos      # or: pip install noos
    python examples/basic_smoke.py
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from noos import LLMEvent, Regulator
from noos_langchain import NoosCallbackHandler


def _llm_result(text: str, tokens_in: int, tokens_out: int):
    return SimpleNamespace(
        generations=[[SimpleNamespace(text=text, message=None, generation_info=None)]],
        llm_output={
            "token_usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            }
        },
    )


def scope_drift_demo() -> None:
    print("── 1. Scope drift ─────────────────────────────────────────")
    handler = NoosCallbackHandler(Regulator.for_user("alice"))
    root = uuid4()

    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={
            "input": (
                "Refactor fetch_user to be async. "
                "Keep the database lookup logic unchanged."
            )
        },
        run_id=root,
    )
    # 10 drift keywords, zero anchor keywords.
    drifted = "added counter timing retry cache wrapper handler middleware logger queue"
    handler.on_llm_end(
        response=_llm_result(drifted, tokens_in=40, tokens_out=180),
        run_id=uuid4(),
    )

    d = handler.last_decision
    print(f"  last_decision.kind = {d.kind}")
    if d.is_scope_drift():
        print(f"  drift_score       = {d.drift_score:.2f}")
        print(f"  drift_tokens[:5]  = {d.drift_tokens[:5]}")


def tool_loop_demo() -> None:
    print("\n── 2. Tool-call loop ──────────────────────────────────────")
    handler = NoosCallbackHandler(Regulator.for_user("bob"))

    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "Find user 42"},
        run_id=uuid4(),
    )

    for i in range(5):
        tool_id = uuid4()
        handler.on_tool_start(
            serialized={"name": "search_orders"},
            input_str=f'{{"user_id": 42, "attempt": {i}}}',
            run_id=tool_id,
        )
        handler.on_tool_end(output="[]", run_id=tool_id)

    d = handler.last_decision
    print(f"  tool_total_calls = {handler.regulator.tool_total_calls()}")
    print(f"  decision.kind    = {d.kind}")
    if d.is_circuit_break():
        print(f"  reason.kind              = {d.reason.kind}")
        print(f"  reason.tool_name         = {d.reason.tool_name}")
        print(f"  reason.consecutive_count = {d.reason.consecutive_count}")


def cost_break_demo() -> None:
    print("\n── 3. Cost circuit break ─────────────────────────────────")
    regulator = Regulator.for_user("carol").with_cost_cap(1000)
    handler = NoosCallbackHandler(regulator)

    for attempt, q in enumerate([0.50, 0.35, 0.20], start=1):
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": f"optimize query attempt {attempt}"},
            run_id=uuid4(),
        )
        handler.on_llm_end(
            response=_llm_result(f"attempt {attempt}", tokens_in=25, tokens_out=400),
            run_id=uuid4(),
        )
        # Quality feedback has no direct LangChain callback — the app
        # injects it manually when it has a grader.
        handler.regulator.on_event(LLMEvent.quality_feedback(quality=q))
        handler.on_chain_end(outputs={}, run_id=uuid4())

        d = handler.regulator.decide()
        print(
            f"  turn {attempt}: quality={q:.2f} "
            f"cumulative_out={handler.regulator.total_tokens_out()} "
            f"decision={d.kind}"
        )
        if d.is_circuit_break():
            print(f"    reason.kind   = {d.reason.kind}")
            print(f"    suggestion    = {d.suggestion}")
            break


def procedural_memory_demo() -> None:
    print("\n── 4. Procedural correction memory ───────────────────────")
    regulator = Regulator.for_user("dave")
    handler = NoosCallbackHandler(regulator)

    queries_and_corrections = [
        ("Make my auth module async",        "no, stop adding telemetry"),
        ("Refactor auth to support async",   "I said no telemetry"),
        ("Change my auth function to async", "please — no telemetry this time"),
    ]
    for q, correction in queries_and_corrections:
        root = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": q},
            run_id=root,
        )
        handler.on_llm_end(
            response=_llm_result(
                "(async handler with extensive telemetry and logging)",
                tokens_in=20, tokens_out=100,
            ),
            run_id=uuid4(),
        )
        handler.on_chain_end(outputs={}, run_id=root)
        # Corrections land via the regulator directly — LangChain has no
        # direct "user corrected the previous response" hook. Apps that
        # capture thumbs-down / feedback should emit this event.
        handler.regulator.on_event(
            LLMEvent.user_correction(
                correction_message=correction,
                corrects_last=True,
            )
        )

    # Round-trip the regulator to simulate a process restart.
    snapshot = handler.regulator.export_json()
    print(f"  snapshot size       = {len(snapshot)} bytes")
    restored = Regulator.from_json(snapshot)
    handler = NoosCallbackHandler(restored)

    # Next-session turn on the same cluster: warning fires BEFORE the
    # LLM call (no on_llm_end yet), from just the turn_start event.
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "Add async handling to my auth"},
        run_id=uuid4(),
    )
    d = handler.regulator.decide()
    print(f"  post-restore decision = {d.kind}")
    if d.is_procedural_warning():
        for p in d.patterns:
            print(f"    pattern          = {p.pattern_name}")
            print(f"    confidence       = {p.confidence:.2f}")
            print(f"    examples ({len(p.example_corrections)}):")
            for ex in p.example_corrections:
                print(f"      - {ex}")


if __name__ == "__main__":
    scope_drift_demo()
    tool_loop_demo()
    cost_break_demo()
    procedural_memory_demo()
