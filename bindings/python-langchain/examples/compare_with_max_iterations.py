"""Concrete comparison: Noos vs ``max_iterations`` on the tool-loop pathology.

Simulates the November 2025 LangChain tool-loop incident pattern — an
agent calling the same tool (``search_orders``) with varied args, each
returning empty results. The agent keeps retrying because every call
*succeeds* at the protocol level; it's the agent's decision to
re-invoke that's pathological.

Two control strategies compared against the same synthetic stream:

1. **``max_iterations=N``** — LangChain / CrewAI / AutoGen's built-in
   guard. Halts only on total step count regardless of what's
   repeating. Lets N calls fire before halting.
2. **``NoosCallbackHandler`` + ``raise_on_circuit_break=True``** —
   fires ``CircuitBreak(RepeatedToolCallLoop)`` structurally once the
   same tool name hits 5 consecutive calls (``TOOL_LOOP_THRESHOLD``
   from the Rust crate). Agent aborts at the 5th call regardless of
   the configured iteration cap.

Both halt the pathology, but Noos halts *sooner* and cheaper. Run this
to get the exact comparison numbers for your own reporting.

Run::

    pip install noos noos-langchain
    python examples/compare_with_max_iterations.py

No LLM or network required. Deterministic.
"""
from __future__ import annotations

from uuid import uuid4

from noos import LLMEvent, Regulator
from noos_langchain import CircuitBreakError, NoosCallbackHandler

# Synthetic tool cost — each tool call triggers one LLM round-trip of
# ~100 output tokens (a realistic estimate for a tool-result →
# reasoning-step → next tool_call chain with a chat model).
TOKENS_PER_CALL = 100
# Scenario: agent would loop 20 times before max_iterations halts. A
# higher iteration cap widens Noos's savings.
MAX_ITERATIONS = 20


def simulate_max_iterations(max_iterations: int) -> dict:
    """Baseline: LC halts after N iterations; no per-signature check."""
    return {
        "iterations_before_halt": max_iterations,
        "tokens_out_spent": max_iterations * TOKENS_PER_CALL,
        "halt_reason": f"max_iterations={max_iterations} reached",
    }


def simulate_with_noos(max_iterations: int) -> dict:
    """Noos handler fires ``CircuitBreak(RepeatedToolCallLoop)`` structurally."""
    regulator = Regulator.for_user("demo_agent")
    handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)

    root_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "Find orders for user 42"},
        run_id=root_id,
    )

    halted_at = None
    halt_reason = None

    try:
        for i in range(max_iterations):
            tool_id = uuid4()
            handler.on_tool_start(
                serialized={"name": "search_orders"},
                input_str=f'{{"user_id": 42, "attempt": {i}}}',
                run_id=tool_id,
            )
            handler.on_tool_end(output="[]", run_id=tool_id)
            # Simulated LLM round-trip emits cost.
            handler.regulator.on_event(
                LLMEvent.cost(
                    tokens_in=10,
                    tokens_out=TOKENS_PER_CALL,
                    wallclock_ms=0,
                )
            )
    except CircuitBreakError as e:
        halted_at = (
            regulator.tool_total_calls()
            if hasattr(regulator, "tool_total_calls")
            else None
        )
        halt_reason = (
            f"{e.decision.reason.kind} "
            f"(consecutive_count={e.decision.reason.consecutive_count})"
        )

    tokens_spent = (halted_at or max_iterations) * TOKENS_PER_CALL
    return {
        "iterations_before_halt": halted_at or max_iterations,
        "tokens_out_spent": tokens_spent,
        "halt_reason": halt_reason or "did not fire (unexpected)",
    }


def main() -> None:
    baseline = simulate_max_iterations(MAX_ITERATIONS)
    with_noos = simulate_with_noos(MAX_ITERATIONS)

    print("=" * 70)
    print(" Tool-loop pathology: repeated same-name tool call, empty results ")
    print("=" * 70)
    print(f" Scenario: {MAX_ITERATIONS} iterations max, {TOKENS_PER_CALL} output tokens/call")
    print()
    print(" Without Noos (max_iterations only):")
    print(f"   iterations before halt: {baseline['iterations_before_halt']}")
    print(f"   tokens_out spent:       {baseline['tokens_out_spent']}")
    print(f"   halt reason:            {baseline['halt_reason']}")
    print()
    print(" With Noos CircuitBreak(RepeatedToolCallLoop):")
    print(f"   iterations before halt: {with_noos['iterations_before_halt']}")
    print(f"   tokens_out spent:       {with_noos['tokens_out_spent']}")
    print(f"   halt reason:            {with_noos['halt_reason']}")
    print()

    savings_pct = (
        (baseline["tokens_out_spent"] - with_noos["tokens_out_spent"])
        / max(baseline["tokens_out_spent"], 1)
        * 100.0
    )
    speedup = baseline["iterations_before_halt"] / max(
        with_noos["iterations_before_halt"], 1
    )

    print(" Comparison:")
    print(f"   Noos halts {speedup:.1f}x sooner ({with_noos['iterations_before_halt']} vs "
          f"{baseline['iterations_before_halt']} iterations)")
    print(f"   Noos saves {savings_pct:.0f}% of the tokens spent on the pathology")
    print()
    print(" The savings scale linearly with the iteration cap. A higher")
    print(" max_iterations (e.g. the unbounded deployments behind the")
    print(" $47k November 2025 LangChain incident) widens the gap")
    print(" proportionally — Noos still halts at the 5th consecutive")
    print(" call regardless of how much rope max_iterations gave.")


if __name__ == "__main__":
    main()
