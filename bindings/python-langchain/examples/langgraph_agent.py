"""LangGraph React agent with Noos tool-loop + cost protection.

LangGraph uses the same ``langchain_core`` callback plumbing underneath
``StateGraph.compile()``, so :class:`NoosCallbackHandler` works with
LangGraph graphs unchanged — pass it via the ``config={"callbacks": [...]}``
argument to ``invoke`` / ``ainvoke``.

This demo builds a classic React-style agent with one toy tool that
always returns empty results. Without a regulator the model retries
indefinitely; with Noos the agent halts once the same tool has been
called five times in a row (``RepeatedToolCallLoop``) or once cumulative
``tokens_out`` exceeds the cost cap (``CostCapReached``).

Run::

    pip install noos noos-langchain langgraph langchain-openai
    export OPENAI_API_KEY=sk-...
    python examples/langgraph_agent.py

Use ``examples/basic_smoke.py`` for a zero-network demo that exercises
the same handler against fabricated LangChain payloads.
"""
from __future__ import annotations

import os
import sys

from noos import Regulator
from noos_langchain import CircuitBreakError, NoosCallbackHandler


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print(
            "OPENAI_API_KEY is not set. This example needs a real OpenAI key.\n"
            "For a no-network demo, run examples/basic_smoke.py instead.",
            file=sys.stderr,
        )
        return 1

    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        print(
            f"Missing optional dependency: {e}\n"
            "Install with: pip install langgraph langchain-openai",
            file=sys.stderr,
        )
        return 1

    # ── Toy tool that always disappoints the agent ───────────────────
    call_log: list[str] = []

    @tool
    def search_orders(query: str) -> str:
        """Search the order database. Always returns 'no results'."""
        call_log.append(f"search_orders({query})")
        return "no results"

    # ── Build the LangGraph React agent ──────────────────────────────
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    agent = create_react_agent(
        ChatOpenAI(model=model_name, temperature=0),
        tools=[search_orders],
    )

    # ── Wire the regulator via `config={"callbacks": [...]}` ─────────
    regulator = Regulator.for_user("langgraph_demo").with_cost_cap(5_000)
    handler = NoosCallbackHandler(
        regulator,
        raise_on_circuit_break=True,
        on_decision=lambda d: (
            print(f"    [Noos] {d.kind}") if d.kind != "continue" else None
        ),
    )

    # ── Run ──────────────────────────────────────────────────────────
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content="Find order 42")]},
            config={"callbacks": [handler]},
        )
        print("\n── Final state ───────────────────────────────────────")
        print(result["messages"][-1].content)
    except CircuitBreakError as e:
        print("\n── Agent halted by Noos ──────────────────────────────")
        print(f"Reason:     {e.decision.reason.kind}")
        print(f"Suggestion: {e.decision.suggestion}")
        if e.decision.reason.kind == "repeated_tool_call_loop":
            print(f"Looping tool: {e.decision.reason.tool_name}")
            print(f"Consecutive:  {e.decision.reason.consecutive_count}")
        elif e.decision.reason.kind == "cost_cap_reached":
            print(
                f"Tokens spent: {e.decision.reason.tokens_spent} / "
                f"{e.decision.reason.tokens_cap}"
            )

    print(f"\nTotal tool invocations: {len(call_log)}")
    print(f"Noos total_tokens_out:  {handler.regulator.total_tokens_out()}")
    print(f"Noos metrics_snapshot:  {dict(handler.regulator.metrics_snapshot())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
