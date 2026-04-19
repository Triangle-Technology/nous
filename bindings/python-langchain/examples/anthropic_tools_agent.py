"""End-to-end LangChain + Anthropic agent with Noos halt protection.

Uses Claude Haiku via ``langchain-anthropic``'s ``ChatAnthropic`` and
the provider-agnostic ``create_tool_calling_agent`` factory. Two toy
tools always return useless data, which tempts the model into a retry
loop — Noos halts on the 5th consecutive same-tool call.

This is the first example that exercises the full pipeline against a
real LLM (Haiku was used in the Session 36 real-judge eval); run it
once with a live key to validate the adapter end-to-end.

Run::

    pip install noos noos-langchain langchain langchain-anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/anthropic_tools_agent.py

For a no-network version, see ``examples/basic_smoke.py``.
"""
from __future__ import annotations

import os
import sys

from noos import Regulator
from noos_langchain import CircuitBreakError, NoosCallbackHandler


def main() -> int:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "ANTHROPIC_API_KEY is not set. This example needs a real Anthropic key.\n"
            "For a no-network demo, run examples/basic_smoke.py instead.",
            file=sys.stderr,
        )
        return 1

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool
    except ImportError as e:
        print(
            f"Missing optional dependency: {e}\n"
            "Install with: pip install langchain langchain-anthropic",
            file=sys.stderr,
        )
        return 1

    call_log: list[str] = []

    @tool
    def search_orders(query: str) -> str:
        """Search the order database. Always returns 'no results'."""
        call_log.append(f"search_orders({query})")
        return "no results"

    @tool
    def lookup_user(user_id: int) -> str:
        """Look up a user by ID. Always returns 'unknown user'."""
        call_log.append(f"lookup_user({user_id})")
        return "unknown user"

    tools = [search_orders, lookup_user]

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    llm = ChatAnthropic(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided tools to answer. "
                "Stop retrying the same tool after two empty results.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)

    regulator = Regulator.for_user("anthropic_demo").with_cost_cap(5_000)
    handler = NoosCallbackHandler(
        regulator,
        raise_on_circuit_break=True,
        on_decision=lambda d: (
            print(f"    [Noos] {d.kind}") if d.kind != "continue" else None
        ),
    )

    try:
        result = executor.invoke(
            {"input": "Find user 42 and look up their orders."},
            config={"callbacks": [handler]},
        )
        print("\n── Result ────────────────────────────────────────────")
        print(result.get("output", result))
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
    print(f"Noos metrics:           {dict(handler.regulator.metrics_snapshot())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
