"""Wire NoosCallbackHandler into a real LangChain OpenAI tools agent.

Requires ``OPENAI_API_KEY`` in the environment. Uses ``gpt-4o-mini`` by
default; override via ``OPENAI_MODEL`` env var.

The demo defines two toy tools that tempt the agent into a retry loop,
then caps cost at 5_000 output tokens. When the agent tool-loops or
exhausts the cap, :class:`CircuitBreakError` aborts the ``invoke``.

Run::

    pip install noos noos-langchain langchain langchain-openai
    export OPENAI_API_KEY=sk-...
    python examples/openai_tools_agent.py
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
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(
            f"Missing optional dependency: {e}\n"
            "Install with: pip install langchain langchain-openai",
            file=sys.stderr,
        )
        return 1

    # ── Toy tools that tempt the agent into a loop ───────────────────
    call_log: list[str] = []

    @tool
    def search_orders(query: str) -> str:
        """Search the order database. Returns empty results for any query."""
        call_log.append(f"search_orders({query})")
        return "[]"  # always empty — pushes agent to retry

    @tool
    def lookup_user(user_id: int) -> str:
        """Look up a user by ID. Returns 'unknown user' for any id."""
        call_log.append(f"lookup_user({user_id})")
        return "unknown user"

    tools = [search_orders, lookup_user]

    # ── Build the agent ──────────────────────────────────────────────
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided tools to answer. "
                "If tools return empty results, stop after trying twice.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # ── Wire the regulator ───────────────────────────────────────────
    regulator = Regulator.for_user("demo_user").with_cost_cap(5_000)
    handler = NoosCallbackHandler(
        regulator,
        raise_on_circuit_break=True,
        on_decision=lambda d: (
            print(f"    [Noos] decision: {d.kind}") if d.kind != "continue" else None
        ),
    )

    # ── Run ──────────────────────────────────────────────────────────
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
