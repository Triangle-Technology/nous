"""CrewAI agent with Noos halt protection via the LangChain callback path.

CrewAI 0.80+ accepts LangChain-compatible LLM instances on its ``Agent``
class. Pass :class:`NoosCallbackHandler` to the LLM constructor and the
regulator observes the same ``on_llm_start`` / ``on_llm_end`` /
``on_tool_start`` / ``on_tool_end`` stream that LangChain agents fire —
no separate ``noos-crewai`` package needed.

Tool-loop detection depends on the underlying LLM's tool-call events
propagating through LangChain callbacks. For Anthropic (via
``langchain-anthropic``) and OpenAI (via ``langchain-openai``) this
works out of the box; other LLM wrappers may surface tools via custom
hooks and require a dedicated adapter.

Run::

    pip install noos noos-langchain crewai crewai-tools langchain-anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/crewai_agent.py
"""
from __future__ import annotations

import os
import sys

from noos import Regulator
from noos_langchain import CircuitBreakError, NoosCallbackHandler


def main() -> int:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "ANTHROPIC_API_KEY is not set. This example needs a real Anthropic key.",
            file=sys.stderr,
        )
        return 1

    try:
        from crewai import Agent, Crew, Task
        from langchain_anthropic import ChatAnthropic
        from langchain_core.tools import tool
    except ImportError as e:
        print(
            f"Missing optional dependency: {e}\n"
            "Install with: pip install crewai crewai-tools langchain-anthropic",
            file=sys.stderr,
        )
        return 1

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base. Returns 'no matching entries' for all queries."""
        return "no matching entries"

    # ── Wire the regulator via the LLM's callbacks ───────────────────
    regulator = Regulator.for_user("crewai_demo").with_cost_cap(5_000)
    handler = NoosCallbackHandler(
        regulator,
        raise_on_circuit_break=True,
        on_decision=lambda d: (
            print(f"    [Noos] {d.kind}") if d.kind != "continue" else None
        ),
    )

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    llm = ChatAnthropic(model=model_name, temperature=0, callbacks=[handler])

    # ── Build the crew ───────────────────────────────────────────────
    researcher = Agent(
        role="Researcher",
        goal="Find information about user accounts",
        backstory="You are a careful researcher who avoids repeating futile queries.",
        llm=llm,
        tools=[search_knowledge_base],
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Search the knowledge base for information about user 42. "
            "If the first two searches yield nothing, report failure."
        ),
        agent=researcher,
        expected_output="A one-paragraph summary of findings or a failure report.",
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=True)

    # ── Run ──────────────────────────────────────────────────────────
    try:
        result = crew.kickoff()
        print("\n── Result ────────────────────────────────────────────")
        print(result)
    except CircuitBreakError as e:
        print("\n── Crew halted by Noos ───────────────────────────────")
        print(f"Reason:     {e.decision.reason.kind}")
        print(f"Suggestion: {e.decision.suggestion}")
        if e.decision.reason.kind == "repeated_tool_call_loop":
            print(f"Looping tool: {e.decision.reason.tool_name}")

    print(f"\nNoos total_tokens_out: {handler.regulator.total_tokens_out()}")
    print(f"Noos metrics:          {dict(handler.regulator.metrics_snapshot())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
