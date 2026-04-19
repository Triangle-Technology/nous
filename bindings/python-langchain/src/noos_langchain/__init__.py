"""LangChain callback adapters for Noos.

Wire a Noos :class:`Regulator` into any LangChain or LangGraph agent
by passing one of :class:`NoosCallbackHandler` (sync) or
:class:`AsyncNoosCallbackHandler` (async) to the runnable's
``callbacks=[...]`` list. The handler observes LLM + tool events and
pushes them into the regulator; the application reads
``handler.last_decision`` (or calls ``handler.regulator.decide()``
directly) at any interrupt point.

Sync example::

    from noos import Regulator
    from noos_langchain import NoosCallbackHandler
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    regulator = Regulator.for_user("alice").with_cost_cap(10_000)
    handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)

    agent_executor = AgentExecutor(
        agent=create_openai_tools_agent(ChatOpenAI(), tools, prompt),
        tools=tools,
        callbacks=[handler],
    )
    agent_executor.invoke({"input": "Find order 42"})

Async example::

    from noos_langchain import AsyncNoosCallbackHandler

    handler = AsyncNoosCallbackHandler(regulator)
    await agent_executor.ainvoke(
        {"input": "Find order 42"},
        config={"callbacks": [handler]},
    )

LangGraph example::

    from langgraph.graph import StateGraph
    from noos_langchain import NoosCallbackHandler

    graph = StateGraph(...)
    app = graph.compile()
    handler = NoosCallbackHandler(regulator)
    app.invoke({"input": "..."}, config={"callbacks": [handler]})

In all cases, inspect ``handler.last_decision`` after execution or
pass ``raise_on_circuit_break=True`` to abort the run the moment a
circuit-break decision fires (raises :class:`CircuitBreakError`).
"""

from noos_langchain.callback import (
    AsyncNoosCallbackHandler,
    CircuitBreakError,
    NoosCallbackHandler,
)

__all__ = [
    "NoosCallbackHandler",
    "AsyncNoosCallbackHandler",
    "CircuitBreakError",
    "__version__",
]
__version__ = "0.1.0"
