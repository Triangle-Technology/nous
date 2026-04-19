# noos-langchain

LangChain + LangGraph callback adapter for [Noos](https://pypi.org/project/noos/) — the pre-delivery reliability regulator for LLM agents.

Drop `NoosCallbackHandler` into any agent built on `langchain-core` and get tool-loop halts, cost circuit breaks, scope-drift warnings, and procedural correction memory from the agent's existing event stream. No instrumentation inside your chains.

**Overhead**: one `on_event` dispatch is 20–250 ns; a `decide()` check is ~2 µs. Source: [Noos criterion benchmarks, Session 33](https://github.com/Triangle-Technology/noos/blob/main/benches/regulator.rs). An LLM call at 200 tokens/sec runs ~500 ms per turn — the regulator sits six orders of magnitude below that floor.

## Install

```bash
pip install noos noos-langchain
```

Requires Python ≥3.9, `noos>=0.1.0`, `langchain-core>=0.3.0`. Works with LangChain, LangGraph, CrewAI, and anything else using the `langchain-core` callback interface.

## Quick start — LangChain

```python
from noos import Regulator
from noos_langchain import NoosCallbackHandler, CircuitBreakError
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. Construct a regulator per user (or per session). Persist across
#    processes via export_json / from_json.
regulator = Regulator.for_user("alice").with_cost_cap(10_000)

# 2. Wire the handler via the standard callbacks list.
handler = NoosCallbackHandler(
    regulator,
    raise_on_circuit_break=True,   # abort agent on halt decisions
)

agent = create_openai_tools_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[...],
    prompt=ChatPromptTemplate.from_messages([...]),
)
executor = AgentExecutor(agent=agent, tools=[...], callbacks=[handler])

try:
    result = executor.invoke({"input": "Find order #42 and email the summary."})
except CircuitBreakError as e:
    print(f"Agent halted: {e.decision.reason.kind}")
    print(f"Suggestion: {e.decision.suggestion}")

# 3. Inspect the last decision at any time.
if handler.last_decision and handler.last_decision.is_scope_drift():
    print(f"Scope drift score: {handler.last_decision.drift_score:.2f}")
```

## Quick start — LangGraph

LangGraph uses the same `langchain-core` callback plumbing, so the handler works unchanged:

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from noos import Regulator
from noos_langchain import NoosCallbackHandler

agent = create_react_agent(llm, tools=[...])

handler = NoosCallbackHandler(
    Regulator.for_user("alice").with_cost_cap(10_000),
    raise_on_circuit_break=True,
)

agent.invoke(
    {"messages": [HumanMessage(content="Find order 42")]},
    config={"callbacks": [handler]},
)
```

Full demo: [`examples/langgraph_agent.py`](examples/langgraph_agent.py).

## Quick start — async

Use `AsyncNoosCallbackHandler` for `ainvoke` / `astream` / LangGraph async:

```python
from noos_langchain import AsyncNoosCallbackHandler

handler = AsyncNoosCallbackHandler(regulator)
await agent_executor.ainvoke(
    {"input": "..."},
    config={"callbacks": [handler]},
)
```

## What the handler observes

| LangChain hook | Noos event | Decision impact |
|----------------|------------|-----------------|
| `on_chain_start` (root) | `turn_start` | Resets per-turn scope + cost + tool-stats |
| `on_chat_model_start` / `on_llm_start` | `turn_start` (fallback) | — |
| `on_llm_new_token` | `token` (opt-in via `emit_tokens=True`) | Feeds confidence signal |
| `on_llm_end` | `turn_complete` + `cost` | Updates scope-drift, cost-cap, quality-decline predicates |
| `on_llm_error` | — | No event; regulator view stays consistent with last `turn_start` |
| `on_tool_start` | `tool_call` | Per-turn tool stats + consecutive-same-tool counter |
| `on_tool_end` | `tool_result(success=True, duration_ms=…)` | Closes the tool-call signature |
| `on_tool_error` | `tool_result(success=False, error_summary=…)` | Feeds tool failure count |
| `on_chain_end` / `on_chain_error` (root) | — | Clears turn flag; no event emitted |

After each event that can change the regulator's decision, the handler calls `regulator.decide()` and stores the result on `handler.last_decision`. Applications choose how to consume:

1. **Poll after `invoke`** — read `handler.last_decision` when execution completes.
2. **React mid-run** — pass `on_decision=callback` to observe every decision transition.
3. **Abort on halt** — pass `raise_on_circuit_break=True` to raise `CircuitBreakError` the moment a circuit-break decision fires.

## Migration patterns

### From `recursion_limit` → tool-loop detection

LangChain / CrewAI / AutoGen's `recursion_limit` caps total step count regardless of what repeats. That missed the [$47k LangChain incident](https://github.com/langchain-ai/langchain/issues/13099) where the agent called the same tool with the same arguments 400+ times.

```python
# Before: crude step cap
executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=20,  # still lets the same tool fire 20×
)

# After: fires on the signature of the pathology
handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)
executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
# CircuitBreak(RepeatedToolCallLoop) fires after 5 consecutive calls
# to the same tool name, regardless of arguments.
```

### From `tenacity` / manual retry → cost × quality compound halt

```python
# Before: retry N times regardless of quality trend
@retry(stop=stop_after_attempt(5))
def call_with_retry(): ...

# After: halts when BOTH cost cap AND quality decline trip
regulator = Regulator.for_user(uid).with_cost_cap(5_000)
handler = NoosCallbackHandler(regulator, raise_on_circuit_break=True)
# CircuitBreak(CostCapReached) fires when cumulative tokens_out ≥ cap
# AND mean recent quality drops below a threshold.
```

### Paired with Langfuse / Helicone observability

Noos decides pre-delivery; observability tools log post-delivery. Both fit:

```python
from langfuse.callback import CallbackHandler as LangfuseHandler
from noos_langchain import NoosCallbackHandler

# Both in the callbacks list — Langfuse records traces,
# Noos emits halt/warn decisions before the response ships.
executor.invoke(
    {"input": "..."},
    config={"callbacks": [LangfuseHandler(), NoosCallbackHandler(regulator)]},
)

# Pipe Noos metrics into your existing stack:
for key, value in handler.regulator.metrics_snapshot().items():
    statsd_client.gauge(key, value)  # noos.confidence, noos.total_tokens_out, ...
```

## Behavioural notes

- **One root chain = one Noos turn.** Nested chains (`LLMChain` inside `AgentExecutor`, sub-graph nodes in LangGraph) stay in the same turn; their costs sum; the final `on_llm_end` wins for scope-drift scoring.
- **Costs are cumulative across all LLM calls within the turn.** The cost cap (set via `regulator.with_cost_cap(n)`) trips when cumulative `tokens_out` crosses `n` AND recent quality is below threshold — see the [Noos regulator guide](https://github.com/Triangle-Technology/noos/blob/main/docs/regulator-guide.md).
- **Tool-loop detection uses tool name only.** Five consecutive calls to the same tool name (regardless of arguments) trigger `CircuitBreak(RepeatedToolCallLoop)`. Threshold is fixed at 5 per the Rust crate's `TOOL_LOOP_THRESHOLD` constant.
- **`on_llm_error` is a no-op.** LangChain propagates the error; the regulator view stays consistent with the last `turn_start`. The next root `on_chain_start` opens a fresh turn cleanly.
- **Token-per-token emission (`emit_tokens=True`) is off by default.** LangChain doesn't surface per-token logprobs through callbacks, so the regulator uses its structural confidence fallback regardless. Enable only if you need pre-turn-complete decisions.
- **Not thread-safe.** A `Regulator` is single-threaded. Create one handler per agent run. For concurrent executions, wrap each call in its own handler + regulator, then merge state via `export_json` afterward.

## Persistence across sessions

Procedural correction memory and learned strategies survive process restarts:

```python
# End of session — persist the regulator's state.
snapshot = regulator.export_json()
redis.set(f"noos:{user_id}", snapshot)

# Next session — restore before constructing the handler.
saved = redis.get(f"noos:{user_id}")
regulator = Regulator.from_json(saved) if saved else Regulator.for_user(user_id)
```

Correction patterns require at least 3 `user_correction` events on the same topic cluster before they fire a `ProceduralWarning`. Implicit-correction detection (fast same-cluster retries without explicit `UserCorrection`) is available via `regulator.with_implicit_correction_window_secs(30.0)`; see [regulator-guide.md §6.5](https://github.com/Triangle-Technology/noos/blob/main/docs/regulator-guide.md) for the gotchas.

## Examples

- [`examples/basic_smoke.py`](examples/basic_smoke.py) — runs the handler against fabricated LangChain payloads. No LLM or API key required.
- [`examples/openai_tools_agent.py`](examples/openai_tools_agent.py) — full OpenAI tools agent with tool-loop halt protection. Requires `OPENAI_API_KEY`.
- [`examples/anthropic_tools_agent.py`](examples/anthropic_tools_agent.py) — same shape against Claude Haiku via `langchain-anthropic`. Requires `ANTHROPIC_API_KEY`.
- [`examples/langgraph_agent.py`](examples/langgraph_agent.py) — LangGraph React agent with cost cap + tool-loop halt. Requires `OPENAI_API_KEY`.
- [`examples/crewai_agent.py`](examples/crewai_agent.py) — CrewAI agent via the LangChain LLM callback path — no separate `noos-crewai` package needed. Requires `ANTHROPIC_API_KEY`.

## Related

- [`noos`](https://pypi.org/project/noos/) — the core regulator (Python bindings over the Rust crate).
- [`noos` on crates.io](https://crates.io/crates/noos) — the Rust crate.
- [Migration guide (crate)](https://github.com/Triangle-Technology/noos/blob/main/docs/migrating.md) — LangChain `recursion_limit` / `tenacity` / Mem0 / Langfuse → Noos recipes.
- [Regulator guide](https://github.com/Triangle-Technology/noos/blob/main/docs/regulator-guide.md) — event lifecycle, Decision recipes, P10 priority rules.

## License

MIT.
