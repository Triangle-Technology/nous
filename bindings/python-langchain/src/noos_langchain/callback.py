"""LangChain callback handlers that pipe agent events into Noos.

Two concrete handlers share all logic via :class:`_BaseHandlerLogic`:

* :class:`NoosCallbackHandler` — sync. Works with synchronous LangChain
  agents (``AgentExecutor.invoke``), LangGraph graphs
  (``app.invoke(config={"callbacks": [handler]})``), and any other
  callbacks-driven runnable.
* :class:`AsyncNoosCallbackHandler` — async. Inherits from
  :class:`langchain_core.callbacks.base.AsyncCallbackHandler` and wraps
  the same logic in ``async def`` hooks. Use with ``ainvoke`` /
  ``astream`` / LangGraph async flows.

Hook → :class:`LLMEvent` mapping (identical in both variants):

    ``on_chain_start``      → ``LLMEvent.turn_start`` (root chain only)
    ``on_chat_model_start`` → ``LLMEvent.turn_start`` (fallback)
    ``on_llm_start``        → ``LLMEvent.turn_start`` (fallback)
    ``on_llm_new_token``    → ``LLMEvent.token`` (opt-in via ``emit_tokens``)
    ``on_llm_end``          → ``LLMEvent.turn_complete`` + ``LLMEvent.cost``
    ``on_tool_start``       → ``LLMEvent.tool_call``
    ``on_tool_end``         → ``LLMEvent.tool_result(success=True, ...)``
    ``on_tool_error``       → ``LLMEvent.tool_result(success=False, ...)``
    ``on_llm_error``        → no-op (regulator has no failure event)
    ``on_chain_end`` /
    ``on_chain_error`` (root) → clears the internal turn flag

After each event that can change the decision (``on_llm_end``,
``on_tool_end``, ``on_tool_error``), the handler calls
``regulator.decide()`` and stores the result on ``last_decision``.
Three consumption modes:

  * read ``handler.last_decision`` after ``invoke()`` / ``ainvoke()`` returns,
  * pass ``on_decision=callback`` to react mid-run, or
  * set ``raise_on_circuit_break=True`` to abort via
    :class:`CircuitBreakError` the moment a halt fires.

One root chain = one Noos turn. Nested chains, sub-graph nodes, and
multiple LLM calls inside an ``AgentExecutor`` run accumulate into the
same turn — their costs sum; the final ``turn_complete`` wins for
scope-drift scoring.

Thread / async safety: a :class:`noos.Regulator` is ``unsendable``
(single-threaded). Create one handler per agent run. Do not share one
handler across concurrent ``ainvoke`` calls — spawn a fresh handler +
regulator per call, merge via ``export_json`` afterward if you need
cumulative state.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from noos import Decision, LLMEvent, Regulator

from noos_langchain._compat import (
    extract_chat_messages,
    extract_prompts_as_message,
    extract_response_text,
    extract_token_usage,
    extract_user_message,
    tool_name_from_serialized,
)


class CircuitBreakError(RuntimeError):
    """Raised when the regulator emits a ``circuit_break`` decision and
    ``raise_on_circuit_break=True``.

    Carries the offending :class:`noos.Decision` on ``.decision`` so
    ``except`` handlers can inspect the halt reason and surface a useful
    error upward.
    """

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        reason_kind = (
            decision.reason.kind if decision.reason is not None else "unknown"
        )
        suggestion = decision.suggestion or ""
        super().__init__(f"Noos circuit break: {reason_kind} — {suggestion}")


class _BaseHandlerLogic:
    """Shared state and method bodies for sync + async handlers.

    Not a full callback handler on its own — it has no LangChain base
    class. Mix with either :class:`BaseCallbackHandler` (sync) or
    :class:`AsyncCallbackHandler` (async). The handler class supplies
    the LC-shaped hooks; they delegate all real work to the ``_handle_*``
    methods here so the sync and async variants stay in lockstep.
    """

    regulator: Regulator
    last_decision: Optional[Decision]
    on_decision: Optional[Callable[[Decision], None]]
    raise_on_circuit_break: bool
    emit_tokens: bool
    _turn_started: bool
    _tool_runs: Dict[str, Tuple[str, int]]

    def _init_logic(
        self,
        regulator: Regulator,
        *,
        on_decision: Optional[Callable[[Decision], None]],
        raise_on_circuit_break: bool,
        emit_tokens: bool,
    ) -> None:
        """Mutable: seed per-handler state. Called from each concrete
        handler's ``__init__`` after ``super().__init__()``. Separate
        from ``__init__`` so the mixin stays free of LangChain init
        side-effects.
        """
        self.regulator = regulator
        self.on_decision = on_decision
        self.raise_on_circuit_break = raise_on_circuit_break
        self.emit_tokens = emit_tokens
        self.last_decision = None
        self._turn_started = False
        self._tool_runs = {}

    # ── Turn-boundary helpers ───────────────────────────────────────────

    def _handle_chain_start(self, inputs: Any, parent_run_id: Optional[UUID]) -> None:
        if parent_run_id is None and not self._turn_started:
            user_msg = extract_user_message(inputs)
            if user_msg:
                self.regulator.on_event(LLMEvent.turn_start(user_msg))
                self._turn_started = True

    def _handle_chain_end_or_error(self, parent_run_id: Optional[UUID]) -> None:
        if parent_run_id is None:
            self._turn_started = False

    def _handle_llm_start(self, prompts: List[str]) -> None:
        if not self._turn_started:
            user_msg = extract_prompts_as_message(prompts)
            if user_msg:
                self.regulator.on_event(LLMEvent.turn_start(user_msg))
                self._turn_started = True

    def _handle_chat_model_start(self, messages: List[List[Any]]) -> None:
        if not self._turn_started:
            user_msg = extract_chat_messages(messages)
            if user_msg:
                self.regulator.on_event(LLMEvent.turn_start(user_msg))
                self._turn_started = True

    def _handle_llm_new_token(self, token: str) -> None:
        if self.emit_tokens and token:
            # LC doesn't surface per-token logprobs; the 0.0 sentinel
            # trips the regulator's structural-confidence fallback.
            self.regulator.on_event(LLMEvent.token(token, 0.0, 0))

    def _handle_llm_end(self, response: Any) -> None:
        full_text = extract_response_text(response)
        self.regulator.on_event(LLMEvent.turn_complete(full_text))
        tokens_in, tokens_out = extract_token_usage(response)
        self.regulator.on_event(
            LLMEvent.cost(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                wallclock_ms=0,
            )
        )
        self._update_decision()

    def _handle_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: Any,
        run_id: UUID,
    ) -> None:
        name = tool_name_from_serialized(serialized)
        self._tool_runs[str(run_id)] = (name, time.monotonic_ns())
        args_json = input_str if isinstance(input_str, str) else None
        self.regulator.on_event(LLMEvent.tool_call(name, args_json))

    def _handle_tool_end(self, run_id: UUID) -> None:
        name, duration_ms = self._pop_tool_run(run_id)
        self.regulator.on_event(
            LLMEvent.tool_result(
                tool_name=name,
                success=True,
                duration_ms=duration_ms,
            )
        )
        self._update_decision()

    def _handle_tool_error(self, error: BaseException, run_id: UUID) -> None:
        name, duration_ms = self._pop_tool_run(run_id)
        self.regulator.on_event(
            LLMEvent.tool_result(
                tool_name=name,
                success=False,
                duration_ms=duration_ms,
                error_summary=str(error)[:200],
            )
        )
        self._update_decision()

    # ── Internals ───────────────────────────────────────────────────────

    def _pop_tool_run(self, run_id: UUID) -> Tuple[str, int]:
        """Return ``(tool_name, duration_ms)`` for a finished tool run.

        Defensive: returns ``("unknown", 0)`` when ``on_tool_end`` fires
        without a matching ``on_tool_start``. This shape still lets the
        regulator count the call and feeds a zero-duration sample into
        ``tool_total_duration_ms``; never raises.
        """
        name, start_ns = self._tool_runs.pop(str(run_id), ("unknown", 0))
        if start_ns:
            duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        else:
            duration_ms = 0
        return name, int(duration_ms)

    def _update_decision(self) -> None:
        """Mutable: refresh ``last_decision`` and dispatch hooks.

        Reason: every decision-changing event (turn_complete + cost,
        tool_result) must refresh the snapshot so callers polling
        ``last_decision`` see the freshest answer and so any
        ``on_decision`` or ``raise_on_circuit_break`` hook fires as
        early as the signal allows.
        """
        decision = self.regulator.decide()
        self.last_decision = decision
        if self.on_decision is not None:
            self.on_decision(decision)
        if self.raise_on_circuit_break and decision.is_circuit_break():
            raise CircuitBreakError(decision)


class NoosCallbackHandler(_BaseHandlerLogic, BaseCallbackHandler):
    """Synchronous LangChain callback handler.

    Parameters
    ----------
    regulator:
        The Noos regulator instance to feed. Caller owns construction
        and persistence (``export_json`` / ``from_json``).
    on_decision:
        Optional callback invoked with the fresh :class:`Decision`
        whenever the handler refreshes ``last_decision``. Fires even on
        ``continue`` so subscribers see the full signal stream.
    raise_on_circuit_break:
        When ``True``, any ``circuit_break`` decision raises
        :class:`CircuitBreakError` inside the callback, aborting the
        LangChain run. Default ``False``.
    emit_tokens:
        When ``True``, ``on_llm_new_token`` calls
        ``regulator.on_event(LLMEvent.token(...))`` per token. Adds
        Python-side overhead proportional to stream length; off by
        default.

    Notes
    -----
    ``raise_error = True`` is set at the class level so
    :class:`CircuitBreakError` (and any real bug inside the handler)
    propagates through LangChain's callback manager instead of being
    silently logged. ``run_inline = True`` keeps the handler on the
    same thread as the agent for stable UUID-based tool tracking.
    """

    raise_error: bool = True
    run_inline: bool = True

    def __init__(
        self,
        regulator: Regulator,
        *,
        on_decision: Optional[Callable[[Decision], None]] = None,
        raise_on_circuit_break: bool = False,
        emit_tokens: bool = False,
    ) -> None:
        super().__init__()
        self._init_logic(
            regulator,
            on_decision=on_decision,
            raise_on_circuit_break=raise_on_circuit_break,
            emit_tokens=emit_tokens,
        )

    # ── Turn boundary ───────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_start(inputs, parent_run_id)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end_or_error(parent_run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end_or_error(parent_run_id)

    # ── LLM lifecycle ───────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_start(prompts)

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chat_model_start(messages)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_new_token(token)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_end(response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # No regulator event maps cleanly to a provider-level failure.
        # LangChain propagates the error; the regulator's view stays
        # consistent with the last turn_start (no turn_complete fires).
        return

    # ── Tool lifecycle ──────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_start(serialized, input_str, run_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_end(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_error(error, run_id)


class AsyncNoosCallbackHandler(_BaseHandlerLogic, AsyncCallbackHandler):
    """Async variant of :class:`NoosCallbackHandler`.

    Identical semantics; use with ``ainvoke`` / ``astream`` / LangGraph
    async flows. The ``async def`` hooks are thin wrappers around the
    shared ``_handle_*`` helpers — none of the regulator operations is
    itself awaitable, so the body never needs ``await``.

    Pass a fresh instance per concurrent call:
    :class:`noos.Regulator` is single-threaded and must not be shared
    across concurrent agent runs.
    """

    raise_error: bool = True
    run_inline: bool = True

    def __init__(
        self,
        regulator: Regulator,
        *,
        on_decision: Optional[Callable[[Decision], None]] = None,
        raise_on_circuit_break: bool = False,
        emit_tokens: bool = False,
    ) -> None:
        super().__init__()
        self._init_logic(
            regulator,
            on_decision=on_decision,
            raise_on_circuit_break=raise_on_circuit_break,
            emit_tokens=emit_tokens,
        )

    # ── Turn boundary ───────────────────────────────────────────────────

    async def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_start(inputs, parent_run_id)

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end_or_error(parent_run_id)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chain_end_or_error(parent_run_id)

    # ── LLM lifecycle ───────────────────────────────────────────────────

    async def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_start(prompts)

    async def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_chat_model_start(messages)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_new_token(token)

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_llm_end(response)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        return

    # ── Tool lifecycle ──────────────────────────────────────────────────

    async def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_start(serialized, input_str, run_id)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_end(run_id)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._handle_tool_error(error, run_id)
