//! # Regulator demo 4 — Tool-call loop intercept (Path A, 0.3.0)
//!
//! New in 0.3.0: the regulator observes tool calls via
//! [`LLMEvent::ToolCall`] / [`LLMEvent::ToolResult`] and fires
//! [`Decision::CircuitBreak`] with
//! [`CircuitBreakReason::RepeatedToolCallLoop`] when an agent calls the
//! same tool five or more consecutive times within a single turn
//! without interleaving a reasoning step or different tool.
//!
//! This catches a very common failure mode of autonomous agents: the
//! retry loop that keeps calling the same failing API / the same search
//! endpoint with nearly-identical queries, burning tokens without
//! making progress. Generic retry-backoff libraries (tenacity, backoff)
//! only look at transport errors — they do not flag a tool call that
//! succeeds at the protocol level but keeps returning the same unhelpful
//! result.
//!
//! ## Scenario
//!
//! An agent is asked to find a user's recent orders. It calls
//! `search_orders` repeatedly with small query tweaks, each call
//! returning an empty result. Without the regulator, the agent would
//! continue until it exhausts its cost budget or some other hard limit.
//! With the regulator, after the fifth consecutive `search_orders`
//! call the agent halts with an actionable suggestion.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example regulator_tool_loop_demo
//! ```
//!
//! ## What this closes that competitors cannot
//!
//! - **tenacity / backoff / retry crates**: react to transport errors
//!   only (4xx / 5xx / exception types). A tool call that returns
//!   `{"results": []}` looks successful to them.
//! - **Langfuse / Arize**: surface the same tool-call stream but only
//!   after the fact, as an observability dashboard. No in-loop halt.
//! - **LLM agent frameworks (LangChain, CrewAI, AutoGen)**: typically
//!   have a `max_iterations` knob that stops ALL iteration including
//!   productive ones. The regulator's signal is structural — it fires
//!   specifically on consecutive same-tool calls, not on overall turn
//!   depth.

use noos::regulator::tools::TOOL_LOOP_THRESHOLD;
use noos::regulator::{CircuitBreakReason, Decision, LLMEvent, Regulator};

fn main() {
    println!("══════ Regulator demo 4 — tool-call loop intercept ══════\n");
    println!("Agent task: find recent orders for user 7");
    println!("Agent strategy: call `search_orders` repeatedly with query tweaks");
    println!("Each call returns `{{\"results\": []}}` — protocol-level success, no progress.\n");

    let mut regulator = Regulator::for_user("user_7");
    regulator.on_event(LLMEvent::TurnStart {
        user_message: "find recent orders for user 7".into(),
    });

    // Sequence of `search_orders` calls the agent makes during a single
    // turn. Each call fires a ToolCall + ToolResult pair through the
    // regulator.
    let queries = [
        r#"{"user_id": 7, "days": 7}"#,
        r#"{"user_id": 7, "days": 14}"#,
        r#"{"user_id": 7, "days": 30}"#,
        r#"{"user_id": 7, "days": 60}"#,
        r#"{"user_id": 7, "days": 90}"#,
        r#"{"user_id": 7, "days": 180}"#,
    ];

    let mut halted = false;
    for (i, args) in queries.iter().enumerate() {
        regulator.on_event(LLMEvent::ToolCall {
            tool_name: "search_orders".into(),
            args_json: Some((*args).to_string()),
        });
        regulator.on_event(LLMEvent::ToolResult {
            tool_name: "search_orders".into(),
            success: true, // protocol-level success
            duration_ms: 95,
            error_summary: None,
        });

        // Render the per-call progress bar + probe the decision.
        println!(
            "── Call {call_num:>2} / threshold {threshold} ──",
            call_num = i + 1,
            threshold = TOOL_LOOP_THRESHOLD,
        );
        println!("  args: {args}");
        println!("  result: {{\"results\": []}}");

        match regulator.decide() {
            Decision::CircuitBreak {
                reason:
                    CircuitBreakReason::RepeatedToolCallLoop {
                        ref tool_name,
                        consecutive_count,
                    },
                ref suggestion,
            } => {
                println!("\n[Decision::CircuitBreak]  ←  HALT");
                println!("  reason:             RepeatedToolCallLoop");
                println!("  tool_name:          {tool_name}");
                println!("  consecutive_count:  {consecutive_count}");
                println!("  suggestion:         {suggestion}");
                println!(
                    "\nRegulator observability at halt:\n  \
                     tool_total_calls:        {}\n  \
                     tool_total_duration_ms:  {} ms\n  \
                     tool_failure_count:      {}",
                    regulator.tool_total_calls(),
                    regulator.tool_total_duration_ms(),
                    regulator.tool_failure_count(),
                );
                halted = true;
                break;
            }
            Decision::Continue => {
                println!("  decision: Continue  (agent keeps calling)");
            }
            other => {
                println!("  decision: {other:?}");
            }
        }
        println!();
    }

    println!();
    println!("══════ Take-away ══════");
    if halted {
        println!(
            "Regulator halted at call {TOOL_LOOP_THRESHOLD} — the signature of a \
             runaway loop on a tool that keeps returning unhelpful results."
        );
    } else {
        println!(
            "(The loop did not fire on this run — this should only happen if \
             TOOL_LOOP_THRESHOLD was modified.)"
        );
    }
    println!();
    println!("Baseline agent (no regulator):");
    println!("  • continues calling until total turn cost / max_iterations trips;");
    println!("  • wastes tool-latency budget on structurally identical queries;");
    println!("  • burns LLM tokens re-summarising the same empty response.");
    println!();
    println!("Retry / backoff crates (tenacity, backoff, tokio-retry):");
    println!("  • trip on transport errors only — a protocol-level success with");
    println!("    empty payload does not register as a retryable failure.");
    println!();
    println!("Agent frameworks (LangChain, CrewAI, AutoGen):");
    println!("  • use `max_iterations` / `max_steps` that bounds ALL iteration,");
    println!("    including productive interleaved tool calls — the regulator's");
    println!("    signal is structural (consecutive same-tool) so productive");
    println!("    agents are not penalised.");
}
