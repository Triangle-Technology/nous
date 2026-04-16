//! # Regulator demo 4 — Prompt intervention from learned corrections
//!
//! The bridge between `Decision::ProceduralWarning` and actually *using*
//! that warning. Demo 3 showed a pattern can be persisted across
//! sessions; this demo shows how the app turns that pattern into text
//! spliced into the next prompt, so the LLM sees the user's prior
//! corrections **before** generating the next response.
//!
//! The core new primitive is
//! [`CorrectionPattern::as_prompt_addendum`](nous::CorrectionPattern::as_prompt_addendum):
//! given a learned pattern, return a concise text block the app can
//! prepend to a system prompt, prefix a user message, or wrap in its own
//! framing.
//!
//! ## Scenario
//!
//! Same four user messages / three corrections as demo 3 — all hash
//! (empirically) to the same cluster `async+auth`. Demo structure:
//!
//! 1. **Phase 1** — same three learning turns (condensed; see demo 3 for
//!    the detailed play-by-play).
//! 2. **Phase 2** — next-session `TurnStart`. `decide()` returns
//!    `ProceduralWarning { patterns }`. Show the addendum text.
//! 3. **Phase 3 (BEFORE)** — call the LLM with the bare user message. In
//!    canned mode: a response that ignores past corrections.
//! 4. **Phase 4 (AFTER)** — call the LLM with the addendum spliced in.
//!    In canned mode: a response that respects the pattern.
//!
//! Live modes (Ollama / Anthropic) make two real LLM calls with the same
//! system parameters but different user-message bodies, so you can
//! directly compare what the model does with vs without the learned
//! context.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example regulator_prompt_intervention_demo                  # canned
//! cargo run --example regulator_prompt_intervention_demo -- ollama        # live
//! cargo run --example regulator_prompt_intervention_demo -- anthropic     # live
//! ```
//!
//! ## What this adds over demo 3
//!
//! Demo 3's take-away stopped at "ProceduralWarning fires with patterns
//! attached." The integrator question it left open: *"now what?"* This
//! demo answers the now-what with a single method call and one
//! `format!` splice. The intervention surface is that small by design —
//! it stays P9b-compliant (no English regex, no rule translation) and
//! keeps the LLM as the rule interpreter.

use std::env;

use nous::{Decision, LLMEvent, Regulator, RegulatorState};

#[path = "regulator_common/mod.rs"]
mod regulator_common;
use regulator_common::{call_anthropic, call_ollama};

const USER_ID: &str = "user_123";

/// Four messages that all hash to cluster `async+auth` (same as demo 3,
/// verified empirically before writing). Index 0..=2 are learning turns;
/// index 3 is the post-restart turn where intervention kicks in.
const USER_MESSAGES: &[&str] = &[
    "Make my auth module async",
    "Refactor auth to support async",
    "Change my auth function to async",
    "Add async handling to my auth",
];

/// Canned LLM responses during Phase 1 learning. Each includes a
/// docstring the user will correct.
const LEARNING_RESPONSES: &[&str] = &[
    "async fn auth() { /* ... */ }\n/// This function authenticates requests.",
    "pub async fn auth() -> Result<User, AuthError> { /* ... */ }\n/// Authenticates the incoming user.",
    "async fn auth_validate() { /* ... */ }\n/// Params: token (&str). Returns Session.",
];

/// Three corrections the user sends across Phase 1. All say "no
/// docstrings" in different words — Nous extracts a STRUCTURAL pattern
/// without parsing the English.
const CORRECTIONS: &[&str] = &[
    "Don't add docstrings to the refactor output please",
    "Skip the doc comments — just show the code",
    "No new docstrings. I want minimal diffs this time",
];

/// Canned Phase 3 response (without addendum) — ignores the pattern,
/// adds a docstring the user has repeatedly said they don't want.
const CANNED_RESPONSE_WITHOUT_ADDENDUM: &str =
    "pub async fn auth(token: &str) -> Result<User, AuthError> { /* ... */ }\n\
     /// Handles async authentication for the incoming token.";

/// Canned Phase 4 response (with addendum) — respects the pattern,
/// code only, no new docstring.
const CANNED_RESPONSE_WITH_ADDENDUM: &str =
    "pub async fn auth(token: &str) -> Result<User, AuthError> { /* ... */ }";

/// Conservative per-turn cost numbers so the cost-cap predicate never
/// fires during the learning phase — this demo is about intervention,
/// not circuit breaks.
const PER_TURN_TOKENS_IN: u32 = 25;
const PER_TURN_TOKENS_OUT: u32 = 120;
const PER_TURN_WALLCLOCK_MS: u32 = 900;

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "canned".into());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Regulator demo 4 — Prompt intervention from learned           ║");
    println!("║  corrections (the bridge from ProceduralWarning to action)     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Mode: {mode}");
    println!("User: {USER_ID}\n");

    // ── Phase 1: Learn the pattern ───────────────────────────────────
    println!("══════ Phase 1: Learning ({} corrections) ══════\n", CORRECTIONS.len());

    let mut regulator = Regulator::for_user(USER_ID);

    for i in 0..CORRECTIONS.len() {
        let user_msg = USER_MESSAGES[i];
        let response = LEARNING_RESPONSES[i];

        regulator.on_event(LLMEvent::TurnStart {
            user_message: user_msg.into(),
        });
        regulator.on_event(LLMEvent::TurnComplete {
            full_response: response.into(),
        });
        regulator.on_event(LLMEvent::Cost {
            tokens_in: PER_TURN_TOKENS_IN,
            tokens_out: PER_TURN_TOKENS_OUT,
            wallclock_ms: PER_TURN_WALLCLOCK_MS,
            provider: Some("learning".into()),
        });
        regulator.on_event(LLMEvent::UserCorrection {
            correction_message: CORRECTIONS[i].into(),
            corrects_last: true,
        });
        println!(
            "Turn {} · correction recorded: {}",
            i + 1,
            CORRECTIONS[i]
        );
    }
    println!("\n3/3 corrections recorded on cluster `async+auth`.\n");

    // Persist + restore to prove the addendum survives restart.
    let state: RegulatorState = regulator.export();
    let json = match serde_json::to_string(&state) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not serialise RegulatorState: {e}");
            std::process::exit(1);
        }
    };
    drop(regulator);
    let restored: RegulatorState = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not deserialise RegulatorState: {e}");
            std::process::exit(1);
        }
    };
    let mut regulator = Regulator::import(restored);
    println!("[Process restart simulated via {}-byte JSON roundtrip.]\n", json.len());

    // ── Phase 2: Next session — ProceduralWarning + extract addendum ─
    println!("══════ Phase 2: Next session — extract addendum ══════\n");

    let next_msg = USER_MESSAGES[3];
    println!("User (new session): {next_msg}\n");

    // `decide()` probe must happen AFTER TurnStart but BEFORE
    // TurnComplete — scope tracker has task keywords only at this point,
    // so `drift_score` returns None and `ScopeDriftWarn` skips;
    // `ProceduralWarning` can surface. See regulator-guide §6.2.
    regulator.on_event(LLMEvent::TurnStart {
        user_message: next_msg.into(),
    });

    let addendum = match regulator.decide() {
        Decision::ProceduralWarning { patterns } => {
            println!("Regulator (pre-generation): ProceduralWarning fired.");
            println!("patterns.len() = {}", patterns.len());
            // Compose addenda for all patterns that apply. In this demo
            // there's exactly one; the `join("\n")` shape is future-proof
            // against multi-cluster agents.
            patterns
                .iter()
                .map(|p| p.as_prompt_addendum())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        }
        other => {
            eprintln!(
                "(Unexpected) decide() returned {other:?} — check cluster stability."
            );
            std::process::exit(1);
        }
    };

    println!("\n── Addendum text (from `CorrectionPattern::as_prompt_addendum`) ──");
    print_indent(&addendum, "  ");

    // ── Phase 3: BEFORE — bare prompt ────────────────────────────────
    println!("\n══════ Phase 3: BEFORE intervention — bare user message ══════\n");
    println!("Prompt sent to LLM:");
    println!("  {next_msg}\n");

    let before_response = match mode.as_str() {
        "canned" => CANNED_RESPONSE_WITHOUT_ADDENDUM.to_string(),
        "ollama" => match call_ollama(next_msg) {
            Ok((r, _, _, _)) => r,
            Err(e) => {
                eprintln!("⚠  Ollama failed: {e}; using canned fallback\n");
                CANNED_RESPONSE_WITHOUT_ADDENDUM.to_string()
            }
        },
        "anthropic" => match call_anthropic(next_msg) {
            Ok((r, _, _, _)) => r,
            Err(e) => {
                eprintln!("⚠  Anthropic failed: {e}; using canned fallback\n");
                CANNED_RESPONSE_WITHOUT_ADDENDUM.to_string()
            }
        },
        other => {
            eprintln!("Unknown mode {other:?}. Use canned|ollama|anthropic.");
            std::process::exit(2);
        }
    };
    println!("LLM output:");
    print_indent(&before_response, "  ");

    // ── Phase 4: AFTER — splice addendum into user message ───────────
    println!("\n══════ Phase 4: AFTER intervention — prompt includes addendum ══════\n");
    let enhanced = format!("{addendum}\nCurrent request: {next_msg}");
    println!("Prompt sent to LLM:");
    print_indent(&enhanced, "  ");
    println!();

    let after_response = match mode.as_str() {
        "canned" => CANNED_RESPONSE_WITH_ADDENDUM.to_string(),
        "ollama" => match call_ollama(&enhanced) {
            Ok((r, _, _, _)) => r,
            Err(e) => {
                eprintln!("⚠  Ollama failed: {e}; using canned fallback\n");
                CANNED_RESPONSE_WITH_ADDENDUM.to_string()
            }
        },
        "anthropic" => match call_anthropic(&enhanced) {
            Ok((r, _, _, _)) => r,
            Err(e) => {
                eprintln!("⚠  Anthropic failed: {e}; using canned fallback\n");
                CANNED_RESPONSE_WITH_ADDENDUM.to_string()
            }
        },
        _ => unreachable!("mode validated above"),
    };
    println!("LLM output:");
    print_indent(&after_response, "  ");

    // ── Take-away ────────────────────────────────────────────────────
    println!("\n══════ Take-away ══════");
    println!();
    println!("The intervention is one method + one format! call:");
    println!("  let addendum = pattern.as_prompt_addendum();");
    println!("  let prompt = format!(\"{{addendum}}\\nCurrent request: {{user_msg}}\");");
    println!();
    println!("Nous does not rewrite the prompt, does not parse the corrections,");
    println!("does not call the LLM. It surfaces the structural pattern + raw");
    println!("example texts; the app composes them into whatever prompt shape");
    println!("its stack uses. The LLM does the rule interpretation — P9b-compliant");
    println!("by construction.");
    println!();
    println!("Canned mode shows the mechanism with a deterministic \"bad/good\" split.");
    println!("Live modes (ollama / anthropic) show what a real model does when the");
    println!("same question arrives with vs without the pattern in context.");
}

/// Print `text` with each line indented by `indent`. Used for all
/// multi-line payload displays in the demo so the log stays readable.
fn print_indent(text: &str, indent: &str) {
    for line in text.lines() {
        println!("{indent}{line}");
    }
    if text.is_empty() {
        println!("{indent}(empty)");
    }
}
