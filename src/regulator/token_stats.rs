//! Token statistics accumulator — rolling logprob window over an LLM's
//! token stream with an entropy-based confidence readout.
//!
//! **Scope note (P1 / P9b)**: this is an I/O sub-module of
//! [`regulator`](super), not a cognitive module. All computations here
//! operate on numeric tokens + logprobs (primary path) or on
//! language-neutral structural signals (length, question-mark density —
//! fallback path). No sentiment lexicons, no topic regex. P1 applies to
//! the wrapped [`CognitiveSession`](crate::session::CognitiveSession);
//! here we satisfy P9b by construction.
//!
//! ## R2 decision (Session 17)
//!
//! Per-token logprobs are not universally available:
//!
//! - **Available**: OpenAI (`logprobs: true`), vLLM, local candle
//!   inference, most open-model runtimes.
//! - **Not available by default**: Anthropic API (as of 2026-04).
//!
//! This module follows a hybrid strategy (recommended in the Path 2
//! architecture plan R2 row):
//!
//! 1. **Primary path** — when the provider exposes logprobs, compute
//!    confidence from mean negative-log-likelihood over a rolling
//!    window. This is the high-fidelity signal that detects gibberish
//!    (uniformly high NLL on OOD text) and local uncertainty runs.
//! 2. **Fallback path** — when logprobs are absent (callers pass
//!    `LOGPROB_UNAVAILABLE` = `0.0` per the `LLMEvent::Token` contract),
//!    fall back to language-neutral structural heuristics on the final
//!    response text: very short replies or high question-mark density
//!    cap confidence below neutral.
//!
//! The fallback is deliberately conservative: it cannot distinguish a
//! confident short answer from a refusal, so it errs low. Callers that
//! need stronger calibration should prefer a provider that exposes
//! logprobs.
//!
//! ## Scope for Session 17
//!
//! This module lands the scalar confidence readout. Per-fragment
//! `ConfidenceSpan` analysis (from [`Decision::LowConfidenceSpans`]) is
//! Session 18+ work — it needs span metadata this MVP doesn't track.
//!
//! ## Gating (P10)
//!
//! This module does NOT produce a [`Decision`] variant in 0.3.0. The
//! scalar it exposes via [`Regulator::confidence`] is an observability
//! readout — callers can use it to build their own heuristics or to
//! feed external logging.
//!
//! - **Suppresses**: nothing (pure observability).
//! - **Suppressed by**: nothing.
//! - **Inactive when**: no tokens have been observed this turn AND no
//!   [`TurnComplete`](super::LLMEvent::TurnComplete) has landed yet.
//!   In that state [`Regulator::confidence`] returns
//!   [`NEUTRAL_CONFIDENCE`] (0.5) rather than claiming false certainty.
//!
//! The reserved [`Decision::LowConfidenceSpans`] variant will consume
//! the same rolling-window data once span-level tracking lands; its
//! own gating section will be added with that change.
//!
//! [`CognitiveSession`]: crate::session::CognitiveSession
//! [`Decision`]: super::Decision
//! [`Decision::LowConfidenceSpans`]: super::Decision::LowConfidenceSpans
//! [`Regulator::confidence`]: super::Regulator::confidence

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────────

/// Default rolling window size for per-token logprobs.
///
/// 128 tokens ≈ one or two short paragraphs at typical tokenisation rates.
/// Large enough to smooth per-token noise, small enough that a local
/// uncertainty run (a confused span in the middle of a response) is not
/// drowned out by surrounding confident tokens. Derivation: at an average
/// of ~1 token / 4 chars and ~600-char confident chunks surrounding a
/// 200-char confused span, 128 tokens captures the confused span
/// proportionally.
pub const DEFAULT_WINDOW_SIZE: usize = 128;

/// Sentinel value for "logprob unavailable", per the
/// [`LLMEvent::Token`](super::LLMEvent::Token) contract documented in
/// `docs/regulator-design.md` §3.2.
///
/// Real natural-log probabilities are always ≤ 0 (ln of a value in
/// `(0, 1]` is non-positive); a logprob of exactly `0.0` would mean
/// p = 1.0 which effectively never happens on LLM output. Using `0.0`
/// as the sentinel avoids a type-level `Option<f64>` while keeping the
/// collision risk negligible in practice. Spurious positive or non-finite
/// values are also treated as "unavailable" (fail-open per P5).
pub const LOGPROB_UNAVAILABLE: f64 = 0.0;

/// Mean negative-log-likelihood (nats) that maps to confidence = 0.5.
///
/// Derivation: modern LLMs on in-distribution English text show mean
/// per-token NLL around 2.5–3.5 nats (perplexity 12–33). OOD / gibberish
/// text pushes NLL to 5–7 nats (perplexity 150–1100). 4.0 sits between
/// the two regimes — confident output pulls above 0.5, uncertain /
/// gibberish pulls below 0.5. Calibration is linear in MVP; a logistic
/// fit is tracked for Session 24 real-LLM eval.
pub const MEAN_NLL_CONFIDENCE_MIDPOINT: f64 = 4.0;

/// Half-range of the linear NLL→confidence mapping (nats).
///
/// With midpoint 4.0 and half-range 3.0, NLL = 1.0 (very confident) maps
/// to confidence 1.0, NLL = 7.0 (very uncertain / gibberish) maps to
/// confidence 0.0. Symmetric around the midpoint.
pub const MEAN_NLL_CONFIDENCE_HALF_RANGE: f64 = 3.0;

/// Structural fallback: minimum response length in characters below
/// which confidence is capped low.
///
/// Very short replies are disproportionately likely to be refusals
/// ("I can't help with that") or failed attempts ("Sorry, I don't know")
/// rather than decisive short answers. The cap is conservative by design.
pub const STRUCT_MIN_LENGTH_CHARS: usize = 40;

/// Structural fallback: question-mark-to-character ratio above which the
/// response is treated as clarification-seeking rather than answer-giving.
///
/// 0.02 = 2 `?` per 100 chars. A normal declarative response is 0%. A
/// response asking back ("Which file did you mean? And which function?")
/// typically exceeds 5% on any non-trivial clarification.
pub const STRUCT_HIGH_QUESTION_RATIO: f64 = 0.02;

/// Structural fallback: confidence returned when the response text looks
/// unremarkable (not too short, not question-heavy).
///
/// Set at 0.7 rather than neutral 0.5 because absence of red flags is
/// mildly positive evidence — but we cap below 1.0 to reflect that the
/// fallback path lacks the signal a logprob window provides.
pub const STRUCT_FALLBACK_DEFAULT: f64 = 0.7;

/// Structural fallback: confidence returned when one red flag is present
/// (short response OR question-heavy, but not both).
///
/// Set at 0.4 — below `NEUTRAL_CONFIDENCE` (0.5) so callers that use
/// `confidence < 0.5` as an abstention trigger fire, but not so far
/// below that it overrides the "clearly uncertain" band reserved for
/// both-flags. The 0.25 spacing from `NEUTRAL_CONFIDENCE` gives each
/// tri-bucket band a clearly separated regime label.
pub const STRUCT_FALLBACK_WEAK: f64 = 0.4;

/// Structural fallback: confidence returned when multiple red flags are
/// present (short AND question-heavy).
///
/// Set at 0.2 — 0.2 below `STRUCT_FALLBACK_WEAK` (0.4) to preserve
/// the 0.2-band spacing between regimes. Low enough that `confidence
/// < 0.3` thresholds (used by stricter abstention callers) fire
/// reliably, while staying non-zero so downstream code doesn't treat
/// the signal as "missing" when it's really "clearly uncertain".
pub const STRUCT_FALLBACK_STRONG: f64 = 0.2;

/// Confidence returned when no signal is available (empty accumulator,
/// no response text). Matches the legacy `CognitiveSignals::confidence`
/// base so Path 1 ↔ Path 2 users see the same default.
pub const NEUTRAL_CONFIDENCE: f64 = 0.5;

// ── Accumulator ────────────────────────────────────────────────────────

/// Rolling statistics over the current turn's LLM output.
///
/// Produces a turn-level confidence estimate from per-token logprobs via
/// [`logprob_confidence`](Self::logprob_confidence). The accumulator is
/// per-turn: callers drive lifecycle via
/// [`begin_turn`](Self::begin_turn) / [`on_token`](Self::on_token) and
/// read state via [`has_logprobs`](Self::has_logprobs) +
/// [`logprob_confidence`](Self::logprob_confidence).
///
/// Structural fallback is a free function ([`structural_confidence`])
/// rather than a method — it needs the response text, which the
/// accumulator doesn't store (the wrapping [`Regulator`](super::Regulator)
/// already buffers it in `pending_response`). Use
/// [`confidence_with_fallback`] to compose both paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStatsAccumulator {
    /// Rolling window of recent token logprobs. Only finite, non-sentinel
    /// logprobs enter the window. Oldest values are evicted when the
    /// window fills past `window_size`.
    logprobs: VecDeque<f64>,
    /// Count of tokens seen in the current turn, including those whose
    /// logprob was unavailable.
    total_tokens: usize,
    /// Count of tokens in the current turn that arrived with
    /// `LOGPROB_UNAVAILABLE`. Used by [`logprob_coverage`](Self::logprob_coverage)
    /// to surface how much of the turn is logprob-backed.
    unavailable_count: usize,
    /// Maximum window size. Constructor clamps to >= 1.
    window_size: usize,
}

impl TokenStatsAccumulator {
    /// Construct with [`DEFAULT_WINDOW_SIZE`].
    pub fn new() -> Self {
        Self::with_window(DEFAULT_WINDOW_SIZE)
    }

    /// Construct with a custom window size (clamped to ≥ 1 for
    /// fail-open P5 behaviour).
    pub fn with_window(window_size: usize) -> Self {
        Self {
            logprobs: VecDeque::new(),
            total_tokens: 0,
            unavailable_count: 0,
            window_size: window_size.max(1),
        }
    }

    /// Mutable: reset all per-turn state. Called at each `TurnStart`
    /// boundary. Requires mutation because the accumulator is a rolling
    /// statistic that must not leak across turn boundaries.
    pub fn begin_turn(&mut self) {
        self.logprobs.clear();
        self.total_tokens = 0;
        self.unavailable_count = 0;
    }

    /// Mutable: record one token from the current turn. Requires
    /// mutation because every token updates the rolling window and
    /// coverage counters.
    ///
    /// Logprobs ≥ 0.0 or non-finite are treated as "unavailable" and
    /// bypass the window (fail-open per P5): real natural-log
    /// probabilities are strictly < 0, and the
    /// [`LOGPROB_UNAVAILABLE`] sentinel is exactly `0.0`.
    pub fn on_token(&mut self, logprob: f64) {
        self.total_tokens += 1;
        if logprob >= 0.0 || !logprob.is_finite() {
            self.unavailable_count += 1;
            return;
        }
        self.logprobs.push_back(logprob);
        if self.logprobs.len() > self.window_size {
            self.logprobs.pop_front();
        }
    }

    /// Whether the rolling window contains any logprobs (primary
    /// confidence path is available).
    pub fn has_logprobs(&self) -> bool {
        !self.logprobs.is_empty()
    }

    /// Confidence from the rolling logprob window, in `[0, 1]`.
    ///
    /// Returns [`NEUTRAL_CONFIDENCE`] when the window is empty. Otherwise
    /// maps mean negative-log-likelihood linearly through
    /// `[MIDPOINT - HALF_RANGE, MIDPOINT + HALF_RANGE]` → `[1, 0]` and
    /// clamps to `[0, 1]`.
    pub fn logprob_confidence(&self) -> f64 {
        if self.logprobs.is_empty() {
            return NEUTRAL_CONFIDENCE;
        }
        let mean_nll: f64 =
            self.logprobs.iter().map(|lp| -lp).sum::<f64>() / self.logprobs.len() as f64;
        let offset = MEAN_NLL_CONFIDENCE_MIDPOINT - mean_nll;
        let confidence = 0.5 + 0.5 * (offset / MEAN_NLL_CONFIDENCE_HALF_RANGE);
        confidence.clamp(0.0, 1.0)
    }

    /// Total tokens observed in the current turn (including unavailable
    /// logprobs).
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }

    /// Fraction of tokens in the current turn whose logprobs were
    /// available, in `[0, 1]`. 0.0 when no tokens have been observed.
    ///
    /// Callers can use this to decide between the primary and fallback
    /// paths: coverage near 1.0 means the logprob confidence is
    /// well-supported; coverage near 0.0 means the structural fallback
    /// should be preferred.
    pub fn logprob_coverage(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        let available = self.total_tokens - self.unavailable_count;
        available as f64 / self.total_tokens as f64
    }
}

impl Default for TokenStatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Structural fallback ────────────────────────────────────────────────

/// Language-neutral structural confidence heuristic.
///
/// Used when the provider does not expose per-token logprobs. Signals:
///
/// - **Short response** (under [`STRUCT_MIN_LENGTH_CHARS`] chars): red
///   flag — disproportionately indicates refusal or failed attempt.
/// - **High question-mark density** (ratio ≥
///   [`STRUCT_HIGH_QUESTION_RATIO`]): red flag — response is
///   clarification-seeking rather than answering.
///
/// Both flags → [`STRUCT_FALLBACK_STRONG`] (0.2). One flag →
/// [`STRUCT_FALLBACK_WEAK`] (0.4). Neither flag →
/// [`STRUCT_FALLBACK_DEFAULT`] (0.7). Empty string →
/// [`NEUTRAL_CONFIDENCE`] (caller has no signal to work with).
///
/// This is the P9b-compliant fallback: no sentiment lexicon, no language
/// assumption. Works on any language the LLM produces — but is less
/// discriminating than logprob-based confidence.
pub fn structural_confidence(response_text: &str) -> f64 {
    let len = response_text.chars().count();
    if len == 0 {
        return NEUTRAL_CONFIDENCE;
    }
    let q_count = response_text.chars().filter(|c| *c == '?').count();
    let q_ratio = q_count as f64 / len as f64;

    let short = len < STRUCT_MIN_LENGTH_CHARS;
    let question_heavy = q_ratio >= STRUCT_HIGH_QUESTION_RATIO;

    match (short, question_heavy) {
        (true, true) => STRUCT_FALLBACK_STRONG,
        (true, false) | (false, true) => STRUCT_FALLBACK_WEAK,
        (false, false) => STRUCT_FALLBACK_DEFAULT,
    }
}

/// Compose the primary logprob-based path with the structural fallback.
///
/// - `stats` has any logprobs → use [`logprob_confidence`](TokenStatsAccumulator::logprob_confidence).
/// - Otherwise, `response_text` is `Some(text)` → use
///   [`structural_confidence`].
/// - Otherwise → [`NEUTRAL_CONFIDENCE`].
///
/// This is the function [`Regulator::confidence`](super::Regulator::confidence)
/// delegates to; it's exposed so external callers that assemble their own
/// regulation pipelines (bypassing `Regulator`) can reuse the same
/// composition rule.
pub fn confidence_with_fallback(
    stats: &TokenStatsAccumulator,
    response_text: Option<&str>,
) -> f64 {
    if stats.has_logprobs() {
        return stats.logprob_confidence();
    }
    if let Some(text) = response_text {
        return structural_confidence(text);
    }
    NEUTRAL_CONFIDENCE
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TokenStatsAccumulator ──────────────────────────────────────────

    #[test]
    fn empty_accumulator_is_neutral() {
        let stats = TokenStatsAccumulator::new();
        assert!(!stats.has_logprobs());
        assert_eq!(stats.token_count(), 0);
        assert_eq!(stats.logprob_coverage(), 0.0);
        assert!(
            (stats.logprob_confidence() - NEUTRAL_CONFIDENCE).abs() < 1e-9,
            "empty window must return neutral confidence"
        );
    }

    #[test]
    fn high_logprob_tokens_raise_confidence() {
        // -0.1 nats per token → mean NLL 0.1 → well below midpoint 4.0
        // → confidence pinned at 1.0 by the mapping.
        let mut stats = TokenStatsAccumulator::new();
        for _ in 0..20 {
            stats.on_token(-0.1);
        }
        let c = stats.logprob_confidence();
        assert!(c > 0.9, "high-probability tokens should push confidence >0.9 (got {c})");
    }

    #[test]
    fn low_logprob_tokens_lower_confidence() {
        // -7.0 nats per token → mean NLL 7.0 → at MIDPOINT + HALF_RANGE
        // → confidence at 0.0 after clamp.
        let mut stats = TokenStatsAccumulator::new();
        for _ in 0..20 {
            stats.on_token(-7.0);
        }
        let c = stats.logprob_confidence();
        assert!(
            c < 0.1,
            "high-NLL (gibberish-like) tokens should pull confidence <0.1 (got {c})"
        );
    }

    #[test]
    fn mid_range_logprobs_produce_mid_confidence() {
        // mean NLL 4.0 (midpoint) → confidence exactly 0.5.
        let mut stats = TokenStatsAccumulator::new();
        for _ in 0..10 {
            stats.on_token(-4.0);
        }
        let c = stats.logprob_confidence();
        assert!(
            (c - 0.5).abs() < 1e-9,
            "mean NLL at midpoint must map to confidence 0.5 (got {c})"
        );
    }

    #[test]
    fn begin_turn_resets_accumulator() {
        let mut stats = TokenStatsAccumulator::new();
        stats.on_token(-1.0);
        stats.on_token(-2.0);
        assert!(stats.has_logprobs());

        stats.begin_turn();
        assert!(!stats.has_logprobs());
        assert_eq!(stats.token_count(), 0);
        assert_eq!(stats.logprob_coverage(), 0.0);
    }

    #[test]
    fn rolling_window_evicts_oldest() {
        // Contract: once the window fills, older logprobs are evicted so
        // confidence reflects only the most recent `window_size` tokens.
        // We push 6 tokens into a window of 4; if eviction works, the
        // surviving window is [-3, -4, -5, -6] (mean NLL = 4.5), which
        // maps to confidence = 0.5 + 0.5 * (4.0 - 4.5) / 3.0 ≈ 0.4167.
        // If eviction were broken and all 6 survived, mean NLL = 3.5
        // would give confidence ≈ 0.5833 — observably different.
        let mut reference = TokenStatsAccumulator::with_window(4);
        reference.on_token(-3.0);
        reference.on_token(-4.0);
        reference.on_token(-5.0);
        reference.on_token(-6.0);
        let expected = reference.logprob_confidence();

        let mut stats = TokenStatsAccumulator::with_window(4);
        stats.on_token(-1.0); // evicted
        stats.on_token(-2.0); // evicted
        stats.on_token(-3.0);
        stats.on_token(-4.0);
        stats.on_token(-5.0);
        stats.on_token(-6.0);

        assert!(
            (stats.logprob_confidence() - expected).abs() < 1e-9,
            "eviction must match a reference built only from surviving tokens \
             (got {}, expected {})",
            stats.logprob_confidence(),
            expected
        );
        // `total_tokens` counts ALL tokens seen (observable via
        // `token_count`), not just the ones in the window.
        assert_eq!(stats.token_count(), 6);
    }

    #[test]
    fn unavailable_logprob_does_not_enter_window() {
        let mut stats = TokenStatsAccumulator::new();
        stats.on_token(LOGPROB_UNAVAILABLE);
        assert!(!stats.has_logprobs());
        assert_eq!(stats.token_count(), 1);
        assert_eq!(stats.logprob_coverage(), 0.0);
    }

    #[test]
    fn non_finite_or_positive_logprob_treated_as_unavailable() {
        // Fail-open: spurious positive values + NaN + Inf all bypass
        // the window rather than corrupt the mean.
        let mut stats = TokenStatsAccumulator::new();
        stats.on_token(1.5); // positive — not a real logprob
        stats.on_token(f64::NAN);
        stats.on_token(f64::INFINITY);
        stats.on_token(f64::NEG_INFINITY);
        assert!(!stats.has_logprobs());
        assert_eq!(stats.token_count(), 4);
        assert_eq!(stats.logprob_coverage(), 0.0);
    }

    #[test]
    fn logprob_coverage_tracks_available_fraction() {
        let mut stats = TokenStatsAccumulator::new();
        stats.on_token(-1.0);
        stats.on_token(LOGPROB_UNAVAILABLE);
        stats.on_token(-2.0);
        stats.on_token(LOGPROB_UNAVAILABLE);
        // 2 of 4 tokens have real logprobs.
        assert!((stats.logprob_coverage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn zero_window_size_clamped_to_one() {
        // Contract: `with_window(0)` must not divide-by-zero or panic;
        // fail-open clamps to 1. Observable: after two pushes, confidence
        // reflects only the most recent logprob (the one that survived
        // eviction in a size-1 window).
        let mut reference = TokenStatsAccumulator::with_window(1);
        reference.on_token(-2.0);
        let expected = reference.logprob_confidence();

        let mut stats = TokenStatsAccumulator::with_window(0);
        stats.on_token(-1.0);
        stats.on_token(-2.0);
        assert!(
            (stats.logprob_confidence() - expected).abs() < 1e-9,
            "size-0 window should clamp to 1 and evict oldest \
             (got {}, expected {})",
            stats.logprob_confidence(),
            expected
        );
    }

    // ── structural_confidence ──────────────────────────────────────────

    #[test]
    fn structural_empty_text_is_neutral() {
        assert!((structural_confidence("") - NEUTRAL_CONFIDENCE).abs() < 1e-9);
    }

    #[test]
    fn structural_short_response_is_low() {
        // Under STRUCT_MIN_LENGTH_CHARS (40), no questions.
        let c = structural_confidence("I don't know.");
        assert!(
            (c - STRUCT_FALLBACK_WEAK).abs() < 1e-9,
            "short response should return weak fallback (got {c})"
        );
    }

    #[test]
    fn structural_question_heavy_short_is_strongest_low() {
        // Short AND question-heavy — both red flags fire.
        let c = structural_confidence("What? How? When?");
        assert!(
            (c - STRUCT_FALLBACK_STRONG).abs() < 1e-9,
            "short + question-heavy should return strong-low fallback (got {c})"
        );
    }

    #[test]
    fn structural_question_heavy_long_is_weak() {
        // Long enough but lots of `?` — one red flag.
        let c = structural_confidence(
            "Which file did you mean? And which function inside it? \
             Also, should the refactor preserve the existing signature?",
        );
        assert!(
            (c - STRUCT_FALLBACK_WEAK).abs() < 1e-9,
            "question-heavy long response should return weak fallback (got {c})"
        );
    }

    #[test]
    fn structural_normal_response_is_default() {
        // Long enough, few/no questions — no red flags.
        let c = structural_confidence(
            "Here is the refactored function. It preserves the original \
             signature and moves the body into an async block returning \
             a Future. No behaviour changes for synchronous callers.",
        );
        assert!(
            (c - STRUCT_FALLBACK_DEFAULT).abs() < 1e-9,
            "unremarkable response should return default fallback (got {c})"
        );
    }

    // ── confidence_with_fallback ───────────────────────────────────────

    #[test]
    fn fallback_prefers_logprobs_when_available() {
        let mut stats = TokenStatsAccumulator::new();
        for _ in 0..10 {
            stats.on_token(-0.5); // very confident
        }
        // Even with deliberately unfavourable structural text, logprobs
        // win.
        let c = confidence_with_fallback(&stats, Some("???"));
        assert!(c > 0.8, "logprob path should override structural (got {c})");
    }

    #[test]
    fn fallback_uses_structural_when_no_logprobs() {
        let stats = TokenStatsAccumulator::new();
        let c = confidence_with_fallback(
            &stats,
            Some("Here is a clear answer with enough length to pass the minimum."),
        );
        assert!(
            (c - STRUCT_FALLBACK_DEFAULT).abs() < 1e-9,
            "empty-logprobs + clean text should use structural default (got {c})"
        );
    }

    #[test]
    fn fallback_neutral_when_no_signal() {
        let stats = TokenStatsAccumulator::new();
        let c = confidence_with_fallback(&stats, None);
        assert!((c - NEUTRAL_CONFIDENCE).abs() < 1e-9);
    }

    #[test]
    fn fallback_gibberish_path_yields_low_confidence() {
        // End-to-end: simulate a gibberish token stream (high NLL) and
        // confirm the Regulator-facing confidence drops into the low
        // band. This is the Session 17 test-target scenario
        // ("gibberish text produces low confidence") at the sub-module
        // level.
        let mut stats = TokenStatsAccumulator::new();
        for _ in 0..30 {
            stats.on_token(-6.5); // ~perplexity 670 — very uncertain
        }
        let c = confidence_with_fallback(&stats, None);
        assert!(
            c < 0.2,
            "gibberish-level mean NLL should land confidence in the low band (got {c})"
        );
    }
}
