//! Cost accumulator — cumulative token / wallclock counters plus a
//! rolling quality history. Drives
//! [`Decision::CircuitBreak`](super::Decision::CircuitBreak) predicates
//! from [`Regulator::decide`](super::Regulator::decide).
//!
//! **Scope note (P1 / P9b)**: I/O adapter sub-module, not cognitive.
//! Everything here is arithmetic over integers and bounded floats —
//! token counts, milliseconds, `[0, 1]` quality scalars. No sentiment
//! lexicon, no topic inference. P1 applies to the wrapped
//! [`CognitiveSession`](crate::session::CognitiveSession); P9b is
//! satisfied by construction.
//!
//! ## R2-style design decision: `LLMEvent::Cost` → `[0, 1]` normalization
//!
//! Session 19 owes a mapping from concrete
//! [`LLMEvent::Cost`](super::LLMEvent::Cost)
//! `{ tokens_out, wallclock_ms }` onto the `[0, 1]` scale that
//! [`CognitiveSession::track_cost`](crate::session::CognitiveSession::track_cost)
//! expects. The chosen formula is
//!
//! ```text
//! normalized = TOKEN_COST_WEIGHT · clamp(tokens_out / TYPICAL_TURN_TOKENS_OUT, 0, 1)
//!            + (1 − TOKEN_COST_WEIGHT) · clamp(wallclock_ms / TYPICAL_TURN_WALLCLOCK_MS, 0, 1)
//! ```
//!
//! - **Tokens dominate (0.7 weight)** because they are the direct
//!   billing metric on every major provider.
//! - **Wallclock contributes (0.3 weight)** because infrastructure /
//!   SLA cost doesn't show up in billing but matters operationally.
//! - **Both components are clamped** so that single-turn outliers
//!   (a runaway 100k-token reply) cap the per-turn depletion at 1.0
//!   instead of exploding the scale.
//!
//! ## Gating (P10)
//!
//! This module produces the top-priority
//! [`Decision::CircuitBreak`](super::Decision::CircuitBreak) variants —
//! [`CircuitBreakReason::CostCapReached`](super::CircuitBreakReason::CostCapReached)
//! and
//! [`CircuitBreakReason::QualityDeclineNoRecovery`](super::CircuitBreakReason::QualityDeclineNoRecovery).
//!
//! - **Suppresses**:
//!   [`CircuitBreakReason::RepeatedToolCallLoop`](super::CircuitBreakReason::RepeatedToolCallLoop),
//!   [`Decision::ScopeDriftWarn`](super::Decision::ScopeDriftWarn),
//!   [`Decision::ProceduralWarning`](super::Decision::ProceduralWarning),
//!   [`Decision::Continue`](super::Decision::Continue). The two cost
//!   variants are themselves ordered `CostCapReached >
//!   QualityDeclineNoRecovery` inside
//!   [`Regulator::decide`](super::Regulator::decide).
//! - **Suppressed by**: nothing — cost-driven circuit breaks are the
//!   highest-priority signals the regulator emits.
//! - **Inactive when**: the cumulative token / wallclock counters have
//!   not crossed the cap, OR the rolling quality history is still too
//!   shallow to evaluate [`QUALITY_DECLINE_WINDOW`] /
//!   [`POOR_QUALITY_MEAN`]. Both predicates AND the quality guard —
//!   a cap reached with still-high quality does not fire.
//!
//! ## CircuitBreak predicates
//!
//! The accumulator exposes three queries the Regulator uses to decide
//! whether to fire a [`Decision::CircuitBreak`](super::Decision::CircuitBreak):
//!
//! - [`total_tokens_out`](CostAccumulator::total_tokens_out) vs
//!   [`cap_tokens`](CostAccumulator::cap_tokens) — hard budget cap.
//! - [`mean_quality_last_n`](CostAccumulator::mean_quality_last_n) —
//!   average recent quality; pairs with the cap predicate so budget
//!   isn't severed while quality is still fine.
//! - [`quality_decline_over_n`](CostAccumulator::quality_decline_over_n) —
//!   first-vs-last drop across a trailing window; detects
//!   recovery-failure patterns independently of the budget cap.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::math::clamp;

// ── Constants ──────────────────────────────────────────────────────────

/// Default cumulative output-token cap for one Regulator instance.
///
/// 10000 tokens ≈ 10 typical single-turn replies (each ~1000 tokens).
/// Beyond this, an agent working on a single task is usually stuck in
/// a retry loop; the cap + poor-quality predicate gives a graceful
/// stop point. Callers tuning for demos or tight-budget flows override
/// via [`CostAccumulator::with_cap`] — the plan test target uses 1000.
pub const DEFAULT_TOKEN_CAP: u32 = 10_000;

/// Maximum number of per-turn quality values retained for trend analysis.
///
/// 20 turns balances memory footprint against enough history for
/// decline-detection windows up to `n = 20`. VecDeque evicts the
/// oldest entry on overflow.
pub const DEFAULT_HISTORY_WINDOW: usize = 20;

/// Default window for [`quality_decline_over_n`](CostAccumulator::quality_decline_over_n).
///
/// 3 matches the plan test target ("3-turn quality decline"). Small
/// enough to react within one agent task; large enough that
/// single-turn noise can't alone fire a decline.
pub const QUALITY_DECLINE_WINDOW: usize = 3;

/// Minimum first-to-last quality drop (in `[0, 1]` units) that
/// constitutes a "decline" worth flagging.
///
/// 0.15 ≈ one quality tier (fine → mediocre, mediocre → bad). Pairs
/// with [`POOR_QUALITY_MEAN`] to avoid flagging responses that are
/// merely oscillating between good values.
pub const QUALITY_DECLINE_MIN_DELTA: f64 = 0.15;

/// Mean-quality threshold below which the recent window is treated as
/// "poor" for CircuitBreak purposes.
///
/// 0.5 = neutral midpoint on `[0, 1]`. Matches the
/// [`NEUTRAL_CONFIDENCE`](super::token_stats::NEUTRAL_CONFIDENCE)
/// constant used by `token_stats` so Path 2 signals share the same
/// "below neutral = worrying" axis.
pub const POOR_QUALITY_MEAN: f64 = 0.5;

/// Typical per-turn output-token count used as the denominator in
/// cost normalization.
///
/// 1000 tokens ≈ 750 English words ≈ 4–5 paragraphs. A turn larger
/// than this is "longer than typical" — the normalization saturates at
/// 1.0 so runaway replies still deplete the full per-turn body-budget
/// rate rather than exploding the scale.
pub const TYPICAL_TURN_TOKENS_OUT: u32 = 1_000;

/// Typical per-turn wallclock in milliseconds used as the denominator
/// in cost normalization.
///
/// 10_000 ms = 10 seconds. Interactive assistant responses target
/// under this; batch agent turns can run 60 s+. 10 s is the interactive
/// "this is expensive" boundary.
pub const TYPICAL_TURN_WALLCLOCK_MS: u32 = 10_000;

/// Weight on the token component of normalized cost.
///
/// 0.7: tokens are the direct billing metric on every major provider
/// (Anthropic, OpenAI, Google), so they dominate. Wallclock at 0.3
/// captures infrastructure / latency cost that doesn't show up in
/// billing but matters for SLA budgets. Sum of the two weights is 1.0
/// by construction.
pub const TOKEN_COST_WEIGHT: f64 = 0.7;

// ── CostAccumulator ───────────────────────────────────────────────────

/// Cumulative cost counters + rolling quality history.
///
/// Lifecycle: unlike [`TokenStatsAccumulator`](super::token_stats::TokenStatsAccumulator)
/// and [`ScopeTracker`](super::scope::ScopeTracker) (both per-turn),
/// this accumulator persists **across turns for the lifetime of the
/// Regulator instance**. The "agent task" is the whole multi-turn run;
/// the cap is cumulative. Start a fresh [`Regulator`](super::Regulator)
/// to reset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAccumulator {
    total_tokens_in: u32,
    total_tokens_out: u32,
    /// Cumulative wallclock in ms. `u64` to avoid overflow on long
    /// runs (u32 ms caps at ~49 days; u64 ms at ~584 million years).
    total_wallclock_ms: u64,
    /// Cumulative number of `Cost` events received.
    turn_count: usize,
    /// Cap on `total_tokens_out` for CircuitBreak predicate.
    cap_tokens: u32,
    /// Rolling window of per-turn quality values. Bounded by
    /// `history_window`; oldest entries evict on overflow.
    quality_history: VecDeque<f64>,
    history_window: usize,
}

impl CostAccumulator {
    /// Construct with the default token cap ([`DEFAULT_TOKEN_CAP`]) and
    /// history window ([`DEFAULT_HISTORY_WINDOW`]).
    pub fn new() -> Self {
        Self::with_cap(DEFAULT_TOKEN_CAP)
    }

    /// Construct with a custom token cap. Useful for demos and tests
    /// (the plan test target uses cap = 1000) and for callers that
    /// want a tighter budget envelope than [`DEFAULT_TOKEN_CAP`].
    pub fn with_cap(cap_tokens: u32) -> Self {
        Self {
            total_tokens_in: 0,
            total_tokens_out: 0,
            total_wallclock_ms: 0,
            turn_count: 0,
            cap_tokens,
            quality_history: VecDeque::new(),
            history_window: DEFAULT_HISTORY_WINDOW,
        }
    }

    /// Mutable: record one turn's cost. Requires mutation because the
    /// accumulator folds each turn into the running totals and
    /// turn-count.
    ///
    /// Uses saturating arithmetic on the token counters so runaway
    /// inputs clamp at `u32::MAX` / `u64::MAX` rather than wrap
    /// silently (P5 fail-open).
    pub fn record_cost(&mut self, tokens_in: u32, tokens_out: u32, wallclock_ms: u32) {
        self.total_tokens_in = self.total_tokens_in.saturating_add(tokens_in);
        self.total_tokens_out = self.total_tokens_out.saturating_add(tokens_out);
        self.total_wallclock_ms = self
            .total_wallclock_ms
            .saturating_add(u64::from(wallclock_ms));
        self.turn_count = self.turn_count.saturating_add(1);
    }

    /// Mutable: record one turn's ground-truth quality. Requires
    /// mutation because the rolling history must advance per event.
    ///
    /// Non-finite values (NaN, ±Inf) are silently dropped (P5
    /// fail-open) — they can't participate meaningfully in a `[0, 1]`
    /// rolling mean and would poison downstream predicates otherwise.
    /// Finite values are clamped to `[0, 1]`.
    pub fn record_quality(&mut self, quality: f64) {
        if !quality.is_finite() {
            return;
        }
        let q = clamp(quality, 0.0, 1.0);
        self.quality_history.push_back(q);
        if self.quality_history.len() > self.history_window {
            self.quality_history.pop_front();
        }
    }

    // ── Counters ──────────────────────────────────────────────────────

    pub fn total_tokens_in(&self) -> u32 {
        self.total_tokens_in
    }

    pub fn total_tokens_out(&self) -> u32 {
        self.total_tokens_out
    }

    pub fn total_wallclock_ms(&self) -> u64 {
        self.total_wallclock_ms
    }

    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    pub fn cap_tokens(&self) -> u32 {
        self.cap_tokens
    }

    /// Mutable: update the cap without touching accumulated counters.
    /// Requires mutation because the cap is stored state; preserves
    /// the running totals so callers can tune the cap mid-task
    /// without losing history.
    pub fn set_cap(&mut self, cap: u32) {
        self.cap_tokens = cap;
    }

    /// Whether `total_tokens_out` has reached or exceeded the cap.
    pub fn cap_reached(&self) -> bool {
        self.total_tokens_out >= self.cap_tokens
    }

    // ── Quality predicates ────────────────────────────────────────────

    /// Mean of the last `n` recorded quality values. `None` when fewer
    /// than `n` values have been recorded or when `n == 0`.
    pub fn mean_quality_last_n(&self, n: usize) -> Option<f64> {
        if n == 0 || self.quality_history.len() < n {
            return None;
        }
        let start = self.quality_history.len() - n;
        let sum: f64 = self.quality_history.iter().skip(start).sum();
        Some(sum / n as f64)
    }

    /// First-to-last drop across the trailing `n` quality values, in
    /// `[0, 1]` units. Returns `Some(delta)` when
    /// `oldest − newest ≥ min_delta` (a true decline), otherwise
    /// `None`.
    ///
    /// `None` when fewer than `n` values have been recorded, when
    /// `n < 2` (no trend possible), or when the window is stable /
    /// improving.
    pub fn quality_decline_over_n(&self, n: usize, min_delta: f64) -> Option<f64> {
        if n < 2 || self.quality_history.len() < n {
            return None;
        }
        // VecDeque::iter() yields front-to-back (oldest first in our
        // push_back convention). Take the tail of length n.
        let start = self.quality_history.len() - n;
        let oldest = *self.quality_history.get(start)?;
        let newest = *self.quality_history.back()?;
        let delta = oldest - newest;
        if delta >= min_delta {
            Some(delta)
        } else {
            None
        }
    }
}

impl Default for CostAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Normalization ─────────────────────────────────────────────────────

/// Map a single turn's `(tokens_out, wallclock_ms)` onto the `[0, 1]`
/// cost scale accepted by
/// [`CognitiveSession::track_cost`](crate::session::CognitiveSession::track_cost).
///
/// See the module docs for the weighted-blend formula and rationale.
///
/// ## Clamp behaviour on long-running sessions
///
/// Both `tokens_out` and `wallclock_ms` are `u32`. The weighted blend
/// itself clamps each component at `1.0`, so anything past
/// `TYPICAL_TURN_*` contributes nothing extra. The u32 ceiling kicks
/// in only if a single turn somehow accumulates more than `u32::MAX`
/// ms of wallclock (~49.7 days) or 4 billion tokens — both of which
/// should be interpreted as infrastructure failure rather than a
/// legitimate cost signal. The upstream OTel adapter
/// ([`crate::regulator::otel::events_from_span`]) saturates long
/// spans to `u32::MAX` rather than wrapping, so clock-skew or zombie
/// spans can't flip the sign. Apps running for weeks should feed
/// costs per turn, not accumulated.
pub fn normalize_cost(tokens_out: u32, wallclock_ms: u32) -> f64 {
    let tok_component = clamp(
        f64::from(tokens_out) / f64::from(TYPICAL_TURN_TOKENS_OUT),
        0.0,
        1.0,
    );
    let wc_component = clamp(
        f64::from(wallclock_ms) / f64::from(TYPICAL_TURN_WALLCLOCK_MS),
        0.0,
        1.0,
    );
    TOKEN_COST_WEIGHT * tok_component + (1.0 - TOKEN_COST_WEIGHT) * wc_component
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Counters ──────────────────────────────────────────────────────

    #[test]
    fn empty_accumulator_reports_zeros() {
        let acc = CostAccumulator::new();
        assert_eq!(acc.total_tokens_in(), 0);
        assert_eq!(acc.total_tokens_out(), 0);
        assert_eq!(acc.total_wallclock_ms(), 0);
        assert_eq!(acc.turn_count(), 0);
        assert!(!acc.cap_reached());
        assert_eq!(acc.cap_tokens(), DEFAULT_TOKEN_CAP);
    }

    #[test]
    fn record_cost_accumulates_across_turns() {
        let mut acc = CostAccumulator::new();
        acc.record_cost(100, 200, 500);
        acc.record_cost(50, 150, 1_000);
        assert_eq!(acc.total_tokens_in(), 150);
        assert_eq!(acc.total_tokens_out(), 350);
        assert_eq!(acc.total_wallclock_ms(), 1_500);
        assert_eq!(acc.turn_count(), 2);
    }

    #[test]
    fn cap_reached_fires_at_or_above_cap() {
        let mut acc = CostAccumulator::with_cap(1_000);
        acc.record_cost(0, 500, 0);
        assert!(!acc.cap_reached());
        acc.record_cost(0, 500, 0);
        assert!(acc.cap_reached(), "cap exactly met should be reached");
        acc.record_cost(0, 100, 0);
        assert!(acc.cap_reached(), "cap exceeded should stay reached");
    }

    #[test]
    fn saturating_adds_handle_overflow_gracefully() {
        let mut acc = CostAccumulator::new();
        acc.record_cost(u32::MAX, u32::MAX, u32::MAX);
        acc.record_cost(1, 1, 1);
        // Should not panic / wrap — u32 totals saturate, u64 wallclock
        // has headroom.
        assert_eq!(acc.total_tokens_in(), u32::MAX);
        assert_eq!(acc.total_tokens_out(), u32::MAX);
    }

    // ── Quality recording ─────────────────────────────────────────────

    #[test]
    fn record_quality_clamps_and_skips_nonfinite() {
        let mut acc = CostAccumulator::new();
        acc.record_quality(0.8);
        acc.record_quality(-1.0); // below range → clamped to 0.0
        acc.record_quality(2.0); // above range → clamped to 1.0
        acc.record_quality(f64::NAN); // dropped
        acc.record_quality(f64::INFINITY); // dropped

        // Three finite values recorded (0.8, 0.0, 1.0).
        let mean = acc.mean_quality_last_n(3).expect("three values stored");
        assert!(
            ((0.8 + 0.0 + 1.0) / 3.0 - mean).abs() < 1e-9,
            "mean should reflect clamped finite values only (got {mean})"
        );
    }

    #[test]
    fn quality_history_evicts_oldest_on_overflow() {
        let mut acc = CostAccumulator::new();
        for _ in 0..DEFAULT_HISTORY_WINDOW + 5 {
            acc.record_quality(0.5);
        }
        // History bounded; asking for the last DEFAULT_HISTORY_WINDOW
        // entries still works, but asking for more returns None.
        assert!(acc.mean_quality_last_n(DEFAULT_HISTORY_WINDOW).is_some());
        assert!(acc
            .mean_quality_last_n(DEFAULT_HISTORY_WINDOW + 1)
            .is_none());
    }

    // ── mean_quality_last_n ───────────────────────────────────────────

    #[test]
    fn mean_quality_last_n_empty_is_none() {
        let acc = CostAccumulator::new();
        assert!(acc.mean_quality_last_n(3).is_none());
    }

    #[test]
    fn mean_quality_last_n_zero_is_none() {
        // Defensive: n = 0 would divide-by-zero. Must return None.
        let mut acc = CostAccumulator::new();
        acc.record_quality(0.9);
        assert!(acc.mean_quality_last_n(0).is_none());
    }

    #[test]
    fn mean_quality_last_n_computes_trailing_mean() {
        let mut acc = CostAccumulator::new();
        for q in [0.1, 0.2, 0.3, 0.4, 0.5] {
            acc.record_quality(q);
        }
        let mean3 = acc.mean_quality_last_n(3).expect("three trailing values");
        assert!(
            ((0.3 + 0.4 + 0.5) / 3.0 - mean3).abs() < 1e-9,
            "mean of trailing 3 should be 0.4 (got {mean3})"
        );
    }

    // ── quality_decline_over_n ────────────────────────────────────────

    #[test]
    fn quality_decline_detects_monotonic_drop() {
        let mut acc = CostAccumulator::new();
        for q in [0.9, 0.7, 0.5, 0.3] {
            acc.record_quality(q);
        }
        let delta = acc
            .quality_decline_over_n(3, 0.15)
            .expect("three-turn decline must fire");
        // Window = last 3 = [0.7, 0.5, 0.3]. Oldest 0.7, newest 0.3.
        assert!((delta - 0.4).abs() < 1e-9);
    }

    #[test]
    fn quality_decline_returns_none_when_stable() {
        let mut acc = CostAccumulator::new();
        for q in [0.7, 0.65, 0.72, 0.68, 0.7] {
            acc.record_quality(q);
        }
        assert!(
            acc.quality_decline_over_n(3, 0.15).is_none(),
            "stable quality must not register as declining"
        );
    }

    #[test]
    fn quality_decline_returns_none_when_improving() {
        let mut acc = CostAccumulator::new();
        for q in [0.3, 0.5, 0.7] {
            acc.record_quality(q);
        }
        // Oldest - newest = 0.3 - 0.7 = -0.4. Negative delta, no
        // decline.
        assert!(acc.quality_decline_over_n(3, 0.15).is_none());
    }

    #[test]
    fn quality_decline_requires_min_points() {
        let mut acc = CostAccumulator::new();
        acc.record_quality(0.9);
        acc.record_quality(0.3);
        // Only 2 points; with n=3 we need 3.
        assert!(acc.quality_decline_over_n(3, 0.1).is_none());
        // With n=2 and 2 points, the decline does fire.
        assert!(acc.quality_decline_over_n(2, 0.1).is_some());
        // With n=1 (no trend possible) always None.
        assert!(acc.quality_decline_over_n(1, 0.0).is_none());
    }

    #[test]
    fn quality_decline_below_threshold_returns_none() {
        let mut acc = CostAccumulator::new();
        for q in [0.7, 0.65, 0.60] {
            acc.record_quality(q);
        }
        // Delta = 0.1 < min_delta 0.15 → no decline.
        assert!(acc.quality_decline_over_n(3, 0.15).is_none());
    }

    // ── normalize_cost ────────────────────────────────────────────────

    #[test]
    fn normalize_cost_typical_turn_is_midrange() {
        // Typical: 1000 tokens out at 10s wallclock → both components
        // = 1.0. Mix = 0.7 + 0.3 = 1.0 at cap.
        let c = normalize_cost(TYPICAL_TURN_TOKENS_OUT, TYPICAL_TURN_WALLCLOCK_MS);
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_cost_zero_input_is_zero() {
        assert_eq!(normalize_cost(0, 0), 0.0);
    }

    #[test]
    fn normalize_cost_half_typical_gives_half_weight() {
        // 500 tokens + 5s wallclock → both components 0.5 → mix 0.5.
        let c = normalize_cost(500, 5_000);
        assert!((c - 0.5).abs() < 1e-9);
    }

    #[test]
    fn normalize_cost_clamps_runaway_turn() {
        // 10× the typical budget should saturate, not explode.
        let c = normalize_cost(TYPICAL_TURN_TOKENS_OUT * 10, TYPICAL_TURN_WALLCLOCK_MS * 10);
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_cost_weights_tokens_dominantly() {
        // Maximum tokens, zero wallclock → only token component fires.
        // Result should be exactly TOKEN_COST_WEIGHT (0.7).
        let c = normalize_cost(TYPICAL_TURN_TOKENS_OUT, 0);
        assert!((c - TOKEN_COST_WEIGHT).abs() < 1e-9);

        // Zero tokens, maximum wallclock → only wallclock component fires.
        // Result should be 1 - TOKEN_COST_WEIGHT (0.3).
        let c = normalize_cost(0, TYPICAL_TURN_WALLCLOCK_MS);
        assert!((c - (1.0 - TOKEN_COST_WEIGHT)).abs() < 1e-9);
    }
}
