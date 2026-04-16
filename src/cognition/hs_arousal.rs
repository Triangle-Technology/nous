//! Hidden State Arousal — model-derived arousal from SSM state statistics.
//!
//! Replaces regex-based arousal (`emotional.rs`) when SSM hidden state is available.
//! `emotional.rs` is preserved as fallback for API models without hs access.
//!
//! ## Brain analog
//!
//! LC reads aggregate unsigned prediction error from cortex (Grella 2024,
//! Jordan & Keller eLife 2023) — not specific content, but HOW MUCH cortical
//! state is changing. State churn IS the PE signal: high churn = state being
//! overwritten = surprise = needs compensatory retention.
//!
//! This maps directly to LC's function of detecting aggregate unsigned PE
//! and broadcasting a gain control signal (Aston-Jones & Cohen 2005).
//!
//! ## Content-type behavior (eval finding 2026-04-13)
//!
//! Theory predicted churn would differentiate content types. Eval showed:
//! - All categories have similar average churn (~0.40 on mamba-130m)
//! - Emotional: -2.04% (per-token HS better than regex -1.86%)
//! - Technical: +2.37% (HURTS — constructive state-building slowed by compensation)
//! - Creative: -0.14% (slight benefit, extends beyond regex)
//! - Routine: 0.00% (correct — low arousal, no modulation)
//!
//! **Key insight**: Not all churn is equal. Emotional churn = state disruption
//! (compensation helps). Technical churn = constructive state-building
//! (compensation hurts by slowing learning). Raw churn alone cannot distinguish.
//!
//! This module provides the SIGNAL. The discrimination between constructive
//! and destructive churn is a downstream problem for the allostatic system.
//!
//! ## Timing
//!
//! One-token delay: hs from token N feeds arousal for token N+1. This is
//! biologically correct — LC modulates the NEXT processing cycle based on
//! current cortical state, not retroactively.
//!
//! ## Gating (P10)
//!
//! - **Fires when**: SSM hidden state available AND valid (token > 0).
//! - **Does NOT fire**: first token (no previous hs), API models (no hs access).
//! - **Suppresses**: regex arousal in `resolve_arousal()` — hs signal replaces
//!   regex when valid (hs is strictly more informed than text pattern matching).
//! - **Suppressed by**: nothing — this is a source signal, not a competing
//!   downstream consumer. LC decides what to do with the arousal value.
//! - **Inactive when**: state churn below CHURN_FLOOR (routine/predictable
//!   content → arousal ≈ 0 → no modulation downstream).
//!
//! Pure functions, O(1), $0 LLM cost, no candle dependency.

use crate::math::clamp;
use crate::types::intervention::HiddenStateStats;

// ── Constants ──────────────────────────────────────────────────────────
//
// These are MODEL-SPECIFIC calibration constants, not brain-derived values.
// The brain mechanism is "state churn → arousal" (Grella 2024); these
// thresholds map mamba-130m churn range to [0, 1] arousal. Different models
// will need different values. Initial estimates — calibrate from eval data.
// Similar to resource_allocator.rs char-budget constants (application-level,
// not neural computation).

/// State churn below this = calm/routine (no compensation needed).
/// Initial estimate for mamba-130m. Needs calibration from eval data.
const CHURN_FLOOR: f64 = 0.2;

/// State churn above this = maximum arousal (saturated).
/// Initial estimate for mamba-130m. Needs calibration from eval data.
const CHURN_CEILING: f64 = 1.5;

/// How much state magnitude modulates the churn-to-arousal mapping.
/// Primary signal is churn; magnitude is secondary (more stored = more to lose).
const MAGNITUDE_WEIGHT: f64 = 0.15;

/// State magnitude normalization reference (model-specific).
/// For mamba-130m, typical hs magnitude at mid-layers is ~2.0-4.0.
const MAGNITUDE_REFERENCE: f64 = 3.0;

// ── Core Functions ────────────────────────────────────────────────────

/// Compute arousal from hidden state statistics.
///
/// Pure function. Maps SSM state churn to [0, 1] arousal scalar.
///
/// Returns None if stats are not valid (first token — no previous hs).
/// Caller should use regex fallback in that case.
pub fn arousal_from_hs(stats: &HiddenStateStats) -> Option<f64> {
    if !stats.valid {
        return None;
    }

    // Primary signal: state churn → arousal.
    // Linear mapping from [CHURN_FLOOR, CHURN_CEILING] → [0, 1].
    let base = (stats.state_churn - CHURN_FLOOR) / (CHURN_CEILING - CHURN_FLOOR);

    // Secondary signal: magnitude contribution (more stored = more to lose).
    let magnitude_factor = clamp(stats.state_magnitude / MAGNITUDE_REFERENCE, 0.0, 1.0);
    let magnitude_boost = magnitude_factor * MAGNITUDE_WEIGHT;

    Some(clamp(base + magnitude_boost, 0.0, 1.0))
}

/// Combine hs-derived arousal with regex fallback.
///
/// When hs stats are valid, use hs-derived arousal exclusively.
/// When not available or invalid, fall back to regex arousal.
pub fn resolve_arousal(hs_stats: Option<&HiddenStateStats>, regex_arousal: f64) -> f64 {
    match hs_stats {
        Some(stats) => arousal_from_hs(stats).unwrap_or(regex_arousal),
        None => regex_arousal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_stats_returns_none() {
        let stats = HiddenStateStats {
            state_churn: 0.5,
            state_magnitude: 2.0,
            valid: false,
        };
        assert!(arousal_from_hs(&stats).is_none());
    }

    #[test]
    fn zero_churn_zero_arousal() {
        let stats = HiddenStateStats {
            state_churn: 0.0,
            state_magnitude: 0.0,
            valid: true,
        };
        let arousal = arousal_from_hs(&stats).unwrap();
        assert_eq!(arousal, 0.0);
    }

    #[test]
    fn below_floor_very_low_arousal() {
        let stats = HiddenStateStats {
            state_churn: 0.1,
            state_magnitude: 1.0,
            valid: true,
        };
        let arousal = arousal_from_hs(&stats).unwrap();
        // Churn below floor → negative base, small magnitude boost.
        assert!(
            arousal < 0.1,
            "Below-floor churn should produce very low arousal, got {arousal}"
        );
    }

    #[test]
    fn mid_churn_mid_arousal() {
        let stats = HiddenStateStats {
            state_churn: 0.85, // midpoint of [0.2, 1.5]
            state_magnitude: 3.0,
            valid: true,
        };
        let arousal = arousal_from_hs(&stats).unwrap();
        assert!(
            arousal > 0.3 && arousal < 0.8,
            "Mid-range churn should produce mid arousal, got {arousal}"
        );
    }

    #[test]
    fn high_churn_high_arousal() {
        let stats = HiddenStateStats {
            state_churn: 1.2,
            state_magnitude: 3.0,
            valid: true,
        };
        let arousal = arousal_from_hs(&stats).unwrap();
        assert!(
            arousal > 0.5,
            "High churn should produce high arousal, got {arousal}"
        );
    }

    #[test]
    fn ceiling_churn_saturates() {
        let stats = HiddenStateStats {
            state_churn: 2.0,
            state_magnitude: 5.0,
            valid: true,
        };
        let arousal = arousal_from_hs(&stats).unwrap();
        assert_eq!(arousal, 1.0, "Above-ceiling churn should saturate at 1.0");
    }

    #[test]
    fn magnitude_contributes_modestly() {
        let low_mag = HiddenStateStats {
            state_churn: 0.5,
            state_magnitude: 0.5,
            valid: true,
        };
        let high_mag = HiddenStateStats {
            state_churn: 0.5,
            state_magnitude: 5.0,
            valid: true,
        };
        let a_low = arousal_from_hs(&low_mag).unwrap();
        let a_high = arousal_from_hs(&high_mag).unwrap();
        assert!(
            a_high > a_low,
            "Higher magnitude should increase arousal slightly"
        );
        assert!(
            a_high - a_low <= MAGNITUDE_WEIGHT,
            "Magnitude contribution should be bounded by MAGNITUDE_WEIGHT"
        );
    }

    #[test]
    fn resolve_uses_hs_when_valid() {
        let stats = HiddenStateStats {
            state_churn: 1.0,
            state_magnitude: 3.0,
            valid: true,
        };
        let result = resolve_arousal(Some(&stats), 0.0);
        assert!(
            result > 0.3,
            "Should use hs arousal, not regex (0.0), got {result}"
        );
    }

    #[test]
    fn resolve_falls_back_on_invalid() {
        let stats = HiddenStateStats {
            state_churn: 1.0,
            state_magnitude: 3.0,
            valid: false,
        };
        let result = resolve_arousal(Some(&stats), 0.7);
        assert_eq!(result, 0.7, "Should fall back to regex arousal on invalid hs");
    }

    #[test]
    fn resolve_falls_back_on_none() {
        let result = resolve_arousal(None, 0.5);
        assert_eq!(result, 0.5, "Should fall back to regex arousal when no hs");
    }
}
