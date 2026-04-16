//! Locus Coeruleus — NE gain modulation (phasic/tonic/neutral).
//!
//! Brain analog: locus coeruleus projects norepinephrine broadly to cortex,
//! modulating gain without carrying content. Phasic bursts on salient stimuli
//! narrow tuning (exploit); tonic elevation broadens it (explore).
//!
//! ## Mechanism (not metaphor — P1)
//!
//! Substrate: LC is a small brainstem nucleus (~1500 neurons per side) whose
//! NE projections reach the entire cortex. Transformation: phasic burst (brief,
//! high-amplitude) vs tonic (slow, sustained) firing modes determine whether
//! cortical neurons sharpen (exploit current task) or broaden (scan for
//! alternatives). Gating: LC itself receives salience input from the amygdala
//! (fast arousal override) and prefrontal confidence (slow EMA task-utility).
//!
//! ## Gating (P10)
//!
//! - **Suppresses**: implicitly narrows downstream precision-based thresholds
//!   via gain_mode feedback through adaptive_thresholds
//! - **Suppressed by**: nothing — LC is a modulator, not a discrete signal
//! - **Fast override**: high arousal (≥ AROUSAL_PHASIC_THRESHOLD) immediately
//!   forces phasic mode, bypassing the confidence EMA. This is the amygdala
//!   low-road → LC coupling (Aston-Jones 2005, Sara 2009).
//!
//! Key papers: Aston-Jones & Cohen 2005 (adaptive gain theory),
//! Sara 2009 (LC-NE in memory and attention), Berridge 2008 (LC firing modes).
//!
//! ## Cortical functions removed (2026-04)
//!
//! This module previously contained dlPFC goal maintenance, mPFC context
//! tracking, ACC conflict monitoring, and topic-switch detection. Per P9
//! (compensate, don't amplify), these duplicated cortical work the model
//! performs natively. They were deleted after the audit found no external
//! callers — their 9 tests were internal self-validation only.
//!
//! Stateful module (mutable LC state), <1ms per call, $0 LLM cost.

use crate::types::world::{GainMode, LearnedState};

// ── Constants ──────────────────────────────────────────────────────────

/// EMA rate for gain mode smoothing.
const GAIN_EMA_RATE: f64 = 0.35;
/// Smoothed confidence above this = phasic.
const PHASIC_THRESHOLD: f64 = 0.7;
/// Smoothed confidence below this = tonic.
const TONIC_THRESHOLD: f64 = 0.3;
/// Arousal above this triggers phasic gain directly (Aston-Jones 2005).
/// LC responds to salient stimuli with phasic NE burst independent of PFC confidence.
/// 0.35: calibrated to emotional.rs output range (stress ≈ 0.35-0.5, anger ≈ 0.5+).
const AROUSAL_PHASIC_THRESHOLD: f64 = 0.35;

// ── State ──────────────────────────────────────────────────────────────

/// Locus coeruleus gain state.
///
/// Tracks the current NE gain mode and the confidence EMA that drives
/// tonic-phasic transitions. High arousal can override the EMA directly
/// (amygdala fast pathway to LC, Sara 2009).
#[derive(Debug, Clone)]
pub struct LocusCoeruleus {
    pub tick: u32,
    gain_mode: GainMode,
    gain_smoothed_confidence: f64,
    /// Last arousal level for LC gain mode triggering (Aston-Jones 2005).
    last_arousal: f64,
}

impl LocusCoeruleus {
    pub fn new() -> Self {
        Self {
            tick: 0,
            gain_mode: GainMode::Neutral,
            gain_smoothed_confidence: 0.5,
            last_arousal: 0.0,
        }
    }

    /// Get current NE gain mode.
    pub fn gain_mode(&self) -> GainMode {
        self.gain_mode
    }

    /// Mutable: update arousal level and apply LC fast-pathway override.
    ///
    /// Aston-Jones 2005: LC fires phasic bursts on salient stimuli
    /// independent of PFC confidence (amygdala low road to LC).
    /// High arousal directly forces phasic mode without waiting for EMA.
    pub fn set_arousal(&mut self, arousal: f64) {
        self.last_arousal = arousal;
        if arousal >= AROUSAL_PHASIC_THRESHOLD {
            self.gain_mode = GainMode::Phasic;
        }
    }

    /// Mutable: nudge gain mode via gate confidence EMA (slow LC modulation).
    ///
    /// Low confidence drifts toward tonic (broaden search); high confidence
    /// drifts toward phasic (exploit current tuning). Transitions are slow
    /// (~5 messages via EMA) because fast LC changes would destabilize cortex.
    pub fn nudge_gain_from_confidence(&mut self, gate_confidence: f64) {
        self.tick += 1;

        // EMA update on smoothed confidence
        self.gain_smoothed_confidence =
            (1.0 - GAIN_EMA_RATE) * self.gain_smoothed_confidence + GAIN_EMA_RATE * gate_confidence;

        if gate_confidence < 0.4
            && self.gain_mode != GainMode::Tonic
            && self.gain_smoothed_confidence <= TONIC_THRESHOLD
        {
            self.gain_mode = GainMode::Tonic;
        } else if gate_confidence > 0.85
            && self.gain_mode != GainMode::Phasic
            && self.gain_smoothed_confidence >= PHASIC_THRESHOLD
        {
            self.gain_mode = GainMode::Phasic;
        }
    }

    /// Mutable: restore gain state from persisted learned state.
    pub fn sync_from_learned(&mut self, learned: &LearnedState) {
        self.gain_mode = learned.gain_mode;
        self.tick = learned.tick;
    }

    /// Sync to learned state for persistence.
    pub fn sync_to_learned(&self, learned: &mut LearnedState) {
        learned.gain_mode = self.gain_mode;
        learned.tick = self.tick;
    }
}

impl Default for LocusCoeruleus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_neutral() {
        let lc = LocusCoeruleus::new();
        assert_eq!(lc.gain_mode(), GainMode::Neutral);
        assert_eq!(lc.tick, 0);
    }

    #[test]
    fn high_arousal_triggers_phasic() {
        let mut lc = LocusCoeruleus::new();
        lc.set_arousal(0.5);
        assert_eq!(lc.gain_mode(), GainMode::Phasic);
    }

    #[test]
    fn low_arousal_leaves_neutral() {
        let mut lc = LocusCoeruleus::new();
        lc.set_arousal(0.1);
        assert_eq!(lc.gain_mode(), GainMode::Neutral);
    }

    #[test]
    fn sustained_low_confidence_drifts_tonic() {
        let mut lc = LocusCoeruleus::new();
        // Nudge with low confidence many times to build EMA below TONIC_THRESHOLD
        for _ in 0..20 {
            lc.nudge_gain_from_confidence(0.1);
        }
        assert_eq!(lc.gain_mode(), GainMode::Tonic);
    }

    #[test]
    fn sustained_high_confidence_drifts_phasic() {
        let mut lc = LocusCoeruleus::new();
        for _ in 0..20 {
            lc.nudge_gain_from_confidence(0.9);
        }
        assert_eq!(lc.gain_mode(), GainMode::Phasic);
    }

    #[test]
    fn sync_roundtrip() {
        let mut lc = LocusCoeruleus::new();
        lc.set_arousal(0.5); // → Phasic
        lc.tick = 7;

        let mut learned = LearnedState::default();
        lc.sync_to_learned(&mut learned);

        let mut lc2 = LocusCoeruleus::new();
        lc2.sync_from_learned(&learned);
        assert_eq!(lc2.gain_mode(), GainMode::Phasic);
        assert_eq!(lc2.tick, 7);
    }
}
