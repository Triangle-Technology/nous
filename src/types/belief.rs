//! Unified belief model — SharedBeliefState (Friston 2010, Pessoa 2023, Barrett 2017).
//!
//! Affect as continuous ambient signal permeating all processing.
//! Per-conversation state with PERCEIVE → REMEMBER → POST-RESPONSE lifecycle.

use serde::{Deserialize, Serialize};

/// Emotional valence polarity (insula).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AffectValence {
    Positive,
    Negative,
    #[default]
    Neutral,
}

/// Amygdala-inspired affect state.
///
/// - `arousal`: 0-1, amygdala activation (LeDoux 1996)
/// - `valence`: polarity (insula)
/// - `certainty`: 0-1, confidence in perceptual reading
/// - `sustained`: 0-1, Yerkes-Dodson modulator (decays over time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectState {
    pub arousal: f64,
    pub valence: AffectValence,
    pub certainty: f64,
    pub sustained: f64,
}

impl Default for AffectState {
    fn default() -> Self {
        Self {
            arousal: 0.0,
            valence: AffectValence::Neutral,
            certainty: 0.5,
            sustained: 1.0,
        }
    }
}

/// Topic cluster keys — opaque hash for cross-session strategy learning.
///
/// NOT a cognitive topic model. Per 2026-04-11 audit, topic tracking is the
/// model's job (attention handles it natively per research). This struct
/// exists only because dorsomedial striatum-analog strategy EMA needs a
/// stable key for per-cluster learning. Topics are regex-extracted purely
/// as a hashing function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopicBeliefs {
    /// Currently active topic-cluster key components (used as opaque hash).
    pub current: Vec<String>,
    /// Predicted next-turn topic-cluster key components.
    pub predicted: Vec<String>,
}

/// Predictions about the next turn.
///
/// After the 2026-04-11 audit only `next_topics` is retained. It is used as
/// an opaque cluster-key hash for cross-session strategy-success EMA lookup
/// in `world_model::consolidate` (dorsomedial striatum analog, Daw 2005).
/// It is NOT a claim that Nous tracks topics cognitively — the model does
/// that natively via attention.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Predictions {
    pub next_topics: Vec<String>,
}

/// Unified belief state — per-conversation, shared across all cognitive modules.
///
/// Brain basis: Barrett 2017 (affect as ambient signal that permeates all
/// processing). Post-2026-04-11 audit scope: affect + turn counter + topic
/// cluster hash + strategy-prediction hash. All cortical duplication
/// (user model, knowledge model, intent prediction) removed — those were
/// either orphaned or overlapped with the model's own processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedBeliefState {
    pub affect: AffectState,
    pub topic: TopicBeliefs,
    pub predictions: Predictions,
    pub turn: u32,
    pub conversation_id: String,
}

impl SharedBeliefState {
    pub fn new(conversation_id: String) -> Self {
        Self {
            affect: AffectState::default(),
            topic: TopicBeliefs::default(),
            predictions: Predictions::default(),
            turn: 0,
            conversation_id,
        }
    }

    /// Whether arousal exceeds the encoding threshold (flashbulb memory cutoff).
    pub fn is_high_arousal(&self) -> bool {
        self.affect.arousal >= 0.6
    }
}

// ── Constants ──────────────────────────────────────────────────────────

/// Below this, certainty is low.
pub const LOW_AROUSAL_CERTAINTY: f64 = 0.3;
/// Above this, certainty is high.
pub const HIGH_AROUSAL_CERTAINTY: f64 = 0.8;
/// Max predicted topics per turn.
pub const PREDICTION_TOPIC_COUNT: usize = 5;
