//! Thalamic gate types — message classification controlling pipeline depth.
//!
//! Brain analog: thalamus filters ~60% of sensory input before reaching cortex.
//! This module now provides compute-saving routing (3-type classification),
//! not cognitive classification. The original `Familiar` variant was deleted
//! in the 2026-04-11 audit pass 2 because it relied on regex topic overlap
//! (cortical duplication per P9). Cross-turn "familiarity" is now derived
//! from sensory PE in `cognition/dynamics.rs`.

use serde::{Deserialize, Serialize};

/// Gate classification — controls how deep the processing pipeline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateType {
    /// Short acknowledgments ("ok", "yes", emojis). Skip enrichment entirely.
    Routine,
    /// New topic, full enrichment pipeline.
    #[default]
    Novel,
    /// Error/crisis/correction. Highest resource allocation.
    Urgent,
}

/// Result of thalamic gate classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate: GateType,
    /// 0-1, classification confidence.
    pub confidence: f64,
    /// Human-readable explanation (P8: document why).
    pub reason: String,
}

impl Default for GateResult {
    fn default() -> Self {
        Self {
            gate: GateType::Novel,
            confidence: 0.5,
            reason: String::from("default: novel"),
        }
    }
}

/// Input context for gate classification.
#[derive(Debug, Clone)]
pub struct GateContext<'a> {
    pub message: &'a str,
    pub recent_messages: &'a [RecentMessage],
    /// 0-1, from emotional valence.
    pub arousal: f64,
}

/// Extended gate context with convergence loop feedback.
#[derive(Debug, Clone)]
pub struct GateContextWithFeedback<'a> {
    pub base: GateContext<'a>,
    /// 0-1, allostatic load from resource allocator.
    pub resource_pressure: f64,
    /// Previous iteration's gate result (for arousal amplification).
    pub previous_gate: Option<&'a GateResult>,
}

/// A recent message in conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentMessage {
    pub role: String,
    pub content: String,
}

/// Problem type classification — determines which processing mode to activate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProblemType {
    SimpleChat,
    Task,
    Question,
    Dilemma,
    SystemDilemma,
}
