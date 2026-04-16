//! Thalamic Gate — compute-saving routing classification.
//!
//! Brain analog: thalamus filters ~60% of sensory input before reaching cortex.
//! Classifies each message into 3 types to control downstream processing depth.
//!
//! ## Scope clarification (2026-04-11 audit)
//!
//! This is **compute-saving pipeline routing**, not cognitive classification.
//! The regex patterns are heuristics for "is this worth running full pipeline
//! on" — NOT a claim about thalamic function. Real thalamic relay involves
//! sensory filtering before cortex, which has no analog in an SSM (tokens
//! arrive already encoded). What we keep is the *pipeline-depth gating*
//! role of the thalamus (Crick 1984 reticular complex).
//!
//! The `Familiar` classification was deleted in the audit because it relied
//! on regex topic overlap with a context vector — cortical duplication per
//! P9. Cross-turn "familiarity" is now derived from sensory PE in
//! `cognition/dynamics.rs` (Behrens 2007 ACC volatility signal).
//!
//! ## Gating (P10)
//!
//! Priority order: URGENT > ROUTINE > NOVEL (default).
//! URGENT is gated by amygdala salience (high arousal → override).
//! ROUTINE only fires for short acknowledgments.
//!
//! Brain analog: thalamus filters ~60% of sensory input before reaching cortex.
//! Key papers: Crick 1984 (reticular complex), LeDoux 1996 (fast pathway).
//!
//! Pure function, <1ms, $0 LLM cost.

use regex::Regex;
use std::sync::LazyLock;

use crate::types::gate::{GateContext, GateContextWithFeedback, GateResult, GateType};

// ── Constants ──────────────────────────────────────────────────────────

/// Max message length for routine classification.
const ROUTINE_MAX_LENGTH: usize = 20;
/// Amygdala arousal threshold for urgent override.
const AMYGDALA_AROUSAL_THRESHOLD: f64 = 0.6;
/// Min alpha chars to compute CAPS ratio.
const CAPS_MIN_LENGTH: usize = 8;
/// CAPS ratio threshold for urgency.
const CAPS_RATIO_THRESHOLD: f64 = 0.5;

// ── Pattern Definitions ────────────────────────────────────────────────

// P9b note: English-language lexicon regex retained as interim heuristic
// for pipeline-depth routing. Vietnamese lexicon removed 2026-04-14 — the
// eval path does not use Vietnamese input, and the English patterns already
// cover the semantic categories (error/urgency/correction/help). If VN
// coverage is needed again, the correct path is an LLM-driven classification
// at the application layer, not lexicon duplication in the gate.

static ROUTINE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)^(ok|okay|yes|yep|yeah|sure|thanks|thank you|got it|understood|right|alright|cool|nice|great|good|fine|perfect|exactly|correct|agreed|noted|i see)\.?!?$").expect("valid regex"),
        Regex::new(r"^[\p{Emoji}\s]{1,4}$").expect("valid regex"),
        Regex::new(r"^\d{1,2}\.?$").expect("valid regex"),
    ]
});

static URGENT_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?i)\b(error|crash|bug|fail|broke|broken|block)\w*\b").expect("valid regex"),
        Regex::new(r"(?i)\b(urgent|asap|deadline|immediately)\b").expect("valid regex"),
        Regex::new(r"(?i)\b(actually|wait|no i meant|that'?s wrong)\b").expect("valid regex"),
        Regex::new(r"(?i)\b(help!|please fix|fix now|need help)\b").expect("valid regex"),
    ]
});

// ── Core Classification ────────────────────────────────────────────────

/// Classify a message through the thalamic gate.
///
/// Priority: URGENT > ROUTINE > NOVEL (default).
pub fn classify_gate(ctx: &GateContext) -> GateResult {
    let message = ctx.message.trim();

    // 1. URGENT — error/crisis/correction + high arousal
    if check_urgent(message, ctx.arousal) {
        return GateResult {
            gate: GateType::Urgent,
            confidence: if ctx.arousal >= AMYGDALA_AROUSAL_THRESHOLD {
                0.85
            } else {
                0.7
            },
            reason: String::from("urgent: error/crisis/correction pattern detected"),
        };
    }

    // 2. ROUTINE — short acknowledgments
    if check_routine(message) {
        return GateResult {
            gate: GateType::Routine,
            confidence: 0.95,
            reason: String::from("routine: short acknowledgment"),
        };
    }

    // 3. Default: NOVEL
    GateResult {
        gate: GateType::Novel,
        confidence: 0.5,
        reason: String::from("novel: full pipeline"),
    }
}

/// Extended gate classification with convergence loop feedback.
///
/// Feedback connections:
/// - Arousal amplification (LeDoux 1996): previous gate urgency amplifies
///   current classification when arousal remains elevated.
/// - Resource pressure (Barrett 2017): high allostatic load lowers urgent
///   confidence (conserve compute under load — the opposite of panic).
pub fn classify_gate_with_feedback(ctx: &GateContextWithFeedback) -> GateResult {
    let mut result = classify_gate(&ctx.base);

    // Feedback 1: Arousal amplification from previous gate
    if let Some(prev) = ctx.previous_gate {
        if prev.gate == GateType::Urgent && ctx.base.arousal >= 0.4 {
            // Previous urgency + continuing arousal → maintain urgency
            if result.gate != GateType::Urgent {
                result.gate = GateType::Urgent;
                result.confidence = 0.7;
                result.reason = String::from("urgent: arousal amplification from previous gate");
            }
        }
    }

    // Feedback 2: Resource pressure reduces urgent confidence
    // High allostatic load → downgrade urgency confidence (Barrett 2017).
    // The system should not panic when already stressed — compensatory calm.
    if ctx.resource_pressure > 0.7 && result.gate == GateType::Urgent {
        result.confidence = (result.confidence - 0.15).max(0.5);
        result.reason = format!("{} [pressure-damped]", result.reason);
    }

    result
}

fn check_urgent(message: &str, arousal: f64) -> bool {
    // High arousal alone can trigger urgent (fast amygdala pathway)
    if arousal >= AMYGDALA_AROUSAL_THRESHOLD {
        return true;
    }

    // CAPS ratio check
    let alpha_chars: Vec<char> = message.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha_chars.len() >= CAPS_MIN_LENGTH {
        let upper = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
        if upper as f64 / alpha_chars.len() as f64 >= CAPS_RATIO_THRESHOLD {
            return true;
        }
    }

    // Pattern match
    URGENT_PATTERNS.iter().any(|p| p.is_match(message))
}

fn check_routine(message: &str) -> bool {
    if message.len() > ROUTINE_MAX_LENGTH {
        return false;
    }
    ROUTINE_PATTERNS.iter().any(|p| p.is_match(message))
}

// ── Problem Type Classification ────────────────────────────────────────

/// Classify the problem type for mode selection.
pub fn classify_problem_type(
    message: &str,
    dimension_count: usize,
    arousal: f64,
) -> crate::types::gate::ProblemType {
    use crate::types::gate::ProblemType;

    if dimension_count >= 4 && arousal >= 0.3 {
        return ProblemType::SystemDilemma;
    }
    if dimension_count >= 2 {
        return ProblemType::Dilemma;
    }
    if message.contains('?') {
        return ProblemType::Question;
    }

    // Check for task indicators (P9b: English-only interim heuristic,
    // pending replacement by app-layer classification or hidden-state signal).
    static TASK_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
        vec![
            Regex::new(r"(?i)\b(create|build|implement|write|fix|update|add|remove|refactor|deploy)\b").expect("valid regex"),
        ]
    });

    if TASK_PATTERNS.iter().any(|p| p.is_match(message)) {
        return ProblemType::Task;
    }

    ProblemType::SimpleChat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx<'a>(message: &'a str, arousal: f64) -> GateContext<'a> {
        GateContext {
            message,
            recent_messages: &[],
            arousal,
        }
    }

    #[test]
    fn routine_ok() {
        let ctx = make_ctx("ok", 0.0);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Routine);
        assert!(r.confidence > 0.9);
    }

    #[test]
    fn routine_emoji() {
        let ctx = make_ctx("👍", 0.0);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Routine);
    }

    #[test]
    fn urgent_error_keyword() {
        let ctx = make_ctx("I got an error in the build", 0.0);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Urgent);
    }

    #[test]
    fn urgent_high_arousal() {
        let ctx = make_ctx("something happened", 0.7);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Urgent);
    }

    #[test]
    fn urgent_caps() {
        let ctx = make_ctx("WHY IS THIS BROKEN", 0.0);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Urgent);
    }

    #[test]
    fn novel_default() {
        let ctx = make_ctx("Let's discuss quantum computing approaches", 0.0);
        let r = classify_gate(&ctx);
        assert_eq!(r.gate, GateType::Novel);
    }

    #[test]
    fn feedback_pressure_damps_urgent_confidence() {
        // High resource pressure should reduce urgent confidence (compensatory calm,
        // not panic when already stressed — Barrett 2017 allostasis).
        let base = make_ctx("error in the build", 0.0);
        let ctx = GateContextWithFeedback {
            base,
            resource_pressure: 0.8,
            previous_gate: None,
        };
        let r = classify_gate_with_feedback(&ctx);
        assert_eq!(r.gate, GateType::Urgent);
        assert!(r.reason.contains("pressure-damped"));
    }

    #[test]
    fn feedback_arousal_amplification() {
        let base = make_ctx("still dealing with this", 0.5);
        let prev = GateResult {
            gate: GateType::Urgent,
            confidence: 0.85,
            reason: String::from("prev urgent"),
        };
        let ctx = GateContextWithFeedback {
            base,
            resource_pressure: 0.3,
            previous_gate: Some(&prev),
        };
        let r = classify_gate_with_feedback(&ctx);
        // Previous urgency + continuing arousal → maintained urgency
        assert_eq!(r.gate, GateType::Urgent);
    }

    #[test]
    fn problem_type_question() {
        let pt = classify_problem_type("What is the best approach?", 0, 0.0);
        assert_eq!(pt, crate::types::gate::ProblemType::Question);
    }

    #[test]
    fn problem_type_task() {
        let pt = classify_problem_type("Create a new user registration flow", 0, 0.0);
        assert_eq!(pt, crate::types::gate::ProblemType::Task);
    }

    #[test]
    fn problem_type_dilemma() {
        let pt = classify_problem_type("Should we use microservices or monolith", 2, 0.0);
        assert_eq!(pt, crate::types::gate::ProblemType::Dilemma);
    }
}
