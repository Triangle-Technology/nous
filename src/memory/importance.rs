//! Importance — Ebbinghaus forgetting curve with interference decay.
//!
//! Brain analog: Model D Pure Interference Decay (AD-38).
//! Instead of time-based forgetting, atoms decay based on how many
//! NEW atoms have been stored since last access (interference model).
//!
//! Key papers: Ebbinghaus 1885 (forgetting curve), Anderson 2000 (ACT-R decay),
//! McClelland 1995 (CLS dual-rate: episodic fast, semantic slow).
//!
//! Pure functions, <1ms, $0 LLM cost.

use crate::math::clamp;
use crate::types::memory::{AtomType, MemoryAtom};

// ── Constants ──────────────────────────────────────────────────────────

/// Episodic atoms decay faster (hippocampal fast system, CLS McClelland 1995).
pub const EPISODIC_ATOM_SCALE: f64 = 60.0;
/// Semantic atoms decay slowly (neocortical slow system, CLS McClelland 1995).
pub const SEMANTIC_ATOM_SCALE: f64 = 200.0;
/// Max feedback adjustment to importance (±15%).
pub const MAX_FEEDBACK_BOOST: f64 = 0.15;
/// Same-topic atoms cause 50% more interference (proactive interference).
pub const TOPIC_INTERFERENCE_MULTIPLIER: f64 = 1.5;
/// Default prune threshold — atoms below this effective importance get pruned.
pub const PRUNE_THRESHOLD: f64 = 0.05;

// ── Core Functions ─────────────────────────────────────────────────────

/// Get decay scale factor for atom type (CLS dual-rate).
pub fn get_atom_scale(atom_type: AtomType) -> f64 {
    match atom_type {
        AtomType::Episodic => EPISODIC_ATOM_SCALE,
        AtomType::Semantic => SEMANTIC_ATOM_SCALE,
        AtomType::Procedural => SEMANTIC_ATOM_SCALE,
        AtomType::Preference => SEMANTIC_ATOM_SCALE * 1.5, // Preferences are very stable
        AtomType::Digest => SEMANTIC_ATOM_SCALE,
        AtomType::PromptEvolution => SEMANTIC_ATOM_SCALE,
    }
}

/// Compute effective importance with interference decay.
///
/// Pure function — takes atom + context, returns decayed importance.
///
/// # Arguments
/// * `atom` — The memory atom
/// * `feedback_boost` — External feedback adjustment (-0.15 to +0.15)
/// * `current_generation` — Current interference generation counter
/// * `recent_topics` — Topics from recent context (for content-dependent interference)
pub fn compute_effective_importance(
    atom: &MemoryAtom,
    feedback_boost: f64,
    current_generation: Option<u32>,
    recent_topics: &[String],
) -> f64 {
    let base = atom.importance;

    // Stability: base × log(1 + access_count) — LTP analog
    let stability = base * (1.0 + (1.0 + atom.access_count as f64).ln());

    // Pure interference decay (no time-based component)
    let retention = match (current_generation, atom.epoch) {
        (Some(current_gen), Some(atom_gen)) => {
            let atoms_since_access = current_gen.saturating_sub(atom_gen) as f64;

            // Content-dependent interference: same-topic = stronger decay
            let has_topic_overlap = !recent_topics.is_empty()
                && atom.topics.iter().any(|t| {
                    recent_topics
                        .iter()
                        .any(|rt| rt.to_lowercase() == t.to_lowercase())
                });
            let effective_interference = if has_topic_overlap {
                atoms_since_access * TOPIC_INTERFERENCE_MULTIPLIER
            } else {
                atoms_since_access
            };

            let scale = get_atom_scale(atom.atom_type);
            let denom = stability * scale;
            if denom > 0.0 {
                (-effective_interference / denom).exp()
            } else {
                0.0
            }
        }
        _ => 1.0, // No generation data = no decay
    };

    // Apply feedback boost (proportional, not fixed)
    let clamped_boost = clamp(feedback_boost, -MAX_FEEDBACK_BOOST, MAX_FEEDBACK_BOOST);
    let current = base * retention;
    let adjusted = if clamped_boost >= 0.0 {
        current + clamped_boost * (1.0 - current)
    } else {
        current * (1.0 + clamped_boost)
    };

    // Protection floors
    if atom.crystallized {
        return clamp(adjusted, 0.7, 1.0);
    }

    clamp(adjusted, 0.0, 1.0)
}

/// Check if atom is a prune candidate (effective importance below threshold).
pub fn is_prune_candidate(
    atom: &MemoryAtom,
    current_generation: Option<u32>,
    threshold: f64,
) -> bool {
    if atom.crystallized {
        return false;
    }
    let importance = compute_effective_importance(atom, 0.0, current_generation, &[]);
    importance < threshold
}

/// Record an access — updates access count and generation.
///
/// Returns updated fields (caller applies to atom).
pub fn record_access(current_generation: Option<u32>) -> AccessUpdate {
    AccessUpdate {
        access_count_increment: 1,
        generation: current_generation,
    }
}

/// Fields to update after recording access.
pub struct AccessUpdate {
    pub access_count_increment: u32,
    pub generation: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::AtomSource;

    fn make_atom(atom_type: AtomType, importance: f64, access_count: u32) -> MemoryAtom {
        MemoryAtom {
            id: "test".into(),
            content: "test content".into(),
            embedding: None,
            atom_type,
            source: AtomSource::default(),
            importance,
            access_count,
            last_accessed_at: 0.0,
            created_at: 0.0,
            topics: vec!["rust".into(), "async".into()],
            domain: None,
            consolidated_from: None,
            is_consolidated: false,
            parent_id: None,
            depth: None,
            label: None,
            child_ids: None,
            superseded: false,
            suppressed: false,
            dormant: false,
            tags: vec![],
            encoding_context: None,
            retrieval_reward: None,
            reconsolidation_count: None,
            arousal: None,
            valence: None,
            epoch: Some(0),
            crystallized: false,
        }
    }

    #[test]
    fn no_decay_without_generation() {
        let atom = make_atom(AtomType::Episodic, 0.8, 5);
        let eff = compute_effective_importance(&atom, 0.0, None, &[]);
        // No generation data → retention = 1.0 → importance preserved
        assert!(eff > 0.7);
    }

    #[test]
    fn episodic_decays_faster_than_semantic() {
        let episodic = make_atom(AtomType::Episodic, 0.5, 1);
        let semantic = {
            let mut a = make_atom(AtomType::Semantic, 0.5, 1);
            a.atom_type = AtomType::Semantic;
            a
        };

        let eff_ep = compute_effective_importance(&episodic, 0.0, Some(100), &[]);
        let eff_sem = compute_effective_importance(&semantic, 0.0, Some(100), &[]);

        // Episodic scale (60) < Semantic scale (200) → faster decay
        assert!(eff_ep < eff_sem);
    }

    #[test]
    fn topic_interference_accelerates_decay() {
        let atom = make_atom(AtomType::Episodic, 0.5, 1);
        let without = compute_effective_importance(&atom, 0.0, Some(50), &[]);
        let with = compute_effective_importance(
            &atom,
            0.0,
            Some(50),
            &["rust".into()], // Overlaps with atom topics
        );
        assert!(with < without); // Topic interference → faster decay
    }

    #[test]
    fn feedback_boost_positive() {
        let atom = make_atom(AtomType::Episodic, 0.5, 1);
        let base = compute_effective_importance(&atom, 0.0, None, &[]);
        let boosted = compute_effective_importance(&atom, 0.1, None, &[]);
        assert!(boosted > base);
    }

    #[test]
    fn feedback_boost_negative() {
        let atom = make_atom(AtomType::Episodic, 0.5, 1);
        let base = compute_effective_importance(&atom, 0.0, None, &[]);
        let penalized = compute_effective_importance(&atom, -0.1, None, &[]);
        assert!(penalized < base);
    }

    #[test]
    fn crystallized_floor() {
        let mut atom = make_atom(AtomType::Semantic, 0.1, 0);
        atom.crystallized = true;
        let eff = compute_effective_importance(&atom, -0.15, Some(1000), &[]);
        assert!(eff >= 0.7); // Crystallized floor
    }

    #[test]
    fn access_count_increases_stability() {
        let low_access = make_atom(AtomType::Episodic, 0.5, 1);
        let high_access = make_atom(AtomType::Episodic, 0.5, 100);
        let eff_low = compute_effective_importance(&low_access, 0.0, Some(200), &[]);
        let eff_high = compute_effective_importance(&high_access, 0.0, Some(200), &[]);
        assert!(eff_high > eff_low); // More access = more stable
    }

    #[test]
    fn prune_candidate() {
        let mut atom = make_atom(AtomType::Episodic, 0.01, 0);
        atom.epoch = Some(0);
        assert!(is_prune_candidate(&atom, Some(10000), PRUNE_THRESHOLD));
    }

    #[test]
    fn crystallized_never_pruned() {
        let mut atom = make_atom(AtomType::Semantic, 0.01, 0);
        atom.crystallized = true;
        assert!(!is_prune_candidate(&atom, Some(10000), PRUNE_THRESHOLD));
    }
}
