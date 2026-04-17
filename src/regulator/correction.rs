//! Correction store — per-cluster record of user corrections + opaque
//! structural pattern extraction.
//!
//! **Scope note (P1 / P9b)**: I/O adapter sub-module, not cognitive.
//! Pattern extraction is **structural** (cluster-based count
//! threshold), not semantic (no English-specific regex on correction
//! text). Pattern names are opaque identifiers (`corrections_on_{cluster}`)
//! — the app / LLM interprets the meaning by reading the stored
//! example texts. This is the P9b-compliant alternative to regex-based
//! rule extraction ("don't add X" → `no_new_X`), which would embed a
//! language assumption into Noos.
//!
//! ## Design: content-vs-pattern boundary
//!
//! Competitors (Mem0, Letta) store correction CONTENT and retrieve
//! via semantic search. Noos stores STRUCTURAL metadata: which
//! (user, topic) pair has received repeated corrections, how many, with
//! a bounded list of recent example texts. The app / LLM does the
//! rule interpretation by reading examples at retrieval time.
//!
//! The differentiation holds: a [`CorrectionPattern`]
//! tells the app "this cluster has received N corrections, here are
//! examples — consider them before generating." Competitors' memory
//! layers require semantic search on every relevant turn; Noos surfaces
//! the pattern prospectively via [`Decision::ProceduralWarning`](super::Decision::ProceduralWarning)
//! once the count threshold trips.
//!
//! ## Cluster-identity alignment (P3)
//!
//! Corrections are keyed by the SAME cluster hash
//! [`detector::build_topic_cluster`](crate::cognition::detector::build_topic_cluster)
//! produces for `LearnedState.response_strategies`. That alignment
//! means strategy learning + correction patterns refer to the same
//! notion of "topic"; there are never two disagreeing cluster IDs for
//! the same turn.
//!
//! ## Gating (P10)
//!
//! This module produces
//! [`Decision::ProceduralWarning`](super::Decision::ProceduralWarning)
//! via [`Regulator::decide`](super::Regulator::decide).
//!
//! - **Suppresses**:
//!   [`Decision::Continue`](super::Decision::Continue) only.
//!   ProceduralWarning is an advisory signal that fires *before*
//!   generation — apps can read the `example_corrections` and adjust
//!   the upcoming prompt.
//! - **Suppressed by**: every
//!   [`Decision::CircuitBreak`](super::Decision::CircuitBreak) variant
//!   AND [`Decision::ScopeDriftWarn`](super::Decision::ScopeDriftWarn).
//!   Procedural patterns are historical context; a live cost / quality
//!   / tool-loop / scope problem takes precedence.
//! - **Inactive when**:
//!   [`Regulator.current_topic_cluster`](super::Regulator) is empty
//!   (no [`LLMEvent::TurnStart`](super::LLMEvent::TurnStart) yet, or
//!   the user message had no extractable top-2 topics), OR the
//!   current cluster has fewer than
//!   [`MIN_CORRECTIONS_FOR_PATTERN`] recorded corrections.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::CorrectionPattern;

// ── Constants ──────────────────────────────────────────────────────────

/// Minimum number of corrections on a (user, cluster) pair before a
/// [`CorrectionPattern`] is surfaced via [`CorrectionStore::pattern_for`].
///
/// 3 matches the plan test target ("3 similar corrections in different
/// words → single `CorrectionPattern` extracted"). Small enough to
/// respond within a task; large enough that a single misfire or
/// language-nuance correction doesn't immediately generate a standing
/// rule.
pub const MIN_CORRECTIONS_FOR_PATTERN: usize = 3;

/// Maximum number of recent correction texts retained per cluster as
/// `example_corrections` on the extracted pattern.
///
/// 3 matches the extraction threshold — any pattern that fires has
/// exactly as many examples as the count it was extracted from (until
/// more corrections accumulate, at which point the oldest example
/// drops). Bounds snapshot size at ~3 × average correction length per
/// cluster.
pub const MAX_EXAMPLE_CORRECTIONS: usize = 3;

/// Maximum number of correction records kept per cluster (for the
/// `learned_from_turns` count and to bound memory). Above this,
/// oldest entries drop.
///
/// 20 = 20 corrections per (user, cluster) is ample headroom for any
/// real agent task; beyond this the pattern has been "learned" and
/// additional individual corrections don't materially change the
/// confidence signal.
pub const MAX_CORRECTIONS_PER_CLUSTER: usize = 20;

// ── CorrectionStore ────────────────────────────────────────────────────

/// Runtime working memory for user corrections, keyed by topic cluster.
///
/// Lifecycle: persists across turns for the lifetime of the
/// [`Regulator`](super::Regulator) (like
/// [`CostAccumulator`](super::cost::CostAccumulator); unlike
/// [`TokenStatsAccumulator`](super::token_stats::TokenStatsAccumulator)
/// and [`ScopeTracker`](super::scope::ScopeTracker), which are
/// per-turn). Exported / imported via
/// [`RegulatorState`](super::RegulatorState) so patterns survive
/// process restarts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionStore {
    /// Per-cluster list of correction texts (newest last). Each entry
    /// is the raw text the user sent — no regex, no lexicon parsing.
    records: HashMap<String, Vec<String>>,
}

impl CorrectionStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mutable: record one user correction against a topic cluster.
    /// Requires mutation because the store accumulates records per
    /// cluster.
    ///
    /// No-ops (silently drops) corrections with empty `cluster` — an
    /// empty cluster means `build_topic_cluster` couldn't identify a
    /// topic, and attributing the correction to "no cluster" would
    /// poison the any-cluster pattern extraction path.
    ///
    /// Evicts oldest records per cluster past
    /// [`MAX_CORRECTIONS_PER_CLUSTER`] to bound memory.
    pub fn record_correction(&mut self, cluster: &str, correction_text: impl Into<String>) {
        if cluster.is_empty() {
            return;
        }
        let list = self.records.entry(cluster.to_string()).or_default();
        list.push(correction_text.into());
        if list.len() > MAX_CORRECTIONS_PER_CLUSTER {
            list.remove(0);
        }
    }

    /// Number of corrections recorded on this cluster.
    pub fn count_for(&self, cluster: &str) -> usize {
        self.records.get(cluster).map(|v| v.len()).unwrap_or(0)
    }

    /// Extract a [`CorrectionPattern`] for `(user_id, cluster)` if the
    /// correction count meets [`MIN_CORRECTIONS_FOR_PATTERN`]. Returns
    /// `None` when the threshold isn't reached.
    ///
    /// Pattern fields:
    /// - `pattern_name` is opaque (`corrections_on_{cluster}`) — no
    ///   language assumption.
    /// - `learned_from_turns` = count of stored corrections.
    /// - `confidence` = `count / MAX_CORRECTIONS_PER_CLUSTER`,
    ///   saturating at 1.0. This is a simple monotonic signal — more
    ///   corrections → higher confidence that the rule is real — rather
    ///   than a probabilistic calibration.
    /// - `example_corrections` = up to [`MAX_EXAMPLE_CORRECTIONS`]
    ///   most-recent texts, newest first, for the app / LLM to
    ///   interpret.
    pub fn pattern_for(&self, user_id: &str, cluster: &str) -> Option<CorrectionPattern> {
        let list = self.records.get(cluster)?;
        if list.len() < MIN_CORRECTIONS_FOR_PATTERN {
            return None;
        }
        let example_corrections: Vec<String> = list
            .iter()
            .rev()
            .take(MAX_EXAMPLE_CORRECTIONS)
            .cloned()
            .collect();
        let confidence =
            (list.len() as f64 / MAX_CORRECTIONS_PER_CLUSTER as f64).clamp(0.0, 1.0);
        Some(CorrectionPattern {
            user_id: user_id.to_string(),
            topic_cluster: cluster.to_string(),
            pattern_name: format!("corrections_on_{cluster}"),
            learned_from_turns: list.len(),
            confidence,
            example_corrections,
        })
    }

    /// All patterns currently known for `user_id` across every cluster.
    /// Filtered to clusters meeting the extraction threshold.
    ///
    /// Returned in deterministic alphabetical order by cluster key so
    /// snapshot round-trips and test assertions are stable.
    pub fn all_patterns(&self, user_id: &str) -> Vec<CorrectionPattern> {
        let mut clusters: Vec<&String> = self.records.keys().collect();
        clusters.sort();
        clusters
            .into_iter()
            .filter_map(|c| self.pattern_for(user_id, c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store_has_no_patterns() {
        let store = CorrectionStore::new();
        assert_eq!(store.count_for("any"), 0);
        assert!(store.pattern_for("u1", "any").is_none());
        assert!(store.all_patterns("u1").is_empty());
    }

    #[test]
    fn record_correction_accumulates_count() {
        let mut store = CorrectionStore::new();
        store.record_correction("refactor+async", "don't add logging");
        store.record_correction("refactor+async", "stop adding logging");
        assert_eq!(store.count_for("refactor+async"), 2);
        assert_eq!(store.count_for("other+cluster"), 0);
    }

    #[test]
    fn empty_cluster_drops_correction() {
        // Contract: empty cluster (build_topic_cluster couldn't
        // identify a topic) must not silently be stored under a sentinel
        // key — it'd pollute the any-pattern path.
        let mut store = CorrectionStore::new();
        store.record_correction("", "some correction text");
        assert_eq!(store.count_for(""), 0);
        assert!(store.all_patterns("u1").is_empty());
    }

    #[test]
    fn pattern_below_threshold_is_none() {
        // Two corrections on one cluster — MIN threshold is 3.
        let mut store = CorrectionStore::new();
        store.record_correction("cluster_a", "first correction");
        store.record_correction("cluster_a", "second correction");
        assert!(store.pattern_for("u1", "cluster_a").is_none());
    }

    #[test]
    fn pattern_at_threshold_emerges() {
        // Plan test target: 3 similar corrections in different words →
        // single CorrectionPattern extracted.
        let mut store = CorrectionStore::new();
        store.record_correction("refactor+async", "don't add logging");
        store.record_correction("refactor+async", "stop adding logging please");
        store.record_correction("refactor+async", "no more logs");
        let pattern = store
            .pattern_for("user_42", "refactor+async")
            .expect("three corrections on same cluster must emerge as pattern");
        assert_eq!(pattern.user_id, "user_42");
        assert_eq!(pattern.topic_cluster, "refactor+async");
        assert_eq!(pattern.pattern_name, "corrections_on_refactor+async");
        assert_eq!(pattern.learned_from_turns, 3);
        assert!((pattern.confidence - 3.0 / MAX_CORRECTIONS_PER_CLUSTER as f64).abs() < 1e-9);
        assert_eq!(pattern.example_corrections.len(), 3);
        // Newest first.
        assert_eq!(pattern.example_corrections[0], "no more logs");
    }

    #[test]
    fn example_corrections_capped_at_max() {
        let mut store = CorrectionStore::new();
        for i in 0..10 {
            store.record_correction("cluster", format!("correction {i}"));
        }
        let pattern = store
            .pattern_for("u1", "cluster")
            .expect("10 corrections exceed threshold");
        // Bounded list with newest first.
        assert_eq!(pattern.example_corrections.len(), MAX_EXAMPLE_CORRECTIONS);
        assert_eq!(pattern.example_corrections[0], "correction 9");
    }

    #[test]
    fn confidence_saturates_at_max_records() {
        let mut store = CorrectionStore::new();
        // Fill past MAX_CORRECTIONS_PER_CLUSTER to confirm saturation
        // at confidence = 1.0.
        for i in 0..(MAX_CORRECTIONS_PER_CLUSTER + 5) {
            store.record_correction("cluster", format!("correction {i}"));
        }
        let pattern = store.pattern_for("u1", "cluster").expect("above threshold");
        assert!((pattern.confidence - 1.0).abs() < 1e-9);
        // learned_from_turns never exceeds MAX_CORRECTIONS_PER_CLUSTER
        // because oldest are evicted to bound memory.
        assert_eq!(pattern.learned_from_turns, MAX_CORRECTIONS_PER_CLUSTER);
    }

    #[test]
    fn all_patterns_returns_sorted_stable() {
        let mut store = CorrectionStore::new();
        // Populate multiple clusters above threshold, in arbitrary
        // order.
        for cluster in ["zulu", "alpha", "mike"] {
            for i in 0..3 {
                store.record_correction(cluster, format!("{cluster} correction {i}"));
            }
        }
        // One cluster below threshold — must be filtered out.
        store.record_correction("below", "only once");

        let patterns = store.all_patterns("u1");
        let clusters: Vec<&str> = patterns.iter().map(|p| p.topic_cluster.as_str()).collect();
        assert_eq!(clusters, vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn records_per_cluster_bounded() {
        // Contract: no cluster can grow beyond MAX_CORRECTIONS_PER_CLUSTER
        // records (oldest dropped). Confirmed via count + confidence
        // saturation.
        let mut store = CorrectionStore::new();
        for i in 0..(MAX_CORRECTIONS_PER_CLUSTER * 2) {
            store.record_correction("cluster", format!("correction {i}"));
        }
        assert_eq!(store.count_for("cluster"), MAX_CORRECTIONS_PER_CLUSTER);
        // The newest example should reflect the most-recent push.
        let pattern = store.pattern_for("u1", "cluster").unwrap();
        assert_eq!(
            pattern.example_corrections[0],
            format!("correction {}", MAX_CORRECTIONS_PER_CLUSTER * 2 - 1)
        );
    }

    #[test]
    fn round_trip_via_serde_json() {
        // Contract: CorrectionStore snapshots survive JSON round-trip
        // for RegulatorState persistence.
        let mut store = CorrectionStore::new();
        store.record_correction("cluster_a", "c1");
        store.record_correction("cluster_a", "c2");
        store.record_correction("cluster_a", "c3");

        let json = serde_json::to_string(&store).expect("serialise");
        let restored: CorrectionStore =
            serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.count_for("cluster_a"), 3);
        assert!(restored.pattern_for("u1", "cluster_a").is_some());
    }
}
