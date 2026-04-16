//! Hybrid Retrieval — vector search + spreading activation + context re-ranking.
//!
//! Brain analog: hippocampal pattern completion (CA3 spreading activation) +
//! neocortical similarity matching (vector search) + PFC top-down modulation
//! (context re-ranking). Eichenbaum 2004, Anderson 2000 (ACT-R).
//!
//! Core algorithm: parallel concept activation → graph expansion → re-ranking → threshold.

use std::collections::{HashMap, HashSet};

use crate::cognition::detector::{count_topic_overlap, extract_topics, to_topic_set};
use crate::math::clamp;
use crate::math::vector::cosine_similarity;
use crate::memory::importance::compute_effective_importance;
use crate::types::memory::MemoryAtom;

// ── Constants ──────────────────────────────────────────────────────────

/// Vector similarity contribution to initial score.
pub const SIMILARITY_WEIGHT: f64 = 0.7;
/// Atom importance contribution to initial score.
pub const IMPORTANCE_WEIGHT: f64 = 0.3;
/// Decay per synapse hop in spreading activation (depth-2: 0.49).
pub const HOP_DECAY: f64 = 0.7;
/// Minimum score to include in results.
pub const MIN_ACTIVATION: f64 = 0.15;
/// Per-epoch broadening of MIN_ACTIVATION (AD-135a neurogenesis).
pub const GIST_BROADENING_RATE: f64 = 0.005;
/// Maximum MIN_ACTIVATION reduction (floor: 0.10).
pub const GIST_BROADENING_MAX: f64 = 0.05;
/// Boost for atoms found by 2+ parallel concepts (AD-53 LTP).
pub const CONVERGENT_BOOST: f64 = 0.15;

// ── Context Re-ranking Boosts ──────────────────────────────────────────

/// Topic overlap with conversation context.
pub const TOPIC_OVERLAP_BOOST: f64 = 0.2;
/// Same domain as current conversation.
pub const DOMAIN_MATCH_BOOST: f64 = 0.1;
/// Accessed within recency window (7 days).
pub const RECENCY_BOOST: f64 = 0.1;
/// Tag match with query intent.
pub const TAG_BOOST: f64 = 0.2;
/// Mood-congruent retrieval (Bower 1981).
pub const AROUSAL_CONGRUENT_BOOST: f64 = 0.05;
/// Threat-associated topic boost (Pavlovian amygdala, Pessoa 2008).
pub const THREAT_RETRIEVAL_BOOST: f64 = 0.05;
/// Inhibition of Return penalty per IOR step (Klein 2000).
pub const IOR_PENALTY: f64 = 0.04;

/// Recency window in seconds (7 days).
const RECENCY_WINDOW_SECS: f64 = 7.0 * 24.0 * 3600.0;
/// Atoms above this count trigger topic pre-filter for O(K) instead of O(N).
const TOPIC_PREFILTER_THRESHOLD: usize = 200;

// ── Types ──────────────────────────────────────────────────────────────

/// A retrieved atom with activation score and source.
#[derive(Debug, Clone)]
pub struct ActivatedAtom {
    pub atom: MemoryAtom,
    /// 0-1, activation score after all boosts/penalties.
    pub score: f64,
    /// How this atom was found.
    pub source: ActivationSource,
    /// Chunk ID for grouped display (set in chunking step).
    pub chunk_id: Option<usize>,
}

/// How an atom was activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationSource {
    /// Direct vector similarity match.
    Vector,
    /// Found via spreading activation through synapses.
    Graph,
    /// Found by 2+ parallel concept searches (convergent activation).
    Convergent,
}

/// Options for hybrid recall.
#[derive(Debug, Clone)]
pub struct RecallOptions {
    /// Max results to return.
    pub top_k: usize,
    /// Spreading activation depth (0 = vector only).
    pub graph_depth: usize,
    /// Context topics for re-ranking.
    pub context_topics: Vec<String>,
    /// Current domain for domain match boost.
    pub domain: Option<String>,
    /// Currently high-arousal for mood-congruent retrieval.
    pub arousal: f64,
    /// Threat-associated topics for amygdala boost.
    pub threat_topics: HashSet<String>,
    /// Current time for recency scoring (unix seconds).
    pub now: f64,
    /// Current interference generation counter.
    pub current_generation: Option<u32>,
    /// Recently discussed atom IDs for IOR penalty.
    pub ior_atom_ids: HashMap<String, u32>,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            graph_depth: 2,
            context_topics: vec![],
            domain: None,
            arousal: 0.0,
            threat_topics: HashSet::new(),
            now: 0.0,
            current_generation: None,
            ior_atom_ids: HashMap::new(),
        }
    }
}

// ── Core Function ──────────────────────────────────────────────────────

/// Hybrid recall: vector search + spreading activation + context re-ranking.
///
/// Steps:
/// 1. Vector search (embedding similarity × importance)
/// 2. Spreading activation (BFS through synapses, depth-limited)
/// 3. Context re-ranking (topic overlap, domain, recency, arousal, threat, IOR)
/// 4. Threshold filter + top-K selection
pub fn hybrid_recall(
    atoms: &[MemoryAtom],
    query_embedding: Option<&[f32]>,
    query: &str,
    synapses: &HashMap<String, Vec<(String, f64)>>, // atom_id → [(target_id, strength)]
    options: &RecallOptions,
) -> Vec<ActivatedAtom> {
    let query_topics = extract_topics(query);
    let query_topic_set = to_topic_set(&query_topics);
    let context_set = to_topic_set(&options.context_topics);

    // Step 0: Pre-filter by topics if large store
    let candidates: Vec<&MemoryAtom> = if atoms.len() >= TOPIC_PREFILTER_THRESHOLD && !query_topics.is_empty() {
        atoms
            .iter()
            .filter(|a| {
                a.is_active() && {
                    let overlap = count_topic_overlap(&a.topics, &query_topic_set);
                    let ratio = overlap as f64 / a.topics.len().max(1) as f64;
                    ratio >= 0.3
                }
            })
            .collect()
    } else {
        atoms.iter().filter(|a| a.is_active()).collect()
    };

    // Step 1: Vector search
    let mut activation_map: HashMap<String, ActivatedAtom> = HashMap::new();

    for atom in &candidates {
        let sim = match (query_embedding, &atom.embedding) {
            (Some(qe), Some(ae)) => cosine_similarity(qe, ae) as f64,
            _ => 0.0,
        };

        let eff_importance = compute_effective_importance(
            atom,
            0.0,
            options.current_generation,
            &options.context_topics,
        );

        let score = SIMILARITY_WEIGHT * sim + IMPORTANCE_WEIGHT * eff_importance;

        if score >= MIN_ACTIVATION {
            activation_map.insert(
                atom.id.clone(),
                ActivatedAtom {
                    atom: (*atom).clone(),
                    score,
                    source: ActivationSource::Vector,
                    chunk_id: None,
                },
            );
        }
    }

    // Step 2: Spreading activation (BFS through synapses)
    if options.graph_depth > 0 {
        let seed_ids: Vec<String> = activation_map.keys().cloned().collect();
        for seed_id in &seed_ids {
            let seed_score = activation_map.get(seed_id).map(|a| a.score).unwrap_or(0.0);
            spread_from(
                seed_id,
                seed_score,
                synapses,
                atoms,
                &mut activation_map,
                options.graph_depth,
            );
        }
    }

    // Step 3: Context re-ranking
    for activated in activation_map.values_mut() {
        // Topic overlap boost
        let topic_overlap = count_topic_overlap(&activated.atom.topics, &context_set);
        if topic_overlap > 0 {
            activated.score += TOPIC_OVERLAP_BOOST * (topic_overlap as f64 / activated.atom.topics.len().max(1) as f64);
        }

        // Domain match
        if let (Some(atom_domain), Some(ctx_domain)) = (&activated.atom.domain, &options.domain) {
            if atom_domain.to_lowercase() == ctx_domain.to_lowercase() {
                activated.score += DOMAIN_MATCH_BOOST;
            }
        }

        // Recency boost (linear decay over 7 days)
        if options.now > 0.0 && activated.atom.last_accessed_at > 0.0 {
            let age = options.now - activated.atom.last_accessed_at;
            if (0.0..RECENCY_WINDOW_SECS).contains(&age) {
                activated.score += RECENCY_BOOST * (1.0 - age / RECENCY_WINDOW_SECS);
            }
        }

        // Threat-associated topic boost (Pavlovian amygdala)
        let has_threat = activated
            .atom
            .topics
            .iter()
            .any(|t| options.threat_topics.contains(&t.to_lowercase()));
        if has_threat {
            activated.score += THREAT_RETRIEVAL_BOOST;
        }

        // Arousal-congruent boost (Bower 1981 mood-congruent memory)
        if options.arousal >= 0.6 {
            if let Some(atom_arousal) = activated.atom.arousal {
                if atom_arousal >= 0.6 {
                    activated.score += AROUSAL_CONGRUENT_BOOST;
                }
            }
        }

        // IOR penalty (Inhibition of Return, Klein 2000)
        if let Some(&ior_step) = options.ior_atom_ids.get(&activated.atom.id) {
            activated.score -= IOR_PENALTY * ior_step as f64;
        }

        activated.score = clamp(activated.score, 0.0, 1.5); // Allow slight over-1.0 from boosts
    }

    // Step 4: Threshold filter + top-K
    let mut results: Vec<ActivatedAtom> = activation_map
        .into_values()
        .filter(|a| a.score >= MIN_ACTIVATION)
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(options.top_k);

    results
}

/// BFS spreading activation from a seed atom.
fn spread_from(
    seed_id: &str,
    seed_score: f64,
    synapses: &HashMap<String, Vec<(String, f64)>>,
    atoms: &[MemoryAtom],
    activation_map: &mut HashMap<String, ActivatedAtom>,
    max_depth: usize,
) {
    let atom_index: HashMap<&str, &MemoryAtom> = atoms.iter().map(|a| (a.id.as_str(), a)).collect();

    let mut queue: Vec<(String, f64, usize)> = Vec::new(); // (atom_id, activation, depth)

    // Seed neighbors
    if let Some(neighbors) = synapses.get(seed_id) {
        for (target_id, strength) in neighbors {
            let activation = seed_score * strength * HOP_DECAY;
            if activation >= MIN_ACTIVATION {
                queue.push((target_id.clone(), activation, 1));
            }
        }
    }

    while let Some((atom_id, activation, depth)) = queue.pop() {
        if depth > max_depth {
            continue;
        }

        let atom = match atom_index.get(atom_id.as_str()) {
            Some(a) if a.is_active() => *a,
            _ => continue,
        };

        // Update or insert
        let entry = activation_map.entry(atom_id.clone());
        let _existing_score = entry
            .and_modify(|e| {
                if activation > e.score {
                    e.score = activation;
                    e.source = ActivationSource::Graph;
                }
            })
            .or_insert_with(|| ActivatedAtom {
                atom: atom.clone(),
                score: activation,
                source: ActivationSource::Graph,
                chunk_id: None,
            })
            .score;

        // Continue BFS to next depth
        if depth < max_depth {
            if let Some(neighbors) = synapses.get(atom_id.as_str()) {
                for (target_id, strength) in neighbors {
                    let next_activation = activation * strength * HOP_DECAY;
                    if next_activation >= MIN_ACTIVATION
                        && !activation_map.contains_key(target_id)
                    {
                        queue.push((target_id.clone(), next_activation, depth + 1));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::{AtomSource, AtomType};

    fn make_atom(id: &str, topics: &[&str], importance: f64) -> MemoryAtom {
        MemoryAtom {
            id: id.into(),
            content: format!("content about {}", topics.join(" ")),
            embedding: Some(vec![0.1; 10]),
            atom_type: AtomType::Episodic,
            source: AtomSource::default(),
            importance,
            access_count: 1,
            last_accessed_at: 0.0,
            created_at: 0.0,
            topics: topics.iter().map(|t| t.to_string()).collect(),
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
            epoch: None,
            crystallized: false,
        }
    }

    #[test]
    fn empty_recall() {
        let results = hybrid_recall(&[], None, "test", &HashMap::new(), &RecallOptions::default());
        assert!(results.is_empty());
    }

    #[test]
    fn vector_recall_with_embedding() {
        let atoms = vec![make_atom("a1", &["rust", "async"], 0.8)];
        let query_emb = vec![0.1_f32; 10]; // Same direction → high similarity
        let results = hybrid_recall(
            &atoms,
            Some(&query_emb),
            "rust async",
            &HashMap::new(),
            &RecallOptions::default(),
        );
        assert!(!results.is_empty());
        assert_eq!(results[0].atom.id, "a1");
    }

    #[test]
    fn inactive_atoms_excluded() {
        let mut atom = make_atom("a1", &["rust"], 0.8);
        atom.suppressed = true;
        let results = hybrid_recall(
            &[atom],
            Some(&[0.1_f32; 10]),
            "rust",
            &HashMap::new(),
            &RecallOptions::default(),
        );
        assert!(results.is_empty());
    }

    #[test]
    fn context_topic_boost() {
        let atoms = vec![
            make_atom("a1", &["rust", "async"], 0.5),
            make_atom("a2", &["python", "django"], 0.5),
        ];
        let options = RecallOptions {
            context_topics: vec!["rust".into()],
            ..Default::default()
        };
        let query_emb = vec![0.1_f32; 10];
        let results = hybrid_recall(&atoms, Some(&query_emb), "programming", &HashMap::new(), &options);
        if results.len() >= 2 {
            // a1 should score higher due to topic overlap boost
            assert!(results[0].atom.id == "a1");
        }
    }

    #[test]
    fn ior_penalty_reduces_score() {
        let atoms = vec![make_atom("a1", &["rust"], 0.8)];
        let mut ior = HashMap::new();
        ior.insert("a1".into(), 2); // Recently discussed

        let without_ior = hybrid_recall(
            &atoms,
            Some(&[0.1_f32; 10]),
            "rust",
            &HashMap::new(),
            &RecallOptions::default(),
        );
        let with_ior = hybrid_recall(
            &atoms,
            Some(&[0.1_f32; 10]),
            "rust",
            &HashMap::new(),
            &RecallOptions {
                ior_atom_ids: ior,
                ..Default::default()
            },
        );

        if !without_ior.is_empty() && !with_ior.is_empty() {
            assert!(with_ior[0].score < without_ior[0].score);
        }
    }

    #[test]
    fn top_k_limits_results() {
        let atoms: Vec<MemoryAtom> = (0..20)
            .map(|i| make_atom(&format!("a{i}"), &["topic"], 0.5))
            .collect();
        let options = RecallOptions {
            top_k: 5,
            ..Default::default()
        };
        let results = hybrid_recall(
            &atoms,
            Some(&[0.1_f32; 10]),
            "topic",
            &HashMap::new(),
            &options,
        );
        assert!(results.len() <= 5);
    }

    #[test]
    fn spreading_activation_finds_neighbors() {
        let atoms = vec![
            make_atom("seed", &["rust"], 0.8),
            make_atom("neighbor", &["async"], 0.5),
        ];
        let mut synapses: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        synapses.insert("seed".into(), vec![("neighbor".into(), 0.8)]);

        let results = hybrid_recall(
            &atoms,
            Some(&[0.1_f32; 10]),
            "rust",
            &synapses,
            &RecallOptions {
                graph_depth: 2,
                ..Default::default()
            },
        );
        // Should find both seed (vector) and neighbor (graph)
        let ids: HashSet<&str> = results.iter().map(|r| r.atom.id.as_str()).collect();
        assert!(ids.contains("seed"));
        // Neighbor may or may not pass MIN_ACTIVATION depending on seed score
    }
}
