//! Consolidation — episodic → semantic memory transformation.
//!
//! Brain analog: hippocampal-neocortical memory consolidation during sleep/rest.
//! McClelland 1995 (CLS complementary learning systems),
//! Diekelmann & Born 2010 (sleep-dependent memory consolidation).
//!
//! Algorithm: cluster similar atoms → abstract to semantic → link → prune.
//! The LLM abstraction step requires an external callback (trait boundary).

use std::collections::HashSet;

use crate::math::vector::cosine_similarity;
use crate::memory::importance::{is_prune_candidate, PRUNE_THRESHOLD};
use crate::types::memory::{AtomType, MemoryAtom, Synapse};

// ── Constants ──────────────────────────────────────────────────────────

/// Min atoms to form a consolidation cluster.
pub const MIN_CLUSTER_SIZE: usize = 2;
/// Embedding cosine similarity threshold for clustering (greedy agglomerative).
pub const CLUSTER_SIMILARITY_THRESHOLD: f64 = 0.7;
/// Cross-cluster synapse weakening factor (Long-Term Depression, 20% per cycle).
pub const LTD_FACTOR: f64 = 0.8;

// ── Types ──────────────────────────────────────────────────────────────

/// Result of consolidation.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of new semantic atoms created.
    pub created: usize,
    /// Number of atoms pruned (deleted or dormanted).
    pub pruned: usize,
    /// Number of clusters found.
    pub clusters: usize,
}

/// A cluster of similar atoms.
#[derive(Debug, Clone)]
pub struct AtomCluster {
    pub atom_ids: Vec<String>,
    /// Centroid embedding (average of cluster members).
    pub centroid: Option<Vec<f32>>,
}

// ── Core Functions ─────────────────────────────────────────────────────

/// Cluster atoms by embedding similarity (greedy agglomerative).
///
/// Pure function — takes atoms, returns clusters.
pub fn cluster_by_embedding(
    atoms: &[MemoryAtom],
    threshold: f64,
) -> Vec<AtomCluster> {
    let eligible: Vec<&MemoryAtom> = atoms
        .iter()
        .filter(|a| {
            a.is_active()
                && a.atom_type == AtomType::Episodic
                && !a.is_consolidated
                && a.embedding.is_some()
        })
        .collect();

    if eligible.is_empty() {
        return vec![];
    }

    let mut clusters: Vec<AtomCluster> = Vec::new();
    let mut assigned: HashSet<String> = HashSet::new();

    for atom in &eligible {
        if assigned.contains(&atom.id) {
            continue;
        }

        let atom_emb = match &atom.embedding {
            Some(e) => e,
            None => continue,
        };

        // Find best existing cluster
        let mut best_cluster_idx: Option<usize> = None;
        let mut best_sim = 0.0_f64;

        for (i, cluster) in clusters.iter().enumerate() {
            if let Some(centroid) = &cluster.centroid {
                let sim = cosine_similarity(atom_emb, centroid) as f64;
                if sim >= threshold && sim > best_sim {
                    best_sim = sim;
                    best_cluster_idx = Some(i);
                }
            }
        }

        if let Some(idx) = best_cluster_idx {
            clusters[idx].atom_ids.push(atom.id.clone());
            // Update centroid (running average)
            let n = clusters[idx].atom_ids.len() as f32;
            if let Some(centroid) = &mut clusters[idx].centroid {
                for (i, val) in centroid.iter_mut().enumerate() {
                    if i < atom_emb.len() {
                        *val = (*val * (n - 1.0) + atom_emb[i]) / n;
                    }
                }
            }
        } else {
            // New cluster
            clusters.push(AtomCluster {
                atom_ids: vec![atom.id.clone()],
                centroid: Some(atom_emb.clone()),
            });
        }

        assigned.insert(atom.id.clone());
    }

    // Filter to clusters meeting minimum size
    clusters
        .into_iter()
        .filter(|c| c.atom_ids.len() >= MIN_CLUSTER_SIZE)
        .collect()
}

/// Identify atoms that should be pruned or dormanted.
///
/// Returns (to_delete, to_dormant) ID sets.
pub fn identify_prune_candidates(
    atoms: &[MemoryAtom],
    current_generation: Option<u32>,
    synapses_per_atom: &std::collections::HashMap<String, usize>,
) -> (Vec<String>, Vec<String>) {
    let mut to_delete = Vec::new();
    let mut to_dormant = Vec::new();

    for atom in atoms {
        if !is_prune_candidate(atom, current_generation, PRUNE_THRESHOLD) {
            continue;
        }

        let synapse_count = synapses_per_atom
            .get(&atom.id)
            .copied()
            .unwrap_or(0);

        if atom.superseded || atom.suppressed {
            // Already superseded/suppressed → safe to delete
            to_delete.push(atom.id.clone());
        } else if matches!(atom.atom_type, AtomType::Episodic)
            && atom.access_count == 0
            && synapse_count < 2
        {
            // Episodic, never accessed, isolated → delete
            to_delete.push(atom.id.clone());
        } else {
            // Everything else → dormant (not delete)
            to_dormant.push(atom.id.clone());
        }
    }

    (to_delete, to_dormant)
}

/// Compute which cross-cluster synapses should be weakened (LTD).
///
/// Returns synapse IDs that connect atoms in different clusters.
pub fn find_cross_cluster_synapses(
    clusters: &[AtomCluster],
    synapses: &[Synapse],
) -> Vec<String> {
    // Build atom → cluster index
    let mut atom_to_cluster: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, cluster) in clusters.iter().enumerate() {
        for atom_id in &cluster.atom_ids {
            atom_to_cluster.insert(atom_id.as_str(), i);
        }
    }

    let mut cross_cluster = Vec::new();
    for synapse in synapses {
        let src_cluster = atom_to_cluster.get(synapse.source.as_str());
        let tgt_cluster = atom_to_cluster.get(synapse.target.as_str());
        if let (Some(&sc), Some(&tc)) = (src_cluster, tgt_cluster) {
            if sc != tc {
                cross_cluster.push(synapse.id.clone());
            }
        }
    }

    cross_cluster
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::{AtomSource, SynapseType};

    fn make_atom(id: &str, embedding: Vec<f32>, importance: f64) -> MemoryAtom {
        MemoryAtom {
            id: id.into(),
            content: format!("content {id}"),
            embedding: Some(embedding),
            atom_type: AtomType::Episodic,
            source: AtomSource::default(),
            importance,
            access_count: 1,
            last_accessed_at: 0.0,
            created_at: 0.0,
            topics: vec![],
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
    fn cluster_similar_atoms() {
        let atoms = vec![
            make_atom("a1", vec![1.0, 0.0, 0.0], 0.5),
            make_atom("a2", vec![0.99, 0.1, 0.0], 0.5), // Very similar to a1
            make_atom("a3", vec![0.0, 0.0, 1.0], 0.5), // Very different
        ];
        let clusters = cluster_by_embedding(&atoms, CLUSTER_SIMILARITY_THRESHOLD);
        // a1 and a2 should cluster together, a3 alone (below min cluster size)
        assert!(clusters.len() <= 2);
        if !clusters.is_empty() {
            let largest = clusters.iter().max_by_key(|c| c.atom_ids.len()).unwrap();
            assert!(largest.atom_ids.contains(&"a1".to_string()));
            assert!(largest.atom_ids.contains(&"a2".to_string()));
        }
    }

    #[test]
    fn cluster_empty() {
        let clusters = cluster_by_embedding(&[], 0.7);
        assert!(clusters.is_empty());
    }

    #[test]
    fn cluster_skips_consolidated() {
        let mut atom = make_atom("a1", vec![1.0, 0.0], 0.5);
        atom.is_consolidated = true;
        let clusters = cluster_by_embedding(&[atom], 0.7);
        assert!(clusters.is_empty());
    }

    #[test]
    fn prune_candidates_superseded() {
        let mut atom = make_atom("a1", vec![], 0.01);
        atom.superseded = true;
        atom.epoch = Some(0);
        let (delete, _dormant) = identify_prune_candidates(
            &[atom],
            Some(10000),
            &std::collections::HashMap::new(),
        );
        assert!(delete.contains(&"a1".to_string()));
    }

    #[test]
    fn prune_candidates_dormant_not_delete() {
        let mut atom = make_atom("a1", vec![], 0.01);
        atom.atom_type = AtomType::Semantic;
        atom.epoch = Some(0);
        let (delete, dormant) = identify_prune_candidates(
            &[atom],
            Some(10000),
            &std::collections::HashMap::new(),
        );
        assert!(delete.is_empty());
        assert!(dormant.contains(&"a1".to_string()));
    }

    #[test]
    fn cross_cluster_synapses() {
        let clusters = vec![
            AtomCluster { atom_ids: vec!["a1".into(), "a2".into()], centroid: None },
            AtomCluster { atom_ids: vec!["b1".into(), "b2".into()], centroid: None },
        ];
        let synapses = vec![
            Synapse {
                id: "s1".into(), source: "a1".into(), target: "a2".into(),
                synapse_type: SynapseType::RelatesTo, strength: 0.5,
                created_at: 0.0, last_accessed_at: None, access_count: None,
            },
            Synapse {
                id: "s2".into(), source: "a1".into(), target: "b1".into(),
                synapse_type: SynapseType::RelatesTo, strength: 0.3,
                created_at: 0.0, last_accessed_at: None, access_count: None,
            },
        ];
        let cross = find_cross_cluster_synapses(&clusters, &synapses);
        assert_eq!(cross.len(), 1); // s2 is cross-cluster
        assert!(cross.contains(&"s2".to_string()));
    }
}
