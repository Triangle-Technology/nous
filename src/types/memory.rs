//! Memory types — MemoryAtom, Synapse, and associated structures.
//!
//! Brain analog: hippocampal memory system with episodic-semantic distinction.

use serde::{Deserialize, Serialize};

/// Type of memory atom — episodic vs semantic vs procedural.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AtomType {
    #[default]
    Episodic,
    Semantic,
    Procedural,
    Preference,
    Digest,
    PromptEvolution,
}

/// Synapse type — relationship between atoms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynapseType {
    RelatesTo,
    CausedBy,
    Contradicts,
    EvolvedInto,
    Supports,
    PartOf,
    UsedIn,
    Hebbian,
    Temporal,
}

/// Source provenance for a memory atom.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AtomSource {
    pub conversation_id: Option<String>,
    pub message_id: Option<String>,
    pub composition_id: Option<String>,
    pub step: Option<String>,
}

/// Encoding context at time of storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncodingContext {
    pub goal: Option<String>,
    pub phase: Option<String>,
    pub gain_mode: Option<String>,
}

/// Fundamental memory unit — an atom of knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAtom {
    pub id: String,
    pub content: String,
    /// Embedding vector (optional — requires embedding provider).
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    pub atom_type: AtomType,

    pub source: AtomSource,

    /// 0-1, subjective importance.
    pub importance: f64,
    pub access_count: u32,
    /// Unix timestamp (seconds).
    pub last_accessed_at: f64,
    /// Unix timestamp (seconds).
    pub created_at: f64,

    pub topics: Vec<String>,
    pub domain: Option<String>,

    /// IDs of atoms this was consolidated from.
    pub consolidated_from: Option<Vec<String>>,
    pub is_consolidated: bool,

    // ── Hierarchy (CÂY layer) ──
    pub parent_id: Option<String>,
    pub depth: Option<u32>,
    pub label: Option<String>,
    pub child_ids: Option<Vec<String>>,

    // ── Lifecycle flags ──
    pub superseded: bool,
    pub suppressed: bool,
    pub dormant: bool,
    pub tags: Vec<String>,

    pub encoding_context: Option<EncodingContext>,

    /// 0-1, learned retrieval value.
    pub retrieval_reward: Option<f64>,
    pub reconsolidation_count: Option<u32>,

    // ── Emotional encoding ──
    pub arousal: Option<f64>,
    pub valence: Option<String>,

    /// Neurogenesis epoch for temporal disambiguation.
    pub epoch: Option<u32>,
    /// Whether crystallized (stable, rarely updated).
    pub crystallized: bool,
}

impl MemoryAtom {
    /// Active atom predicate (P3: shared across all consumers).
    ///
    /// An atom is active if it's not consolidated, superseded, suppressed, or dormant.
    pub fn is_active(&self) -> bool {
        !self.is_consolidated && !self.superseded && !self.suppressed && !self.dormant
    }
}

/// Association between two atoms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub id: String,
    /// Source atom ID.
    pub source: String,
    /// Target atom ID.
    pub target: String,
    pub synapse_type: SynapseType,
    /// 0-1, increases with co-activation.
    pub strength: f64,
    /// Unix timestamp (seconds).
    pub created_at: f64,
    pub last_accessed_at: Option<f64>,
    pub access_count: Option<u32>,
}
