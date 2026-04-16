//! MemoryStore trait + in-memory implementation (P4 trait boundary).
//!
//! Brain analog: hippocampal memory system — fast storage and retrieval
//! with associative linking (Eichenbaum 2004).
//!
//! The cognitive engine uses MemoryStore trait, never a concrete backend.
//! Swap IndexedDB / SQLite / Redis / custom without changing cognitive logic.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::errors::NousResult;
use crate::types::memory::{MemoryAtom, Synapse};

/// Partial update for a memory atom (only set fields that changed).
#[derive(Debug, Clone, Default)]
pub struct AtomUpdate {
    pub content: Option<String>,
    pub importance: Option<f64>,
    pub access_count: Option<u32>,
    pub last_accessed_at: Option<f64>,
    pub is_consolidated: Option<bool>,
    pub superseded: Option<bool>,
    pub suppressed: Option<bool>,
    pub dormant: Option<bool>,
}

/// Abstract memory storage — implement for your persistence backend.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn store_atom(&mut self, atom: MemoryAtom) -> NousResult<()>;
    async fn get_atom(&self, id: &str) -> NousResult<Option<MemoryAtom>>;
    async fn get_all_atoms(&self) -> NousResult<Vec<MemoryAtom>>;
    async fn update_atom_fields(&mut self, id: &str, updates: AtomUpdate) -> NousResult<bool>;
    async fn remove_atom(&mut self, id: &str) -> NousResult<bool>;

    async fn store_synapse(&mut self, synapse: Synapse) -> NousResult<()>;
    async fn get_synapses_for(&self, atom_id: &str) -> NousResult<Vec<Synapse>>;
    async fn remove_synapse(&mut self, id: &str) -> NousResult<bool>;

    fn atom_count(&self) -> usize;
}

/// In-memory store — for testing and lightweight use.
///
/// Not persistent — data lost when dropped.
pub struct InMemoryStore {
    atoms: HashMap<String, MemoryAtom>,
    synapses: HashMap<String, Synapse>,
    /// atom_id → set of synapse_ids touching this atom.
    adjacency: HashMap<String, HashSet<String>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            atoms: HashMap::new(),
            synapses: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn store_atom(&mut self, atom: MemoryAtom) -> NousResult<()> {
        self.atoms.insert(atom.id.clone(), atom);
        Ok(())
    }

    async fn get_atom(&self, id: &str) -> NousResult<Option<MemoryAtom>> {
        Ok(self.atoms.get(id).cloned())
    }

    async fn get_all_atoms(&self) -> NousResult<Vec<MemoryAtom>> {
        Ok(self.atoms.values().cloned().collect())
    }

    async fn update_atom_fields(&mut self, id: &str, updates: AtomUpdate) -> NousResult<bool> {
        if let Some(atom) = self.atoms.get_mut(id) {
            if let Some(v) = updates.content { atom.content = v; }
            if let Some(v) = updates.importance { atom.importance = v; }
            if let Some(v) = updates.access_count { atom.access_count = v; }
            if let Some(v) = updates.last_accessed_at { atom.last_accessed_at = v; }
            if let Some(v) = updates.is_consolidated { atom.is_consolidated = v; }
            if let Some(v) = updates.superseded { atom.superseded = v; }
            if let Some(v) = updates.suppressed { atom.suppressed = v; }
            if let Some(v) = updates.dormant { atom.dormant = v; }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn remove_atom(&mut self, id: &str) -> NousResult<bool> {
        let removed = self.atoms.remove(id).is_some();
        // Clean up adjacency
        if let Some(syn_ids) = self.adjacency.remove(id) {
            for syn_id in syn_ids {
                self.synapses.remove(&syn_id);
            }
        }
        Ok(removed)
    }

    async fn store_synapse(&mut self, synapse: Synapse) -> NousResult<()> {
        self.adjacency
            .entry(synapse.source.clone())
            .or_default()
            .insert(synapse.id.clone());
        self.adjacency
            .entry(synapse.target.clone())
            .or_default()
            .insert(synapse.id.clone());
        self.synapses.insert(synapse.id.clone(), synapse);
        Ok(())
    }

    async fn get_synapses_for(&self, atom_id: &str) -> NousResult<Vec<Synapse>> {
        let syn_ids = self.adjacency.get(atom_id);
        let result = match syn_ids {
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.synapses.get(id).cloned())
                .collect(),
            None => vec![],
        };
        Ok(result)
    }

    async fn remove_synapse(&mut self, id: &str) -> NousResult<bool> {
        if let Some(syn) = self.synapses.remove(id) {
            if let Some(set) = self.adjacency.get_mut(&syn.source) {
                set.remove(id);
            }
            if let Some(set) = self.adjacency.get_mut(&syn.target) {
                set.remove(id);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn atom_count(&self) -> usize {
        self.atoms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::memory::{AtomSource, AtomType};

    fn make_atom(id: &str, content: &str) -> MemoryAtom {
        MemoryAtom {
            id: id.into(),
            content: content.into(),
            embedding: None,
            atom_type: AtomType::Episodic,
            source: AtomSource::default(),
            importance: 0.5,
            access_count: 0,
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

    #[tokio::test]
    async fn store_and_retrieve() {
        let mut store = InMemoryStore::new();
        store.store_atom(make_atom("a1", "hello")).await.unwrap();
        let atom = store.get_atom("a1").await.unwrap();
        assert!(atom.is_some());
        assert_eq!(atom.unwrap().content, "hello");
    }

    #[tokio::test]
    async fn update_atom() {
        let mut store = InMemoryStore::new();
        store.store_atom(make_atom("a1", "old")).await.unwrap();
        store.update_atom_fields("a1", AtomUpdate {
            content: Some("new".into()),
            ..Default::default()
        }).await.unwrap();
        let atom = store.get_atom("a1").await.unwrap().unwrap();
        assert_eq!(atom.content, "new");
    }

    #[tokio::test]
    async fn remove_atom_cleans_synapses() {
        let mut store = InMemoryStore::new();
        store.store_atom(make_atom("a1", "x")).await.unwrap();
        store.store_atom(make_atom("a2", "y")).await.unwrap();
        store.store_synapse(Synapse {
            id: "s1".into(),
            source: "a1".into(),
            target: "a2".into(),
            synapse_type: crate::types::memory::SynapseType::RelatesTo,
            strength: 0.5,
            created_at: 0.0,
            last_accessed_at: None,
            access_count: None,
        }).await.unwrap();

        assert_eq!(store.get_synapses_for("a1").await.unwrap().len(), 1);
        store.remove_atom("a1").await.unwrap();
        // Synapse should be cleaned up
        assert_eq!(store.get_synapses_for("a2").await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn atom_count() {
        let mut store = InMemoryStore::new();
        assert_eq!(store.atom_count(), 0);
        store.store_atom(make_atom("a1", "x")).await.unwrap();
        assert_eq!(store.atom_count(), 1);
    }
}
