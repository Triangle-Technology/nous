//! Intervention traits — model interface for cognitive intervention.
//!
//! Extends the AI provider abstraction (P4 trait boundary) with intervention
//! capabilities: sampling override, logit access, and logit bias injection.
//!
//! Brain analog: the interface between neuromodulatory systems and neural circuits.
//! Neuromodulators (DA, NE, ACh) don't process information — they change HOW
//! circuits process. These traits define that modulation interface.
//!
//! See `docs/intervention.md` Tier 1: Logit Intervention.

use async_trait::async_trait;

use crate::ai::provider::{CompletionRequest, CompletionResponse, StreamChunk};
use crate::errors::{NousError, NousResult};
use crate::types::intervention::{
    CognitiveState, InterventionDepth, LogitBias, SamplingOverride,
};

/// AI provider with intervention support — extends AiProvider with cognitive modulation.
///
/// Models implementing this trait allow Nous's cognitive state to influence
/// generation beyond text I/O. The intervention_depth() method declares what
/// level of intervention the model supports.
///
/// Brain analog: a neural circuit that accepts neuromodulatory input.
/// TextOnly circuits ignore modulation. LogitAccess circuits allow output gating.
/// ActivationAccess circuits allow internal state modification.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// What intervention depth this model supports.
    /// Used by cognitive modules to gracefully degrade (P5).
    fn intervention_depth(&self) -> InterventionDepth;

    /// Completion with cognitive sampling override.
    /// The SamplingOverride modulates temperature, top_p, penalties, and logit biases.
    /// Models that don't support certain overrides (e.g., logit_biases on TextOnly)
    /// should apply what they can and ignore the rest.
    async fn complete_with_override(
        &self,
        request: CompletionRequest,
        sampling: SamplingOverride,
    ) -> NousResult<CompletionResponse>;

    /// Streaming completion with cognitive sampling override.
    async fn stream_with_override(
        &self,
        request: CompletionRequest,
        sampling: SamplingOverride,
        sender: tokio::sync::mpsc::Sender<StreamChunk>,
    ) -> NousResult<()>;

    /// Tier 1+: Access raw logit distribution before sampling.
    ///
    /// Returns logit vector for the next token position.
    /// Default: not supported (returns UnsupportedIntervention error).
    ///
    /// Brain analog: reading the output layer activation before winner-take-all.
    async fn get_next_token_logits(
        &self,
        _request: CompletionRequest,
    ) -> NousResult<Vec<f32>> {
        Err(NousError::UnsupportedIntervention(format!(
            "get_next_token_logits requires {:?}, model supports {:?}",
            InterventionDepth::LogitAccess,
            self.intervention_depth(),
        )))
    }
}

/// Cognitive module logit intervention interface.
///
/// Cognitive modules implement this trait to provide token-level biases
/// based on current cognitive state. The biases are composed (summed)
/// across all active intervenors before being applied to model logits.
///
/// Brain analog: each neuromodulatory pathway (DA, NE, ACh, 5-HT)
/// independently modulates output probability. Their effects combine
/// additively at the synapse.
///
/// Note: Tier 1 Phase 1 — trait defined, no concrete implementations yet.
/// Token-level bias requires tokenizer integration (mapping semantic
/// concepts like "caution" to specific token IDs).
pub trait LogitIntervenor {
    /// Compute logit biases from current cognitive state.
    ///
    /// Returns empty vec if no intervention needed (most common case).
    /// Each LogitBias specifies a token_id and bias magnitude.
    fn compute_logit_biases(&self, state: &CognitiveState) -> Vec<LogitBias>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_intervenor_can_return_empty() {
        struct NoOpIntervenor;
        impl LogitIntervenor for NoOpIntervenor {
            fn compute_logit_biases(&self, _state: &CognitiveState) -> Vec<LogitBias> {
                Vec::new()
            }
        }

        let intervenor = NoOpIntervenor;
        let state = CognitiveState::default();
        let biases = intervenor.compute_logit_biases(&state);
        assert!(biases.is_empty());
    }

    #[test]
    fn logit_intervenor_returns_biases() {
        struct MockIntervenor;
        impl LogitIntervenor for MockIntervenor {
            fn compute_logit_biases(&self, state: &CognitiveState) -> Vec<LogitBias> {
                if state.arousal > 0.6 {
                    vec![LogitBias {
                        token_id: 100,
                        bias: -2.0,
                        source: "mock".into(),
                    }]
                } else {
                    Vec::new()
                }
            }
        }

        let intervenor = MockIntervenor;

        // Low arousal → no biases
        let calm = CognitiveState::default();
        assert!(intervenor.compute_logit_biases(&calm).is_empty());

        // High arousal → bias applied
        let aroused = CognitiveState {
            arousal: 0.8,
            ..CognitiveState::default()
        };
        let biases = intervenor.compute_logit_biases(&aroused);
        assert_eq!(biases.len(), 1);
        assert_eq!(biases[0].bias, -2.0);
    }
}
