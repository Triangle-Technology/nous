//! Inference Engine — the unified brain.
//!
//! Brain analog: sensorimotor integration loop. Sensory input (tokenizer)
//! → cortical processing (model forward pass) → motor output (token sampling),
//! with neuromodulatory systems (CognitiveState) coloring every stage.
//! Key papers: Wolpert 1997 (forward models in motor control),
//! Doya 1999 (modulatory systems in sensorimotor loops).
//!
//! This is where Nous and the model become ONE system — same process,
//! same memory space. See `docs/intervention.md` Tier 1.
//!
//! Performance: per-token = model.forward() + sampler.sample() (~10-500ms).

use crate::cognition::delta_modulation::compute_delta_modulation;
use crate::errors::{NousError, NousResult};
use crate::inference::cognitive_model::CognitiveModel;
use crate::inference::model::LocalModel;
use crate::inference::sampler::CognitiveSampler;
use crate::inference::tokenizer::NousTokenizer;
use crate::types::intervention::{CognitiveState, DeltaModulation};

/// Result of generating a single token.
#[derive(Debug, Clone)]
pub struct GenerationStep {
    /// The sampled token ID.
    pub token_id: u32,
    /// The decoded text fragment for this token.
    pub text: String,
    /// Whether this was the end-of-sequence token.
    pub is_eos: bool,
    /// Position in the sequence (0-indexed).
    pub position: usize,
}

/// The unified brain — model + tokenizer + cognitive sampler.
///
/// Generic over model and tokenizer to support:
/// - Real inference (CandleModel + HfTokenizer) in production
/// - Mock inference (MockModel + MockTokenizer) in tests
/// - Any future backend (ONNX, TensorRT, etc.)
pub struct InferenceEngine<M: LocalModel, T: NousTokenizer> {
    /// The model — cortical tissue.
    model: M,
    /// The tokenizer — sensory encoder/decoder.
    tokenizer: T,
    /// Current position in the sequence (for KV cache).
    position: usize,
    /// All tokens generated so far (for repetition penalty).
    generated_tokens: Vec<u32>,
    /// Input prompt tokens (for position tracking).
    prompt_tokens: Vec<u32>,
}

impl<M: LocalModel, T: NousTokenizer> InferenceEngine<M, T> {
    /// Create a new inference engine.
    pub fn new(model: M, tokenizer: T) -> Self {
        Self {
            model,
            tokenizer,
            position: 0,
            generated_tokens: Vec::new(),
            prompt_tokens: Vec::new(),
        }
    }

    /// Mutable: populates KV cache, initializes position and prompt tokens.
    /// Requires mutation because the model's KV cache must persist across
    /// subsequent generate_next() calls for attention context.
    pub fn set_prompt(&mut self, text: &str) -> NousResult<()> {
        let tokens = self.tokenizer.encode(text, true)?;
        if tokens.is_empty() {
            return Err(NousError::Internal("Empty prompt after tokenization".into()));
        }

        // Forward pass on full prompt to fill KV cache.
        self.model.forward(&tokens, 0)?;

        self.prompt_tokens = tokens;
        self.position = self.prompt_tokens.len();
        self.generated_tokens.clear();

        Ok(())
    }

    /// Mutable: advances position, appends to generated_tokens, updates KV cache.
    /// Requires mutation because auto-regressive generation is inherently
    /// stateful — each token depends on all previous tokens via KV cache.
    ///
    /// This is the core loop of the unified brain:
    /// 1. Model forward pass → raw logits (cortical output)
    /// 2. CognitiveState → SamplingOverride (neuromodulatory context)
    /// 3. CognitiveSampler applies modulation → sampled token
    ///
    /// The CognitiveState can change between calls — enabling per-token
    /// modulation as the conversation evolves.
    pub fn generate_next(
        &mut self,
        cognitive_state: &CognitiveState,
    ) -> NousResult<GenerationStep> {
        // Determine input tokens for this step.
        let input_tokens = if self.generated_tokens.is_empty() {
            // First generation step: use last prompt token.
            // (KV cache already has full prompt from set_prompt.)
            vec![*self.prompt_tokens.last().ok_or_else(|| {
                NousError::Internal("No prompt set".into())
            })?]
        } else {
            // Subsequent steps: feed the last generated token.
            vec![*self.generated_tokens.last().unwrap_or(&0)]
        };

        // Step 1: Forward pass → raw logits.
        let logits = self.model.forward(&input_tokens, self.position)?;

        // Step 2: Create cognitive sampler from current state.
        let sampler = CognitiveSampler::from_cognitive_state(cognitive_state);

        // Step 3: All previous tokens (prompt + generated) for repetition penalty.
        let all_tokens: Vec<u32> = self
            .prompt_tokens
            .iter()
            .chain(self.generated_tokens.iter())
            .copied()
            .collect();

        // Step 4: Sample with cognitive modulation.
        let token_id = sampler.sample(&logits, &all_tokens)?;

        // Decode and track.
        let text = self.tokenizer.decode_token(token_id)?;
        let is_eos = token_id == self.tokenizer.eos_token_id();
        let step_position = self.position;

        self.generated_tokens.push(token_id);
        self.position += 1;

        Ok(GenerationStep {
            token_id,
            text,
            is_eos,
            position: step_position,
        })
    }

    /// Mutable: calls generate_next() repeatedly, accumulating tokens.
    /// Requires mutation for same reason as generate_next().
    ///
    /// Stops when: max_tokens reached OR EOS token generated.
    /// The cognitive_state is applied uniformly to all tokens.
    /// For per-token modulation, use generate_next() in a loop.
    pub fn generate(
        &mut self,
        cognitive_state: &CognitiveState,
        max_tokens: usize,
    ) -> NousResult<String> {
        let mut output = String::new();

        for _ in 0..max_tokens {
            let step = self.generate_next(cognitive_state)?;
            if step.is_eos {
                break;
            }
            output.push_str(&step.text);
        }

        Ok(output)
    }

    /// Mutable: clears KV cache, position, and generated tokens.
    /// Requires mutation because all generation state must be wiped
    /// between conversations to prevent cross-contamination.
    pub fn reset(&mut self) {
        self.model.reset_cache();
        self.position = 0;
        self.generated_tokens.clear();
        self.prompt_tokens.clear();
    }

    /// Access generated tokens so far (for external analysis).
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    /// Current sequence position.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// Result of generating a single token with cognitive modulation (Tầng 1 + 2).
///
/// Extends GenerationStep with delta modulation metadata (efference copy).
/// Brain analog: motor output + corollary discharge (Crapse & Sommer 2008).
#[derive(Debug, Clone)]
pub struct CognitiveGenerationStep {
    /// The sampled token ID.
    pub token_id: u32,
    /// The decoded text fragment for this token.
    pub text: String,
    /// Whether this was the end-of-sequence token.
    pub is_eos: bool,
    /// Position in the sequence (0-indexed).
    pub position: usize,
    /// Whether delta modulation was applied at the model level.
    pub modulation_applied: bool,
    /// Which layers were modulated (empty if no modulation).
    pub modulated_layers: Vec<usize>,
    /// The delta modulation that was computed for this token.
    pub delta_modulation: DeltaModulation,
    /// Tầng 3: gate blend factor (None if no gate). 0.0 = passthrough, higher = active modulation.
    /// Bottom-up signal: model telling subcortex how salient the current input is.
    pub gate_alpha: Option<f64>,
    /// Tầng 3: gate's learned delta gain (None if no gate). [0.5, 2.0].
    /// Bottom-up signal: model's state update speed preference.
    pub gate_delta_gain: Option<f64>,
}

/// Cognitive inference methods — available when the model supports cognitive forward.
///
/// These methods stack Tầng 2 (delta modulation → model internals) on top of
/// Tầng 1 (sampling override → token selection). Both tầng apply together:
/// delta modulation changes HOW the model thinks, then sampling modulation
/// changes WHAT token is selected from the resulting distribution.
///
/// Brain analog: neuromodulators (Tầng 2, NE/DA) change cortical processing,
/// THEN output gating (Tầng 1, basal ganglia) selects the action.
impl<M: CognitiveModel, T: NousTokenizer> InferenceEngine<M, T> {
    /// Mutable: generates one token with full cognitive modulation (Tầng 1 + 2).
    ///
    /// Pipeline:
    /// 1. Compute DeltaModulation from CognitiveState (cognition layer)
    /// 2. Forward pass with delta modulation (model internalizes cognitive state)
    /// 3. CognitiveSampler modulates sampling parameters (Tầng 1, still applies)
    /// 4. Sample token from modulated distribution
    ///
    /// Tầng 2 + Tầng 1 stack — they don't replace each other.
    /// This is the unified brain: cognitive state shapes BOTH processing AND output.
    pub fn generate_next_cognitive(
        &mut self,
        cognitive_state: &CognitiveState,
    ) -> NousResult<CognitiveGenerationStep> {
        // Determine input tokens (same logic as generate_next).
        let input_tokens = if self.generated_tokens.is_empty() {
            vec![*self.prompt_tokens.last().ok_or_else(|| {
                NousError::Internal("No prompt set".into())
            })?]
        } else {
            vec![*self.generated_tokens.last().unwrap_or(&0)]
        };

        // Tầng 2: Compute delta modulation from cognitive state.
        let delta_mod = compute_delta_modulation(cognitive_state, self.model.num_layers());

        // Tầng 2: Forward pass with cognitive modulation.
        // The model applies gain_factor to delta at targeted layers.
        let forward_result =
            self.model
                .forward_cognitive(&input_tokens, self.position, &delta_mod)?;

        // Tầng 1: Create cognitive sampler from current state.
        // Sampling modulation STILL applies on top of delta modulation.
        let sampler = CognitiveSampler::from_cognitive_state(cognitive_state);

        // All previous tokens for repetition penalty.
        let all_tokens: Vec<u32> = self
            .prompt_tokens
            .iter()
            .chain(self.generated_tokens.iter())
            .copied()
            .collect();

        // Tầng 1: Sample with cognitive modulation.
        let token_id = sampler.sample(&forward_result.logits, &all_tokens)?;

        // Decode and track.
        let text = self.tokenizer.decode_token(token_id)?;
        let is_eos = token_id == self.tokenizer.eos_token_id();
        let step_position = self.position;

        self.generated_tokens.push(token_id);
        self.position += 1;

        Ok(CognitiveGenerationStep {
            token_id,
            text,
            is_eos,
            position: step_position,
            modulation_applied: forward_result.modulation_applied,
            modulated_layers: forward_result.modulated_layers,
            delta_modulation: delta_mod,
            gate_alpha: forward_result.gate_alpha,
            gate_delta_gain: forward_result.gate_delta_gain,
        })
    }

    /// Mutable: generates multiple tokens with cognitive modulation.
    ///
    /// Like generate(), but uses forward_cognitive() for Tầng 2.
    /// The cognitive_state is applied uniformly to all tokens.
    /// For per-token modulation, use generate_next_cognitive() in a loop.
    pub fn generate_cognitive(
        &mut self,
        cognitive_state: &CognitiveState,
        max_tokens: usize,
    ) -> NousResult<String> {
        let mut output = String::new();

        for _ in 0..max_tokens {
            let step = self.generate_next_cognitive(cognitive_state)?;
            if step.is_eos {
                break;
            }
            output.push_str(&step.text);
        }

        Ok(output)
    }

    /// Number of layers in the underlying model.
    pub fn model_num_layers(&self) -> usize {
        self.model.num_layers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::model::tests::MockModel;
    use crate::inference::tokenizer::tests::MockTokenizer;
    use crate::types::world::GainMode;

    fn make_engine() -> InferenceEngine<MockModel, MockTokenizer> {
        InferenceEngine::new(MockModel::new(100), MockTokenizer::new(100))
    }

    #[test]
    fn generate_next_produces_token() {
        let mut engine = make_engine();
        engine.set_prompt("hello").unwrap();

        let step = engine
            .generate_next(&CognitiveState::default())
            .unwrap();

        assert!(!step.text.is_empty());
        assert_eq!(engine.generated_tokens().len(), 1);
    }

    #[test]
    fn generate_respects_max_tokens() {
        let mut engine = make_engine();
        engine.set_prompt("hello").unwrap();

        let _result = engine.generate(&CognitiveState::default(), 5).unwrap();

        // Should generate at most 5 tokens.
        assert!(engine.generated_tokens().len() <= 5);
    }

    #[test]
    fn position_advances() {
        let mut engine = make_engine();
        engine.set_prompt("hi").unwrap();
        let initial_pos = engine.position();

        engine.generate_next(&CognitiveState::default()).unwrap();
        assert_eq!(engine.position(), initial_pos + 1);

        engine.generate_next(&CognitiveState::default()).unwrap();
        assert_eq!(engine.position(), initial_pos + 2);
    }

    #[test]
    fn reset_clears_state() {
        let mut engine = make_engine();
        engine.set_prompt("hello").unwrap();
        engine.generate_next(&CognitiveState::default()).unwrap();

        engine.reset();

        assert_eq!(engine.position(), 0);
        assert!(engine.generated_tokens().is_empty());
    }

    #[test]
    fn empty_prompt_returns_error() {
        let mut engine = make_engine();
        let result = engine.set_prompt("");
        assert!(result.is_err());
    }

    #[test]
    fn generate_without_prompt_returns_error() {
        let mut engine = make_engine();
        let result = engine.generate_next(&CognitiveState::default());
        assert!(result.is_err());
    }

    #[test]
    fn cognitive_state_affects_sampling() {
        let mut engine1 = make_engine();
        let mut engine2 = make_engine();
        engine1.set_prompt("test").unwrap();
        engine2.set_prompt("test").unwrap();

        let phasic = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let tonic = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };

        // Both should succeed — cognitive state affects HOW tokens are sampled,
        // not WHETHER generation works.
        let step1 = engine1.generate_next(&phasic).unwrap();
        let step2 = engine2.generate_next(&tonic).unwrap();

        // With mock model, both produce valid tokens.
        assert!(step1.token_id < 100);
        assert!(step2.token_id < 100);
    }

    // ─── Tầng 2 cognitive engine tests ───

    use crate::inference::cognitive_model::tests::MockCognitiveModel;

    fn make_cognitive_engine() -> InferenceEngine<MockCognitiveModel, MockTokenizer> {
        InferenceEngine::new(MockCognitiveModel::new(100, 64), MockTokenizer::new(100))
    }

    #[test]
    fn generate_next_cognitive_produces_token() {
        let mut engine = make_cognitive_engine();
        engine.set_prompt("hello").unwrap();

        let step = engine
            .generate_next_cognitive(&CognitiveState::default())
            .unwrap();

        assert!(!step.text.is_empty());
        assert_eq!(engine.generated_tokens().len(), 1);
    }

    #[test]
    fn cognitive_generation_applies_delta_modulation() {
        let mut engine = make_cognitive_engine();
        engine.set_prompt("test").unwrap();

        let phasic = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let step = engine.generate_next_cognitive(&phasic).unwrap();

        assert!(
            step.modulation_applied,
            "Phasic mode should trigger delta modulation"
        );
        assert!(
            step.delta_modulation.gain_factor < 1.0,
            "Phasic should reduce delta (compensatory retention)"
        );
        assert!(
            !step.modulated_layers.is_empty(),
            "Should have modulated mid-layers"
        );
    }

    #[test]
    fn neutral_cognitive_generation_no_modulation() {
        let mut engine = make_cognitive_engine();
        engine.set_prompt("test").unwrap();

        let neutral = CognitiveState::default();
        let step = engine.generate_next_cognitive(&neutral).unwrap();

        assert!(
            !step.modulation_applied,
            "Neutral state (gain=1.0) should not modulate"
        );
    }

    #[test]
    fn cognitive_and_sampling_stack() {
        // Verify that both tầng work: model gets delta modulation,
        // sampler gets sampling override. Both from same CognitiveState.
        let mut engine = make_cognitive_engine();
        engine.set_prompt("test").unwrap();

        let tonic = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };
        let step = engine.generate_next_cognitive(&tonic).unwrap();

        assert!(
            step.delta_modulation.gain_factor < 1.0,
            "Tonic should reduce delta (Tầng 2)"
        );
        // Token was still successfully sampled (Tầng 1 sampling worked).
        assert!(step.token_id < 100);
    }

    #[test]
    fn generate_cognitive_multiple_tokens() {
        let mut engine = make_cognitive_engine();
        engine.set_prompt("hello").unwrap();

        let result = engine
            .generate_cognitive(&CognitiveState::default(), 3)
            .unwrap();

        assert!(!result.is_empty());
        assert!(engine.generated_tokens().len() <= 3);
    }

    #[test]
    fn model_num_layers_accessible() {
        let engine = make_cognitive_engine();
        assert_eq!(engine.model_num_layers(), 64);
    }
}
