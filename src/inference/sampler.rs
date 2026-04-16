//! Cognitive Sampler — brain-modulated token sampling.
//!
//! This is the point where Noos ACTUALLY intervenes in model output.
//! Raw logits from a model's forward pass are modulated by cognitive state
//! (arousal, gain mode, body budget, etc.) before sampling the next token.
//!
//! Brain analog: the output layer of cortex, where neuromodulators (DA, NE,
//! ACh, 5-HT) have already shaped activation patterns. The "decision" to
//! fire a particular output neuron is the result of both cortical computation
//! (logits) AND neuromodulatory context (CognitiveState).
//!
//! Key papers:
//! - Aston-Jones & Cohen 2005 (gain mode → temperature)
//! - Holtzman 2020 (nucleus sampling / top-p)
//! - Keskar 2019 (repetition penalties)
//! - Barrett 2017 (allostatic conservation → sampling tightening)
//!
//! Pure Rust, no candle dependency. Operates on &[f32] logits from any source.
//! <1ms per sample call.

use rand::Rng;

use crate::cognition::intervention::compute_sampling_override;
use crate::errors::{NoosError, NoosResult};
use crate::math::softmax::softmax_f32;
use crate::types::intervention::{CognitiveState, SamplingOverride};

/// Cognitive sampler — applies brain-derived modulation to raw model logits.
///
/// Created from CognitiveState (output of convergence loop).
/// Stateless per-token: each `sample()` call is pure given its inputs.
#[derive(Debug, Clone)]
pub struct CognitiveSampler {
    /// Sampling parameters derived from cognitive state.
    sampling: SamplingOverride,
}

impl CognitiveSampler {
    /// Create sampler from cognitive state.
    ///
    /// Translates convergence loop output → sampling parameters
    /// via `compute_sampling_override()`.
    pub fn from_cognitive_state(state: &CognitiveState) -> Self {
        Self {
            sampling: compute_sampling_override(state),
        }
    }

    /// Create sampler from explicit sampling override.
    pub fn from_override(sampling: SamplingOverride) -> Self {
        Self { sampling }
    }

    /// Access the computed sampling parameters (for inspection/logging).
    pub fn sampling(&self) -> &SamplingOverride {
        &self.sampling
    }

    /// Sample next token from logits, applying cognitive modulation.
    ///
    /// Pipeline (order matters — matches standard LLM sampling):
    /// 1. Apply logit biases (token-level amplification/suppression)
    /// 2. Apply repetition penalties (frequency + presence)
    /// 3. Apply temperature scaling
    /// 4. Convert to probabilities (softmax)
    /// 5. Apply top-p nucleus filtering
    /// 6. Weighted random sample from filtered distribution
    ///
    /// Returns token ID of the sampled token.
    pub fn sample(&self, logits: &[f32], previous_tokens: &[u32]) -> NoosResult<u32> {
        self.sample_with_rng(logits, previous_tokens, &mut rand::thread_rng())
    }

    /// Sample with explicit RNG (for deterministic testing).
    pub fn sample_with_rng(
        &self,
        logits: &[f32],
        previous_tokens: &[u32],
        rng: &mut impl Rng,
    ) -> NoosResult<u32> {
        if logits.is_empty() {
            return Err(NoosError::Internal("Empty logits vector".into()));
        }

        let mut modified = logits.to_vec();

        // Step 1: Apply logit biases (cognitive module interventions).
        // Brain analog: synaptic facilitation/depression at output layer.
        for bias in &self.sampling.logit_biases {
            let idx = bias.token_id as usize;
            if idx < modified.len() {
                modified[idx] += bias.bias as f32;
            }
        }

        // Step 2: Apply repetition penalties (Keskar 2019).
        // frequency_penalty: proportional to how often token appeared.
        // presence_penalty: flat penalty for any token that appeared at all.
        // Brain analog: habituation (Thompson & Spencer 1966) — repeated
        // stimuli produce weaker responses.
        if self.sampling.frequency_penalty > 0.0 || self.sampling.presence_penalty > 0.0 {
            apply_repetition_penalties(
                &mut modified,
                previous_tokens,
                self.sampling.frequency_penalty as f32,
                self.sampling.presence_penalty as f32,
            );
        }

        // Step 3: Temperature scaling.
        // Brain analog: gain control (Aston-Jones 2005). Low temp = high gain
        // (winner-take-all). High temp = low gain (equal competition).
        let temperature = self.sampling.temperature as f32;
        if temperature < 0.01 {
            // Near-zero temperature → argmax (deterministic).
            return Ok(argmax(&modified));
        }
        for logit in &mut modified {
            *logit /= temperature;
        }

        // Step 4: Softmax → probabilities.
        let probs = softmax_f32(&modified);

        // Step 5: Top-p nucleus filtering (Holtzman 2020).
        // Keep smallest set of tokens whose cumulative probability ≥ top_p.
        // Brain analog: competitive exclusion — only strong-enough activations
        // survive (Desimone & Duncan 1995, biased competition).
        let filtered = top_p_filter(&probs, self.sampling.top_p as f32);

        // Step 6: Weighted random sample from filtered distribution.
        // Brain analog: stochastic neural firing — even high-probability neurons
        // don't fire 100% of the time. Noise is a feature, not a bug
        // (stochastic resonance, Benzi 1981).
        Ok(weighted_sample(&filtered, rng))
    }

    /// Compute the full modulated probability distribution.
    ///
    /// Same pipeline as `sample()` but returns probabilities instead of
    /// sampling. Useful for:
    /// - Predictive coding: compare predicted vs actual distribution (Tier 2)
    /// - Entropy monitoring: detect model uncertainty
    /// - Debugging: visualize cognitive modulation effect
    pub fn modulated_distribution(
        &self,
        logits: &[f32],
        previous_tokens: &[u32],
    ) -> NoosResult<Vec<f32>> {
        if logits.is_empty() {
            return Err(NoosError::Internal("Empty logits vector".into()));
        }

        let mut modified = logits.to_vec();

        for bias in &self.sampling.logit_biases {
            let idx = bias.token_id as usize;
            if idx < modified.len() {
                modified[idx] += bias.bias as f32;
            }
        }

        if self.sampling.frequency_penalty > 0.0 || self.sampling.presence_penalty > 0.0 {
            apply_repetition_penalties(
                &mut modified,
                previous_tokens,
                self.sampling.frequency_penalty as f32,
                self.sampling.presence_penalty as f32,
            );
        }

        let temperature = self.sampling.temperature as f32;
        if temperature >= 0.01 {
            for logit in &mut modified {
                *logit /= temperature;
            }
        }

        let probs = softmax_f32(&modified);
        let filtered = top_p_filter(&probs, self.sampling.top_p as f32);
        Ok(filtered)
    }
}

// ─── Internal functions ──────────────────────────────────────────────

/// Apply frequency and presence penalties to logits.
///
/// For each token in previous_tokens:
/// - frequency_penalty: subtract (count × penalty) from logit
/// - presence_penalty: subtract penalty once if token appeared at all
///
/// Only penalizes positive logits (avoids amplifying already-suppressed tokens).
fn apply_repetition_penalties(
    logits: &mut [f32],
    previous_tokens: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    // Count occurrences of each token.
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &token in previous_tokens {
        *counts.entry(token).or_insert(0) += 1;
    }

    for (&token_id, &count) in &counts {
        let idx = token_id as usize;
        if idx < logits.len() {
            // Only penalize tokens with positive logits (avoid double-suppression).
            if logits[idx] > 0.0 {
                logits[idx] -= count as f32 * frequency_penalty;
                logits[idx] -= presence_penalty;
                // Don't let penalty push below zero.
                if logits[idx] < 0.0 {
                    logits[idx] = 0.0;
                }
            }
        }
    }
}

/// Weighted random sample from probability distribution.
///
/// Draws one token ID proportional to its probability.
/// Falls back to argmax if distribution is degenerate (all zeros).
fn weighted_sample(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return argmax(probs);
    }

    let mut threshold: f32 = rng.gen_range(0.0..sum);
    for (idx, &prob) in probs.iter().enumerate() {
        threshold -= prob;
        if threshold <= 0.0 {
            return idx as u32;
        }
    }

    // Fallback: rounding error → last token (P5: fail-open).
    (probs.len() - 1) as u32
}

/// Argmax — return index of maximum value.
/// Tie-breaking: first occurrence wins.
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0) // P5: fail-open — return token 0 if empty
}

/// Top-p (nucleus) filtering (Holtzman 2020).
///
/// Zeroes out tokens outside the top-p nucleus, then renormalizes.
/// The nucleus is the smallest set of tokens whose cumulative probability ≥ top_p.
fn top_p_filter(probs: &[f32], top_p: f32) -> Vec<f32> {
    if top_p >= 1.0 || probs.is_empty() {
        return probs.to_vec();
    }

    // Sort indices by probability (descending).
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff: smallest set where cumulative prob ≥ top_p.
    let mut cumulative = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out tokens outside the nucleus.
    let mut filtered = vec![0.0f32; probs.len()];
    for &(idx, prob) in &indexed[..cutoff_idx] {
        filtered[idx] = prob;
    }

    // Renormalize.
    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        for p in &mut filtered {
            *p /= sum;
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::belief::AffectValence;
    use crate::types::intervention::LogitBias;
    use crate::types::world::GainMode;

    // ─── Helper ───

    /// Simple logits: token 0 is strongest, decreasing.
    fn simple_logits() -> Vec<f32> {
        vec![2.0, 1.0, 0.5, 0.1, -1.0]
    }

    fn neutral_state() -> CognitiveState {
        CognitiveState::default()
    }

    // ─── CognitiveSampler construction ───

    #[test]
    fn from_cognitive_state_maps_correctly() {
        let state = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let sampler = CognitiveSampler::from_cognitive_state(&state);
        // Phasic → low temperature
        assert!(sampler.sampling().temperature <= 0.3);
    }

    #[test]
    fn from_override_preserves_params() {
        let override_ = SamplingOverride {
            temperature: 0.42,
            top_p: 0.88,
            ..SamplingOverride::default()
        };
        let sampler = CognitiveSampler::from_override(override_.clone());
        assert_eq!(sampler.sampling().temperature, 0.42);
        assert_eq!(sampler.sampling().top_p, 0.88);
    }

    // ─── Temperature ───

    #[test]
    fn deterministic_at_zero_temp() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 0.001, // Near-zero
            ..SamplingOverride::default()
        });
        let logits = simple_logits();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 0, "Near-zero temperature should select argmax (token 0)");
    }

    #[test]
    fn higher_temp_flattens_distribution() {
        let low_temp = CognitiveSampler::from_override(SamplingOverride {
            temperature: 0.1,
            top_p: 1.0,
            ..SamplingOverride::default()
        });
        let high_temp = CognitiveSampler::from_override(SamplingOverride {
            temperature: 2.0,
            top_p: 1.0,
            ..SamplingOverride::default()
        });
        let logits = simple_logits();

        let dist_low = low_temp.modulated_distribution(&logits, &[]).unwrap();
        let dist_high = high_temp.modulated_distribution(&logits, &[]).unwrap();

        // Low temp: top token gets most probability.
        // High temp: more uniform.
        let top_prob_low = dist_low[0];
        let top_prob_high = dist_high[0];
        assert!(
            top_prob_low > top_prob_high,
            "Low temp should concentrate more probability on top token"
        );
    }

    // ─── Top-p ───

    #[test]
    fn top_p_filters_low_probability_tokens() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 0.5, // Tight nucleus
            ..SamplingOverride::default()
        });
        let logits = simple_logits();
        let dist = sampler.modulated_distribution(&logits, &[]).unwrap();

        // With tight top_p, bottom tokens should be zeroed out.
        let nonzero_count = dist.iter().filter(|&&p| p > 0.0).count();
        assert!(
            nonzero_count < logits.len(),
            "Top-p should filter out some tokens"
        );
    }

    #[test]
    fn top_p_1_keeps_all_tokens() {
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let filtered = top_p_filter(&probs, 1.0);
        for (&orig, &filt) in probs.iter().zip(filtered.iter()) {
            assert!((orig - filt).abs() < 0.001);
        }
    }

    // ─── Repetition penalties ───

    #[test]
    fn frequency_penalty_suppresses_repeated_tokens() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 1.0,
            frequency_penalty: 0.5,
            ..SamplingOverride::default()
        });
        // Token 0 appeared 3 times in history.
        let prev = vec![0, 0, 0, 1];
        let logits = vec![2.0, 1.5, 0.5];

        let dist_no_penalty = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 1.0,
            ..SamplingOverride::default()
        })
        .modulated_distribution(&logits, &[])
        .unwrap();

        let dist_with_penalty = sampler.modulated_distribution(&logits, &prev).unwrap();

        // Token 0 should have lower probability with penalty.
        assert!(
            dist_with_penalty[0] < dist_no_penalty[0],
            "Frequency penalty should reduce probability of repeated token"
        );
    }

    #[test]
    fn presence_penalty_suppresses_seen_tokens() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 1.0,
            presence_penalty: 1.0,
            ..SamplingOverride::default()
        });
        // Token 1 appeared once.
        let prev = vec![1];
        let logits = vec![1.0, 1.0, 1.0];

        let dist = sampler.modulated_distribution(&logits, &prev).unwrap();

        // Token 1 should have lower probability than token 0 and 2 (which were never seen).
        assert!(
            dist[1] < dist[0],
            "Presence penalty should reduce probability of seen token"
        );
    }

    // ─── Logit biases ───

    #[test]
    fn logit_bias_amplifies_target() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 1.0,
            logit_biases: vec![LogitBias {
                token_id: 2,
                bias: 5.0,
                source: "test".into(),
            }],
            ..SamplingOverride::default()
        });
        let logits = vec![1.0, 1.0, 1.0]; // All equal
        let dist = sampler.modulated_distribution(&logits, &[]).unwrap();

        // Token 2 should dominate after +5.0 bias.
        assert!(
            dist[2] > dist[0],
            "Positive logit bias should amplify target token"
        );
        assert!(dist[2] > 0.9, "Strong positive bias should dominate");
    }

    #[test]
    fn logit_bias_suppresses_target() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.0,
            top_p: 1.0,
            logit_biases: vec![LogitBias {
                token_id: 0,
                bias: -5.0,
                source: "test".into(),
            }],
            ..SamplingOverride::default()
        });
        let logits = vec![1.0, 1.0, 1.0];
        let dist = sampler.modulated_distribution(&logits, &[]).unwrap();

        assert!(
            dist[0] < dist[1],
            "Negative logit bias should suppress target token"
        );
    }

    #[test]
    fn logit_bias_out_of_range_ignored() {
        let sampler = CognitiveSampler::from_override(SamplingOverride {
            logit_biases: vec![LogitBias {
                token_id: 999, // Beyond vocab
                bias: 10.0,
                source: "test".into(),
            }],
            ..SamplingOverride::default()
        });
        let logits = vec![1.0, 1.0, 1.0];
        // Should not panic.
        let result = sampler.sample(&logits, &[]);
        assert!(result.is_ok());
    }

    // ─── Edge cases (P5: fail-open) ───

    #[test]
    fn empty_logits_returns_error() {
        let sampler = CognitiveSampler::from_cognitive_state(&neutral_state());
        let result = sampler.sample(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn single_token_vocab() {
        let sampler = CognitiveSampler::from_cognitive_state(&neutral_state());
        let token = sampler.sample(&[1.0], &[]).unwrap();
        assert_eq!(token, 0);
    }

    // ─── Internal functions ───

    #[test]
    fn argmax_returns_max_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn weighted_sample_respects_distribution() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // 90% on token 0, 10% on token 1
        let probs = vec![0.9, 0.1];
        let mut counts = [0u32; 2];
        for _ in 0..1000 {
            let token = weighted_sample(&probs, &mut rng);
            counts[token as usize] += 1;
        }
        // Token 0 should be sampled ~900 times (±5%).
        assert!(counts[0] > 800, "Token 0 should be sampled most often");
        assert!(counts[1] > 50, "Token 1 should be sampled sometimes");
    }

    #[test]
    fn high_temp_produces_variety() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        let sampler = CognitiveSampler::from_override(SamplingOverride {
            temperature: 1.5, // High temp → more variety
            top_p: 1.0,
            ..SamplingOverride::default()
        });
        // All logits equal → uniform after softmax → each token equally likely.
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let token = sampler.sample_with_rng(&logits, &[], &mut rng).unwrap();
            seen.insert(token);
        }
        // With uniform distribution and 100 samples, should see most tokens.
        assert!(
            seen.len() >= 3,
            "High temp on uniform logits should produce variety"
        );
    }

    #[test]
    fn top_p_renormalizes() {
        let probs = vec![0.5, 0.3, 0.1, 0.1];
        let filtered = top_p_filter(&probs, 0.7);
        let sum: f32 = filtered.iter().sum();
        // Should be renormalized to ~1.0.
        assert!(
            (sum - 1.0).abs() < 0.01 || sum == 0.0,
            "Filtered probs should sum to ~1.0"
        );
    }

    // ─── Integration: cognitive state → sampling behavior ───

    #[test]
    fn phasic_gain_focuses_output() {
        let phasic = CognitiveState {
            gain_mode: GainMode::Phasic,
            ..CognitiveState::default()
        };
        let tonic = CognitiveState {
            gain_mode: GainMode::Tonic,
            ..CognitiveState::default()
        };

        let logits = simple_logits();
        let dist_phasic = CognitiveSampler::from_cognitive_state(&phasic)
            .modulated_distribution(&logits, &[])
            .unwrap();
        let dist_tonic = CognitiveSampler::from_cognitive_state(&tonic)
            .modulated_distribution(&logits, &[])
            .unwrap();

        // Phasic should concentrate more probability on top token.
        assert!(
            dist_phasic[0] > dist_tonic[0],
            "Phasic gain should focus output more than tonic"
        );
    }

    #[test]
    fn stressed_state_conserves() {
        let stressed = CognitiveState {
            arousal: 0.9,
            valence: AffectValence::Negative,
            body_budget: 0.1,
            ..CognitiveState::default()
        };
        let calm = CognitiveState::default();

        let sampler_stressed = CognitiveSampler::from_cognitive_state(&stressed);
        let sampler_calm = CognitiveSampler::from_cognitive_state(&calm);

        // Stressed should have lower temperature (conservation).
        assert!(
            sampler_stressed.sampling().temperature < sampler_calm.sampling().temperature,
            "Stressed state should produce lower temperature"
        );
        // Stressed should have frequency penalty (tunnel vision).
        assert!(
            sampler_stressed.sampling().frequency_penalty > 0.0,
            "High arousal + negative valence should activate frequency penalty"
        );
    }
}
