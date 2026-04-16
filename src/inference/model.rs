//! Local Model trait — in-process inference abstraction.
//!
//! Brain analog: cortical tissue — processes input and produces activations.
//! Key papers: Yamins & DiCarlo 2016 (deep nets as cortical models),
//! Caucheteux & King 2022 (LLM activations predict brain activity).
//!
//! Unlike AiProvider (text in, text out), LocalModel operates on tokens
//! and returns raw logits — enabling direct cognitive intervention.
//! Concrete implementations (candle, etc.) are behind feature flags.
//!
//! Performance: forward pass latency depends on model size (~10-500ms).

use crate::errors::NoosResult;

/// In-process model inference — produces raw logits from token sequences.
///
/// Brain analog: a cortical region that takes input activations and
/// produces output activations. The logits ARE the activations —
/// not a text summary of what the cortex "decided."
///
/// Implementations must be `Send` (can be moved between threads) but
/// NOT `Sync` (mutable KV cache prevents shared references).
/// Use one model per thread or wrap in Mutex.
pub trait LocalModel: Send {
    /// Run forward pass, return logits for the next token position.
    ///
    /// `tokens`: input token IDs (full sequence or just the new token
    ///           if KV cache is active).
    /// `position`: sequence position for KV cache indexing.
    ///
    /// Returns: logits vector of size `vocab_size()`. Each element
    /// is the unnormalized log-probability of that token being next.
    fn forward(&mut self, tokens: &[u32], position: usize) -> NoosResult<Vec<f32>>;

    /// Vocabulary size — length of the logits vector returned by forward().
    fn vocab_size(&self) -> usize;

    /// Reset KV cache for new conversation.
    ///
    /// Must be called between conversations to prevent cross-contamination
    /// of attention patterns.
    fn reset_cache(&mut self);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Mock model for testing — returns deterministic logits.
    pub(crate) struct MockModel {
        vocab_size: usize,
        call_count: usize,
    }

    impl MockModel {
        pub fn new(vocab_size: usize) -> Self {
            Self {
                vocab_size,
                call_count: 0,
            }
        }
    }

    impl LocalModel for MockModel {
        fn forward(&mut self, _tokens: &[u32], _position: usize) -> NoosResult<Vec<f32>> {
            self.call_count += 1;
            // Return logits where token at index (call_count % vocab_size) is strongest.
            let mut logits = vec![0.0f32; self.vocab_size];
            let peak = self.call_count % self.vocab_size;
            logits[peak] = 5.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn reset_cache(&mut self) {
            self.call_count = 0;
        }
    }

    #[test]
    fn mock_model_returns_logits() {
        let mut model = MockModel::new(10);
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn mock_model_cycles_peak() {
        let mut model = MockModel::new(3);

        let logits1 = model.forward(&[1], 0).unwrap();
        assert_eq!(logits1[1], 5.0); // call_count=1 → peak at 1

        let logits2 = model.forward(&[1], 1).unwrap();
        assert_eq!(logits2[2], 5.0); // call_count=2 → peak at 2
    }

    #[test]
    fn reset_cache_resets_state() {
        let mut model = MockModel::new(3);
        model.forward(&[1], 0).unwrap();
        model.forward(&[1], 1).unwrap();
        model.reset_cache();

        let logits = model.forward(&[1], 0).unwrap();
        assert_eq!(logits[1], 5.0); // Back to call_count=1
    }
}
