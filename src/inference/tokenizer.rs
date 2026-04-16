//! Tokenizer trait — text ↔ token conversion abstraction.
//!
//! Brain analog: sensory transduction (retina → spike trains, cochlea → neural code).
//! Key papers: Barlow 1961 (efficient coding hypothesis — sensory systems
//! encode information to minimize redundancy, analogous to BPE tokenization).
//!
//! This trait is the interface. Concrete implementations (HuggingFace
//! tokenizers, etc.) are behind feature flags.
//!
//! Performance: <1ms per encode/decode call.

use crate::errors::NoosResult;

/// Text-to-token and token-to-text conversion.
///
/// Brain analog: sensory transduction (retina → neural code, cochlea → spike trains).
/// Must be matched to the model — each model has its own tokenizer vocabulary.
pub trait NoosTokenizer: Send + Sync {
    /// Encode text to token IDs.
    ///
    /// `add_special_tokens`: if true, prepend BOS / append EOS as model requires.
    fn encode(&self, text: &str, add_special_tokens: bool) -> NoosResult<Vec<u32>>;

    /// Decode token IDs back to text.
    fn decode(&self, tokens: &[u32]) -> NoosResult<String>;

    /// Decode a single token to its string representation.
    /// Returns empty string for unknown tokens (P5: fail-open).
    fn decode_token(&self, token: u32) -> NoosResult<String>;

    /// Vocabulary size — must match the model's vocab_size().
    fn vocab_size(&self) -> usize;

    /// End-of-sequence token ID (used to detect generation completion).
    fn eos_token_id(&self) -> u32;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Mock tokenizer for testing — simple char-level encoding.
    pub(crate) struct MockTokenizer {
        vocab_size: usize,
        eos_id: u32,
    }

    impl MockTokenizer {
        pub fn new(vocab_size: usize) -> Self {
            Self {
                vocab_size,
                eos_id: (vocab_size - 1) as u32,
            }
        }
    }

    impl NoosTokenizer for MockTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> NoosResult<Vec<u32>> {
            // Simple: each char's byte value mod vocab_size.
            Ok(text
                .bytes()
                .map(|b| (b as u32) % self.vocab_size as u32)
                .collect())
        }

        fn decode(&self, tokens: &[u32]) -> NoosResult<String> {
            // Inverse of encode (lossy — for testing only).
            Ok(tokens
                .iter()
                .map(|&t| (t as u8 + b'a') as char)
                .collect())
        }

        fn decode_token(&self, token: u32) -> NoosResult<String> {
            Ok(((token as u8 + b'a') as char).to_string())
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn eos_token_id(&self) -> u32 {
            self.eos_id
        }
    }

    #[test]
    fn mock_tokenizer_encodes() {
        let tokenizer = MockTokenizer::new(256);
        let tokens = tokenizer.encode("hello", false).unwrap();
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn mock_tokenizer_decodes() {
        let tokenizer = MockTokenizer::new(256);
        let text = tokenizer.decode(&[0, 1, 2]).unwrap();
        assert_eq!(text.len(), 3);
    }

    #[test]
    fn mock_tokenizer_eos() {
        let tokenizer = MockTokenizer::new(100);
        assert_eq!(tokenizer.eos_token_id(), 99);
    }
}
