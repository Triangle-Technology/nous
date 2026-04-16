//! Softmax with temperature — biased competition (Desimone & Duncan 1995).
//!
//! Lower temperature → winner-take-more (phasic/focused).
//! Higher temperature → equal competition (tonic/exploration).

/// Softmax with temperature parameter.
///
/// Returns a probability distribution over the input utilities.
/// - `temperature` near 0 → argmax (winner-take-all)
/// - `temperature` → ∞ → uniform distribution
///
/// Handles edge cases: empty input returns empty, single input returns [1.0].
pub fn softmax(utilities: &[f64], temperature: f64) -> Vec<f64> {
    if utilities.is_empty() {
        return Vec::new();
    }
    if utilities.len() == 1 {
        return vec![1.0];
    }

    let temp = if temperature <= 0.0 { 0.01 } else { temperature };

    // Subtract max for numerical stability
    let max_u = utilities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_values: Vec<f64> = utilities.iter().map(|u| ((u - max_u) / temp).exp()).collect();

    let sum: f64 = exp_values.iter().sum();
    if sum == 0.0 {
        // Fallback: uniform distribution
        let n = utilities.len() as f64;
        return vec![1.0 / n; utilities.len()];
    }

    exp_values.iter().map(|e| e / sum).collect()
}

/// Softmax over f32 slice — for inference logits (P3: single source of truth).
///
/// Same algorithm as `softmax()` but operates on f32 (model logit precision)
/// with temperature=1.0 (temperature scaling is done by the caller before
/// passing logits to softmax).
///
/// Used by `CognitiveSampler` for token probability computation.
pub fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    if logits.len() == 1 {
        return vec![1.0];
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

    if exp_sum == 0.0 {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }

    logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[], 1.0).is_empty());
    }

    #[test]
    fn softmax_single() {
        assert_eq!(softmax(&[5.0], 1.0), vec![1.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let result = softmax(&[1.0, 2.0, 3.0], 1.0);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn softmax_low_temp_concentrates() {
        let result = softmax(&[1.0, 3.0], 0.1);
        // Low temp → almost all weight on the larger value
        assert!(result[1] > 0.99);
    }

    #[test]
    fn softmax_high_temp_equalizes() {
        let result = softmax(&[1.0, 3.0], 100.0);
        // High temp → nearly uniform
        assert!((result[0] - result[1]).abs() < 0.1);
    }

    #[test]
    fn softmax_preserves_order() {
        let result = softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    // ─── softmax_f32 tests ───

    #[test]
    fn softmax_f32_empty() {
        assert!(softmax_f32(&[]).is_empty());
    }

    #[test]
    fn softmax_f32_single() {
        assert_eq!(softmax_f32(&[5.0]), vec![1.0]);
    }

    #[test]
    fn softmax_f32_sums_to_one() {
        let result = softmax_f32(&[2.0, 1.0, 0.5, -1.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn softmax_f32_preserves_order() {
        let result = softmax_f32(&[3.0, 1.0, 2.0]);
        assert!(result[0] > result[2]);
        assert!(result[2] > result[1]);
    }
}
