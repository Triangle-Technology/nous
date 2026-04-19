//! Vector math utilities — cosine similarity, clamp.

/// Cosine similarity between two vectors.
///
/// Returns value in [-1, 1]. Returns 0.0 if either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let ai = *ai as f64;
        let bi = *bi as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    (dot / denom) as f32
}

/// Clamp a value to `[min, max]`, absorbing NaN to `min`.
///
/// Differs from [`f64::clamp`] in NaN handling: `f64::clamp(NaN, a, b)`
/// returns `NaN` unchanged, which then cascades through downstream
/// arithmetic. This function uses `max(min).min(max)` — because
/// [`f64::max`] treats NaN as less than any number and [`f64::min`]
/// treats NaN as greater, a NaN input collapses to `min`. Chosen as
/// the project-wide clamp so that a stray NaN at a boundary cannot
/// corrupt invariants like `body_budget ∈ [0, 1]` (CR4 clamping bounds
/// are safety rails; P5 fail-open).
#[inline]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_different_length_returns_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn clamp_within_range() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
    }

    #[test]
    fn clamp_below_min() {
        assert_eq!(clamp(-0.5, 0.0, 1.0), 0.0);
    }

    #[test]
    fn clamp_above_max() {
        assert_eq!(clamp(1.5, 0.0, 1.0), 1.0);
    }

    #[test]
    fn clamp_nan_absorbs_to_min() {
        // Contract: NaN collapses to `min` so downstream arithmetic
        // cannot propagate NaN past a boundary (P5 fail-open). See the
        // docstring comparison with `f64::clamp`.
        assert_eq!(clamp(f64::NAN, 0.0, 1.0), 0.0);
        assert_eq!(clamp(f64::NAN, -1.0, 1.0), -1.0);
    }

    #[test]
    fn clamp_infinity() {
        assert_eq!(clamp(f64::INFINITY, 0.0, 1.0), 1.0);
        assert_eq!(clamp(f64::NEG_INFINITY, 0.0, 1.0), 0.0);
    }
}
