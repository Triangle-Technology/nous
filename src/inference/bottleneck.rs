//! Bottleneck Steering — Tầng 4 structural compensation.
//!
//! Brain analog: thalamic relay dynamically gates information flow through
//! cortical bottleneck layers. Mamba has a routing bottleneck at Layer 20
//! (in mamba-130m) where diverse information is forced through a narrow
//! parameter subset (SPD entropy spike, KL divergence = 813).
//!
//! Scaling activations at the bottleneck layer widens this constraint,
//! improving downstream performance. This is P9-compatible: compensating
//! a structural limitation (information bottleneck), not amplifying
//! what the model already does well.
//!
//! Key paper:
//! - Mohan, Gupta, Das, Singh 2026 — "Interpreting and Steering
//!   State-Space Models via Activation Subspace Bottlenecks"
//!   (arXiv 2602.22719). +8.27% avg across 5 SSMs / 6 benchmarks.
//!
//! Requires `candle` feature flag.

use candle_core::{Device, Result as CandleResult, Tensor};

// ─── Constants ─────────────────────────────────────────────────────

/// Default bottleneck layer for mamba-130m (24 layers).
/// Layer 20 = 83% depth. Identified via SPD entropy spike + KL=813.
/// (Mohan et al. 2026, Table 4-5: Layer 20 has highest entropy, variance, rank)
const DEFAULT_BOTTLENECK_LAYER_130M: usize = 20;

/// Default scale factor for high-impact channels.
/// Grid search over 0.1-100 found 5× optimal (Mohan et al. 2026, §3.2).
const DEFAULT_SCALE_HIGH: f64 = 5.0;

/// Default scale factor for moderate-impact channels.
/// Second-best after 5×. Applied to channels with |ablation drop| < 2%.
/// (Mohan et al. 2026, §3.2: 155 channels at 2×)
const DEFAULT_SCALE_MODERATE: f64 = 2.0;

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

/// Configuration for bottleneck steering at a specific layer.
///
/// Specifies which layer to steer and per-channel scale factors.
/// The paper identifies channels via delta-sensitivity analysis
/// (ablation on The Pile). For initial validation, uniform scaling
/// across all channels reproduces the core finding.
#[derive(Debug, Clone)]
pub struct BottleneckConfig {
    /// Which layer to apply steering (0-indexed).
    /// Layer 20 for mamba-130m (Mohan et al. 2026).
    pub layer_index: usize,
    /// Residual stream width (d_model). 768 for mamba-130m.
    pub d_model: usize,
    /// Per-channel scale factors. Length must equal d_model.
    /// 1.0 = no steering (passthrough). 5.0 = paper's best.
    pub channel_scales: Vec<f64>,
}

impl BottleneckConfig {
    /// Uniform scaling: all channels get the same scale factor.
    /// Use for initial validation before channel-specific calibration.
    pub fn uniform(layer_index: usize, d_model: usize, scale: f64) -> Self {
        Self {
            layer_index,
            d_model,
            channel_scales: vec![scale; d_model],
        }
    }

    /// Default config for mamba-130m: Layer 20, uniform 5× scaling.
    /// Paper's best result (Mohan et al. 2026, §3.2).
    pub fn default_mamba_130m() -> Self {
        Self::uniform(DEFAULT_BOTTLENECK_LAYER_130M, 768, DEFAULT_SCALE_HIGH)
    }

    /// Layer 20, uniform 2× scaling (paper's second-best).
    pub fn moderate_mamba_130m() -> Self {
        Self::uniform(DEFAULT_BOTTLENECK_LAYER_130M, 768, DEFAULT_SCALE_MODERATE)
    }

    /// Identity config: scale = 1.0 everywhere. For control experiments.
    pub fn identity(layer_index: usize, d_model: usize) -> Self {
        Self::uniform(layer_index, d_model, 1.0)
    }

    /// Check if all scales are 1.0 (no-op steering).
    pub fn is_identity(&self) -> bool {
        self.channel_scales.iter().all(|&s| (s - 1.0).abs() < f64::EPSILON)
    }
}

// ═══════════════════════════════════════════════════════════════════
// BottleneckSteering — pre-computed tensor for efficient inference
// ═══════════════════════════════════════════════════════════════════

/// Pre-computed bottleneck steering tensor.
///
/// Converts channel_scales from Vec<f64> to a Tensor once at construction,
/// then applies via efficient element-wise multiply at inference.
///
/// Brain analog: thalamic relay with fixed gating weights that widen
/// the cortical bottleneck. Future Tầng 4 expansion will make these
/// weights dynamic (cognitive-state-driven).
pub struct BottleneckSteering {
    /// The config that produced this steering.
    config: BottleneckConfig,
    /// Pre-computed scale tensor. Shape: (1, d_model).
    /// Element-wise multiply with mixer output at bottleneck layer.
    scale_tensor: Tensor,
}

impl BottleneckSteering {
    /// Create steering from config. Pre-computes scale tensor on device.
    pub fn new(config: BottleneckConfig, device: &Device) -> CandleResult<Self> {
        let scales_f32: Vec<f32> = config.channel_scales.iter().map(|&s| s as f32).collect();
        let scale_tensor = Tensor::new(&scales_f32[..], device)?
            .unsqueeze(0)?; // (1, d_model)

        Ok(Self {
            config,
            scale_tensor,
        })
    }

    /// Apply steering: element-wise multiply xs by scale tensor.
    ///
    /// `xs`: mixer output at bottleneck layer. Shape: (batch, d_model).
    /// Returns: scaled tensor, same shape.
    ///
    /// No-op if config is identity (all 1.0), but caller should check
    /// `should_steer()` to avoid the call entirely.
    pub fn apply(&self, xs: &Tensor) -> CandleResult<Tensor> {
        xs.broadcast_mul(&self.scale_tensor)
    }

    /// Check if the given layer index matches the bottleneck layer.
    pub fn should_steer(&self, layer_index: usize) -> bool {
        layer_index == self.config.layer_index && !self.config.is_identity()
    }

    /// Access config.
    pub fn config(&self) -> &BottleneckConfig {
        &self.config
    }

    /// Bottleneck layer index.
    pub fn layer_index(&self) -> usize {
        self.config.layer_index
    }
}

// ═══════════════════════════════════════════════════════════════════
// Calibration — identify delta-sensitive channels
// ═══════════════════════════════════════════════════════════════════

/// Result of bottleneck calibration: per-channel variance statistics.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Per-channel variance of mixer output across calibration tokens.
    /// High variance = delta-sensitive (activation varies strongly with input).
    pub channel_variances: Vec<f64>,
    /// Number of tokens used for calibration.
    pub num_tokens: usize,
    /// Target layer that was calibrated.
    pub layer_index: usize,
}

impl CalibrationResult {
    /// Build a BottleneckConfig from calibration results.
    ///
    /// Channels are split into tiers by variance:
    /// - Top `high_frac` channels by variance → `high_scale` (paper: 5×)
    /// - Next `mid_frac` channels → `mid_scale` (paper: 2×)
    /// - Remaining channels → 1.0 (no steering)
    ///
    /// Paper defaults: high_frac=0.57 (435/768), mid_frac=0.20 (155/768).
    pub fn to_config(
        &self,
        high_frac: f64,
        high_scale: f64,
        mid_frac: f64,
        mid_scale: f64,
    ) -> BottleneckConfig {
        let d_model = self.channel_variances.len();
        let mut indexed: Vec<(usize, f64)> = self
            .channel_variances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by variance descending — highest variance first.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let high_count = (d_model as f64 * high_frac).round() as usize;
        let mid_count = (d_model as f64 * mid_frac).round() as usize;

        let mut channel_scales = vec![1.0_f64; d_model];

        for (rank, &(ch_idx, _)) in indexed.iter().enumerate() {
            if rank < high_count {
                channel_scales[ch_idx] = high_scale;
            } else if rank < high_count + mid_count {
                channel_scales[ch_idx] = mid_scale;
            }
        }

        BottleneckConfig {
            layer_index: self.layer_index,
            d_model,
            channel_scales,
        }
    }

    /// Paper defaults: 57% at 5×, 20% at 2×, rest at 1×.
    pub fn to_config_paper_defaults(&self) -> BottleneckConfig {
        self.to_config(0.57, DEFAULT_SCALE_HIGH, 0.20, DEFAULT_SCALE_MODERATE)
    }
}

/// Compute per-channel variance from collected mixer outputs.
///
/// `samples`: Vec of per-token mixer outputs, each Vec<f32> of length d_model.
/// Returns variance per channel across all tokens.
pub fn compute_channel_variance(samples: &[Vec<f32>], d_model: usize) -> Vec<f64> {
    if samples.is_empty() {
        return vec![0.0; d_model];
    }

    let n = samples.len() as f64;
    let mut means = vec![0.0_f64; d_model];
    let mut m2 = vec![0.0_f64; d_model];

    // Welford's online algorithm for numerical stability.
    for (count, sample) in samples.iter().enumerate() {
        let c = (count + 1) as f64;
        for (ch, &val) in sample.iter().enumerate() {
            if ch >= d_model {
                break;
            }
            let v = val as f64;
            let delta = v - means[ch];
            means[ch] += delta / c;
            let delta2 = v - means[ch];
            m2[ch] += delta * delta2;
        }
    }

    m2.iter().map(|&v| if n > 1.0 { v / (n - 1.0) } else { 0.0 }).collect()
}

// ═══════════════════════════════════════════════════════════════════
// Debug
// ═══════════════════════════════════════════════════════════════════

impl std::fmt::Debug for BottleneckSteering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BottleneckSteering")
            .field("layer", &self.config.layer_index)
            .field("d_model", &self.config.d_model)
            .field("is_identity", &self.config.is_identity())
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::DType;

    #[test]
    fn config_uniform_creates_correct_scales() {
        let cfg = BottleneckConfig::uniform(20, 768, 5.0);
        assert_eq!(cfg.layer_index, 20);
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.channel_scales.len(), 768);
        assert!(cfg.channel_scales.iter().all(|&s| (s - 5.0).abs() < f64::EPSILON));
    }

    #[test]
    fn config_default_mamba_130m() {
        let cfg = BottleneckConfig::default_mamba_130m();
        assert_eq!(cfg.layer_index, 20);
        assert_eq!(cfg.d_model, 768);
        assert_relative_eq!(cfg.channel_scales[0], 5.0);
    }

    #[test]
    fn config_identity_is_identity() {
        let cfg = BottleneckConfig::identity(20, 768);
        assert!(cfg.is_identity());
    }

    #[test]
    fn config_non_identity_is_not_identity() {
        let cfg = BottleneckConfig::uniform(20, 768, 5.0);
        assert!(!cfg.is_identity());
    }

    #[test]
    fn steering_identity_passthrough() {
        let cfg = BottleneckConfig::identity(20, 32);
        let steering = BottleneckSteering::new(cfg, &Device::Cpu).unwrap();
        let xs = Tensor::ones((1, 32), DType::F32, &Device::Cpu).unwrap();

        let result = steering.apply(&xs).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let xs_vec: Vec<f32> = xs.flatten_all().unwrap().to_vec1().unwrap();

        for (r, x) in result_vec.iter().zip(xs_vec.iter()) {
            assert_relative_eq!(*r, *x, epsilon = 1e-6);
        }
    }

    #[test]
    fn steering_uniform_5x_scales_correctly() {
        let cfg = BottleneckConfig::uniform(20, 32, 5.0);
        let steering = BottleneckSteering::new(cfg, &Device::Cpu).unwrap();

        let vals: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let xs = Tensor::new(&vals[..], &Device::Cpu).unwrap().unsqueeze(0).unwrap();

        let result = steering.apply(&xs).unwrap();
        let result_vec: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        for (i, &r) in result_vec.iter().enumerate() {
            assert_relative_eq!(r, (i + 1) as f32 * 5.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn steering_output_shape_matches_input() {
        let cfg = BottleneckConfig::uniform(20, 64, 3.0);
        let steering = BottleneckSteering::new(cfg, &Device::Cpu).unwrap();

        let xs = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let result = steering.apply(&xs).unwrap();
        assert_eq!(result.dims(), xs.dims());
    }

    #[test]
    fn should_steer_matches_layer() {
        let cfg = BottleneckConfig::uniform(20, 32, 5.0);
        let steering = BottleneckSteering::new(cfg, &Device::Cpu).unwrap();

        assert!(steering.should_steer(20));
        assert!(!steering.should_steer(19));
        assert!(!steering.should_steer(21));
    }

    #[test]
    fn should_steer_false_for_identity() {
        let cfg = BottleneckConfig::identity(20, 32);
        let steering = BottleneckSteering::new(cfg, &Device::Cpu).unwrap();

        assert!(!steering.should_steer(20));
    }
}
