//! CognitiveGate — learnable cognitive modulation layer for Mamba.
//!
//! Brain analog: thalamo-cortical relay + locus coeruleus gain control.
//! The thalamus reads cortical state to route information, while LC-NE
//! produces gain signals that modulate downstream cortical processing.
//! CognitiveGate combines both: it reads hidden state via learned
//! projections and produces modulation signals that change how subsequent
//! layers process information.
//!
//! This is Tầng 3 (Architecture Integration) — the gate is a NATIVE LAYER
//! in the model, not an external computation. Parameters are learned
//! end-to-end via backpropagation. The gate internalizes what Tầng 2's
//! external `delta_modulation.rs` does with hand-designed heuristics.
//!
//! Key papers:
//! - Crick 1984 (thalamic reticular complex as attentional searchlight)
//! - Aston-Jones & Cohen 2005 (LC-NE gain: phasic/tonic modulation)
//! - Marder 2012 (neuromodulatory space is low-dimensional)
//! - Vaswani 2017 (residual connections preserve gradient flow)
//! - Gu & Dao 2023 (Mamba: selective state spaces)
//! - HiSPA 2026 (mid-layers = critical transport corridor)
//!
//! Safe default: W_gate bias = -3.0 -> sigmoid ~ 0.05 -> near-passthrough.
//! Untrained gate barely modifies hidden state. Training teaches WHEN to modulate.
//!
//! Requires `candle` feature flag.

use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::inference::mamba::{MambaConfig, RmsNorm};

// ─── Helpers ───────────────────────────────────────────────────────

/// Standard sigmoid: 1 / (1 + exp(-x)). Operates element-wise on tensors.
/// Brain analog: sigmoidal activation of neural populations (ubiquitous).
fn sigmoid(xs: &Tensor) -> CandleResult<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}

// ─── Constants ─────────────────────────────────────────────────────

/// Default cognitive dimension — low-dimensional bottleneck.
/// Brain has ~6 major neuromodulatory systems (NE, DA, 5-HT, ACh,
/// histamine, orexin), not 768 independent channels. 64 dims captures
/// this bottleneck while allowing richer representation than 6.
/// (Marder 2012: neuromodulatory space is low-dimensional)
const DEFAULT_COGNITIVE_DIM: usize = 64;

/// Initial bias for W_gate, ensuring near-passthrough at initialization.
/// sigmoid(-3.0) ~ 0.047 -> untrained gate applies only ~5% cognitive
/// modulation. This is the critical safety property: model behaves
/// normally until fine-tuning teaches the gate when to activate.
/// (Vaswani 2017: residual connections with identity init preserve pretrained behavior)
const GATE_INIT_BIAS: f64 = -3.0;

/// Default gate position as fraction of model depth.
/// Mid-depth (~50%) is where Tầng 2 proved delta modulation most effective.
/// (HiSPA 2026: blocks at 44-58% depth show highest correlation with downstream behavior)
const DEFAULT_GATE_DEPTH_FRACTION: f64 = 0.50;

/// Minimum gain factor from gate output.
/// Same safety bounds as Tầng 2 delta modulation.
/// (Mamba Modulation NeurIPS 2025: uniform >2x scaling = catastrophic)
pub(crate) const GATE_GAIN_MIN: f64 = 0.5;

/// Maximum gain factor from gate output.
/// Same safety bounds as Tầng 2 delta modulation.
/// (Mamba Modulation NeurIPS 2025: layer-selective 0.7-1.3x = safe regime)
pub(crate) const GATE_GAIN_MAX: f64 = 2.0;

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

/// Configuration for CognitiveGate layer placement and dimensions.
///
/// Separates gate hyperparameters from model weights so the same config
/// can be used across construction, testing, and fine-tuning.
#[derive(Debug, Clone)]
pub struct CognitiveGateConfig {
    /// Dimension of the cognitive bottleneck (default: 64).
    /// Brain analog: neuromodulatory systems are low-dimensional (~6 systems),
    /// but 64 allows richer learned representations.
    pub cognitive_dim: usize,
    /// Which layer index the gate is placed after (0-indexed).
    /// Layers 0..gate_position are pre-gate, gate_position..n_layer are post-gate.
    pub gate_position: usize,
    /// Model hidden dimension (d_model from MambaConfig). Residual stream width.
    /// Gate WRITES to this dimension (residual blend on xs).
    pub d_model: usize,
    /// SSM inner dimension (d_inner from MambaConfig). Typically d_model * 2.
    /// Gate READS from this dimension (SSM state, mean over D_STATE).
    /// Brain analog: persistent PFC activity dimension — richer than transient cortical output.
    /// (Goldman-Rakic 1995: working memory via sustained recurrent activity in PFC)
    pub d_inner: usize,
}

impl CognitiveGateConfig {
    /// Compute gate config for a given MambaConfig with default positioning.
    /// Places gate at ~50% depth (HiSPA 2026: critical transport corridor).
    pub fn from_mamba_config(config: &MambaConfig) -> Self {
        let gate_position =
            (config.n_layer as f64 * DEFAULT_GATE_DEPTH_FRACTION).round() as usize;
        Self {
            cognitive_dim: DEFAULT_COGNITIVE_DIM,
            gate_position,
            d_model: config.d_model,
            d_inner: config.d_inner(),
        }
    }

    /// Override gate position (layer index).
    pub fn with_position(mut self, position: usize) -> Self {
        self.gate_position = position;
        self
    }

    /// Override cognitive dimension.
    pub fn with_cognitive_dim(mut self, dim: usize) -> Self {
        self.cognitive_dim = dim;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════
// Gate Output
// ═══════════════════════════════════════════════════════════════════

/// Output from CognitiveGate forward pass.
///
/// Brain analog: LC-NE output after reading thalamic state —
/// a gain control signal plus the modulated cortical state.
/// Unlike external delta modulation (hand-designed), these signals
/// are LEARNED from data via backpropagation.
pub struct CognitiveGateOutput {
    /// The modulated hidden state tensor, same shape as input.
    /// Residual blend: (1 - alpha) * input + alpha * cognitive_contribution.
    pub modulated: Tensor,
    /// Learned delta gain for post-gate layers.
    /// Mapped from sigmoid output to [GATE_GAIN_MIN, GATE_GAIN_MAX].
    /// Replaces external DeltaModulation for layers after the gate.
    pub delta_gain: f64,
    /// Gate blend factor. 0 = passthrough, 1 = full cognitive modulation.
    /// Initialized near 0.05 (W_gate bias = -3.0 -> sigmoid ~ 0.047).
    pub gate_alpha: f64,
    /// Cognitive signal — what W_read extracted from hidden state.
    /// Shape: (batch, cognitive_dim). This is the gate's learned "reading"
    /// of the model's internal cognitive state.
    /// Brain analog: thalamic summary of cortical activation (Crick 1984).
    pub cog_signal: Tensor,
}

// ═══════════════════════════════════════════════════════════════════
// CognitiveGate — the learnable layer
// ═══════════════════════════════════════════════════════════════════

/// Learnable cognitive modulation layer inserted between Mamba blocks.
///
/// Brain analog: combined thalamic relay + LC-NE gain control.
///
/// **READS from SSM state** (persistent recurrent memory) — not residual stream.
/// SSM state differentiates content types much more than residual stream
/// (probe result: SSM sim 0.21-0.55 vs residual 0.57). This matches
/// neuroscience: thalamus reads sustained PFC activity, not transient firing.
/// (Goldman-Rakic 1995: working memory via persistent recurrent activity)
///
/// **WRITES to residual stream** (current processing path).
/// Gate modulates what flows to the next layers, not what's stored in memory.
/// Brain analog: neuromodulatory output modulates cortical feedforward processing.
///
/// Operations:
/// 1. **Read** — SSM state (mean over D_STATE) → W_read → cognitive signal
/// 2. **Modulate** — cognitive signal → delta_gain + gate_alpha
/// 3. **Write** — residual blend on hidden state (feedforward path)
///
/// Safe default: gate_alpha ~ 0.05 at initialization.
pub struct CognitiveGate {
    /// Read projection: d_inner -> cognitive_dim.
    /// Reads from SSM state (persistent memory), not residual stream.
    /// Brain analog: thalamus reads sustained PFC activity patterns.
    /// (Goldman-Rakic 1995, Primacy/Recency in Mamba 2025: sparse long-term channels)
    w_read: Linear,
    /// Delta gain projection: cognitive_dim -> 1.
    /// Outputs scalar controlling state update speed in post-gate layers.
    w_delta: Linear,
    /// Gate blend projection: cognitive_dim -> 1.
    /// Controls how much cognitive modulation is applied (0 = passthrough).
    /// Bias initialized to GATE_INIT_BIAS (-3.0) for safe near-passthrough default.
    w_gate: Linear,
    /// Write projection: cognitive_dim -> d_model.
    /// Produces cognitive contribution blended into residual stream (not SSM state).
    /// Brain analog: neuromodulatory output modulates cortical feedforward processing.
    w_write: Linear,
    /// RMSNorm on SSM state input before reading.
    /// Dimension: d_inner (SSM state width after mean over D_STATE).
    norm: RmsNorm,
    /// Config for this gate instance.
    config: CognitiveGateConfig,
}

impl CognitiveGate {
    /// Create gate from VarBuilder — works with VarMap (testing) or safetensors (fine-tuned).
    ///
    /// W_read: d_inner → cognitive_dim (reads SSM state).
    /// W_write: cognitive_dim → d_model (writes to residual stream).
    /// W_gate bias = GATE_INIT_BIAS (-3.0) for safe near-passthrough.
    pub fn new(config: CognitiveGateConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_model = config.d_model;
        let d_inner = config.d_inner;
        let cog_dim = config.cognitive_dim;

        // Read: d_inner -> cognitive_dim (reads SSM state, not residual stream).
        let w_read = linear_no_bias(d_inner, cog_dim, vb.pp("w_read"))?;

        // Delta: cognitive_dim -> 1 (with bias, default init).
        let w_delta_weight = vb.pp("w_delta").get_with_hints(
            (1, cog_dim),
            "weight",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let w_delta_bias = vb.pp("w_delta").get_with_hints(
            1,
            "bias",
            candle_nn::init::Init::Const(0.0),
        )?;
        let w_delta = Linear::new(w_delta_weight, Some(w_delta_bias));

        // Gate: cognitive_dim -> 1 (bias = GATE_INIT_BIAS for safe passthrough).
        let w_gate_weight = vb.pp("w_gate").get_with_hints(
            (1, cog_dim),
            "weight",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let w_gate_bias = vb.pp("w_gate").get_with_hints(
            1,
            "bias",
            candle_nn::init::Init::Const(GATE_INIT_BIAS),
        )?;
        let w_gate = Linear::new(w_gate_weight, Some(w_gate_bias));

        // Write: cognitive_dim -> d_model (writes to residual stream).
        let w_write = linear_no_bias(cog_dim, d_model, vb.pp("w_write"))?;

        // Norm on SSM state input (d_inner dimension).
        let norm = RmsNorm::new(d_inner, vb.pp("norm"))?;

        Ok(Self {
            w_read,
            w_delta,
            w_gate,
            w_write,
            norm,
            config,
        })
    }

    /// Forward pass: read SSM state, compute modulation, write to residual stream.
    ///
    /// `ssm_state`: SSM hidden state, mean over D_STATE. Shape: (batch, d_inner).
    ///   This is the persistent recurrent memory — content-differentiating signal.
    /// `hidden_state`: Residual stream. Shape: (batch, d_model).
    ///   This is the current processing path — gate writes modulation here.
    ///
    /// Brain analog: thalamus reads sustained PFC activity (ssm_state) →
    /// LC-NE computes gain → neuromodulatory output modulates cortical
    /// feedforward processing (hidden_state).
    pub fn forward(
        &self,
        ssm_state: &Tensor,
        hidden_state: &Tensor,
    ) -> CandleResult<CognitiveGateOutput> {
        // Step 1: Normalize SSM state input.
        let normed = self.norm.forward(ssm_state)?;

        // Step 2: Read — project SSM state to cognitive dimension.
        // W_read extracts cognitive signal from persistent memory.
        let cog_signal = self.w_read.forward(&normed)?;

        // Step 3a: Compute delta_gain from cognitive signal.
        let delta_raw = sigmoid(&self.w_delta.forward(&cog_signal)?)?;

        // Step 3b: Compute gate_alpha from cognitive signal.
        let alpha_raw = sigmoid(&self.w_gate.forward(&cog_signal)?)?;

        // Step 4: Compute cognitive contribution → d_model (residual stream width).
        let cog_contribution = self.w_write.forward(&cog_signal)?;

        // Step 5: Residual blend on hidden_state (residual stream, not SSM state).
        // modulated = (1 - alpha) * hidden_state + alpha * cog_contribution
        let one_minus_alpha = (Tensor::ones_like(&alpha_raw)? - &alpha_raw)?;
        let modulated = (hidden_state.broadcast_mul(&one_minus_alpha)?
            + cog_contribution.broadcast_mul(&alpha_raw)?)?;

        // Step 6: Extract scalar values for downstream use.
        let delta_scalar = delta_raw
            .flatten_all()?
            .mean(D::Minus1)?
            .to_scalar::<f32>()? as f64;
        let alpha_scalar = alpha_raw
            .flatten_all()?
            .mean(D::Minus1)?
            .to_scalar::<f32>()? as f64;

        let delta_gain = GATE_GAIN_MIN + delta_scalar * (GATE_GAIN_MAX - GATE_GAIN_MIN);

        Ok(CognitiveGateOutput {
            modulated,
            delta_gain,
            gate_alpha: alpha_scalar,
            cog_signal,
        })
    }

    /// Gate position in the model (layer index).
    pub fn position(&self) -> usize {
        self.config.gate_position
    }

    /// Cognitive dimension (bottleneck size).
    pub fn cognitive_dim(&self) -> usize {
        self.config.cognitive_dim
    }

    /// Access W_read weight tensor for analysis/probing.
    /// Shape: (cognitive_dim, d_model). Each row = one learned cognitive dimension.
    /// Brain analog: thalamic relay's learned selectivity for cortical signals.
    pub fn w_read_weights(&self) -> &Linear {
        &self.w_read
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::{DType, Device};

    /// Helper: create a gate with VarMap (random weights, but correct W_gate bias).
    fn make_test_gate(d_model: usize, d_inner: usize, cognitive_dim: usize) -> CognitiveGate {
        let config = CognitiveGateConfig {
            cognitive_dim,
            gate_position: 2,
            d_model,
            d_inner,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        CognitiveGate::new(config, vb.pp("gate")).expect("Gate construction should succeed")
    }

    #[test]
    fn gate_passthrough_with_default_init() {
        // W_gate bias = -3.0 -> sigmoid ~ 0.047.
        // Gate should barely modify the residual stream at initialization.
        let gate = make_test_gate(32, 64, 8);
        let ssm_state = Tensor::ones((1, 64), DType::F32, &Device::Cpu).unwrap();
        let hidden = Tensor::ones((1, 32), DType::F32, &Device::Cpu).unwrap();

        let output = gate.forward(&ssm_state, &hidden).unwrap();

        assert!(
            output.gate_alpha < 0.15,
            "Gate alpha should be near 0 at init, got {}",
            output.gate_alpha
        );

        // Modulated output should be close to hidden_state (residual stream).
        let hidden_vec: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
        let output_vec: Vec<f32> = output.modulated.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(hidden_vec.len(), output_vec.len());

        let max_diff: f32 = hidden_vec
            .iter()
            .zip(output_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2.0,
            "Near-passthrough gate should not dramatically change residual, max_diff={}",
            max_diff
        );
    }

    #[test]
    fn gate_output_shapes_correct() {
        let gate = make_test_gate(32, 64, 8);
        let ssm_state = Tensor::randn(0f32, 1f32, (1, 64), &Device::Cpu).unwrap();
        let hidden = Tensor::randn(0f32, 1f32, (1, 32), &Device::Cpu).unwrap();

        let output = gate.forward(&ssm_state, &hidden).unwrap();

        // Modulated tensor should have same shape as hidden_state (d_model).
        assert_eq!(output.modulated.dims(), hidden.dims());
        // Cog signal should have cognitive_dim dimension.
        assert_eq!(output.cog_signal.dims(), &[1, 8]);
    }

    #[test]
    fn gate_config_from_mamba_config() {
        let mamba_cfg = MambaConfig::mamba_130m();
        let gate_cfg = CognitiveGateConfig::from_mamba_config(&mamba_cfg);

        assert_eq!(gate_cfg.d_model, 768);
        assert_eq!(gate_cfg.d_inner, 1536);
        assert_eq!(gate_cfg.cognitive_dim, 64);
        assert_eq!(gate_cfg.gate_position, 12);
    }

    #[test]
    fn gate_config_custom_position() {
        let mamba_cfg = MambaConfig::mamba_130m();
        let gate_cfg = CognitiveGateConfig::from_mamba_config(&mamba_cfg)
            .with_position(8)
            .with_cognitive_dim(32);

        assert_eq!(gate_cfg.gate_position, 8);
        assert_eq!(gate_cfg.cognitive_dim, 32);
        assert_eq!(gate_cfg.d_model, 768);
        assert_eq!(gate_cfg.d_inner, 1536); // Unchanged.
    }

    #[test]
    fn different_ssm_states_produce_different_readings() {
        let gate = make_test_gate(32, 64, 8);
        let hidden = Tensor::ones((1, 32), DType::F32, &Device::Cpu).unwrap();

        // Two SSM states with different PATTERNS (not just scale — RmsNorm erases scale).
        // Use explicit ascending vs descending patterns to guarantee pattern difference.
        let vals_a: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let vals_b: Vec<f32> = (0..64).map(|i| 1.0 - i as f32 / 64.0).collect();
        let ssm_a = Tensor::new(&vals_a[..], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let ssm_b = Tensor::new(&vals_b[..], &Device::Cpu).unwrap().unsqueeze(0).unwrap();

        let out_a = gate.forward(&ssm_a, &hidden).unwrap();
        let out_b = gate.forward(&ssm_b, &hidden).unwrap();

        // Check cog_signal (the gate's reading) — should differ for different SSM states.
        let sig_a: Vec<f32> = out_a.cog_signal.flatten_all().unwrap().to_vec1().unwrap();
        let sig_b: Vec<f32> = out_b.cog_signal.flatten_all().unwrap().to_vec1().unwrap();

        // First verify cog_signals are non-zero.
        let a_nonzero = sig_a.iter().any(|v| v.abs() > 1e-10);
        let b_nonzero = sig_b.iter().any(|v| v.abs() > 1e-10);
        assert!(a_nonzero, "cog_signal A should be non-zero, got: {:?}", &sig_a[..4.min(sig_a.len())]);
        assert!(b_nonzero, "cog_signal B should be non-zero, got: {:?}", &sig_b[..4.min(sig_b.len())]);

        let differs = sig_a.iter().zip(sig_b.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Different SSM states should produce different cognitive signals. A: {:?}, B: {:?}",
            &sig_a[..4.min(sig_a.len())], &sig_b[..4.min(sig_b.len())]
        );
    }

    #[test]
    fn gate_delta_gain_in_safe_range() {
        let gate = make_test_gate(32, 64, 8);
        let hidden = Tensor::ones((1, 32), DType::F32, &Device::Cpu).unwrap();

        for scale in [0.01, 0.1, 1.0, 5.0, 10.0] {
            let ssm = (Tensor::ones((1, 64), DType::F32, &Device::Cpu).unwrap() * scale).unwrap();
            let output = gate.forward(&ssm, &hidden).unwrap();

            assert!(
                output.delta_gain >= GATE_GAIN_MIN && output.delta_gain <= GATE_GAIN_MAX,
                "delta_gain {} outside [{}, {}] for scale {}",
                output.delta_gain, GATE_GAIN_MIN, GATE_GAIN_MAX, scale
            );
        }
    }

    #[test]
    fn gate_alpha_in_unit_range() {
        let gate = make_test_gate(32, 64, 8);
        let hidden = Tensor::ones((1, 32), DType::F32, &Device::Cpu).unwrap();

        for scale in [0.01, 1.0, 10.0] {
            let ssm = (Tensor::ones((1, 64), DType::F32, &Device::Cpu).unwrap() * scale).unwrap();
            let output = gate.forward(&ssm, &hidden).unwrap();

            assert!(
                output.gate_alpha >= 0.0 && output.gate_alpha <= 1.0,
                "gate_alpha {} not in [0, 1] for scale {}",
                output.gate_alpha, scale
            );
        }
    }

    #[test]
    fn gate_delta_gain_range_constants() {
        assert_relative_eq!(GATE_GAIN_MIN + 0.0 * (GATE_GAIN_MAX - GATE_GAIN_MIN), 0.5);
        assert_relative_eq!(GATE_GAIN_MIN + 1.0 * (GATE_GAIN_MAX - GATE_GAIN_MIN), 2.0);
        assert_relative_eq!(GATE_GAIN_MIN + 0.5 * (GATE_GAIN_MAX - GATE_GAIN_MIN), 1.25);
    }
}
