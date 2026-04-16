//! Cognitive Mamba Model — SSM with delta modulation injection.
//!
//! Brain analog: cortical tissue (Mamba layers) modulated by neuromodulators
//! (cognitive delta scaling). Hidden state flows sequentially through each
//! layer — Noos modulates delta (state update speed) at targeted mid-layers,
//! directly changing how the model processes information.
//!
//! Forked from candle-transformers mamba.rs with cognitive injection point.
//!
//! Key papers:
//! - Gu & Dao 2023 (Mamba: selective state spaces)
//! - Aston-Jones & Cohen 2005 (gain modulation → delta scaling)
//! - Hidden Attention of Mamba (ACL 2025: delta = implicit attention decay)
//! - HiSPA 2026 (mid-layers = critical transport corridor)
//!
//! Requires `candle` feature flag.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use crate::errors::{NoosError, NoosResult};
use crate::inference::bottleneck::BottleneckSteering;
use crate::inference::cognitive_model::CognitiveModel;
use crate::inference::model::LocalModel;
use crate::types::intervention::{DeltaModulation, ForwardResult, InterventionDepth};

// ─── Constants (Gu & Dao 2023) ─────────────────────────────────────
/// Causal convolution width — standard across all Mamba variants.
const D_CONV: usize = 4;
/// SSM state dimension — 16 for standard Mamba.
const D_STATE: usize = 16;

// ═══════════════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════════════

/// Mamba model configuration.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub pad_vocab_size_multiple: usize,
}

impl MambaConfig {
    /// SSM inner dimension — d_model * 2 (Mamba expansion factor, Gu & Dao 2023).
    pub fn d_inner(&self) -> usize {
        self.d_model * 2
    }

    fn dt_rank(&self) -> usize {
        self.d_model.div_ceil(16)
    }

    fn padded_vocab_size(&self) -> usize {
        let multiple = self.pad_vocab_size_multiple;
        ((self.vocab_size + multiple - 1) / multiple) * multiple
    }

    /// Config for state-spaces/mamba-130m.
    pub fn mamba_130m() -> Self {
        Self {
            d_model: 768,
            n_layer: 24,
            vocab_size: 50280,
            pad_vocab_size_multiple: 8,
        }
    }

    /// Config for state-spaces/mamba-370m.
    pub fn mamba_370m() -> Self {
        Self {
            d_model: 1024,
            n_layer: 48,
            vocab_size: 50280,
            pad_vocab_size_multiple: 8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// State — SSM hidden state + conv buffer
// ═══════════════════════════════════════════════════════════════════

/// Mamba inference state — SSM hidden states + causal conv buffers.
///
/// Brain analog: persistent neural activation patterns across cortical layers.
/// Each layer maintains its own state (no cross-layer state contamination).
pub struct MambaState {
    /// Per-layer SSM hidden states. Shape per layer: (batch, d_inner, D_STATE).
    hs: Vec<Tensor>,
    /// Per-layer conv1d circular buffers. Shape per layer per slot: (batch, d_inner).
    prev_xs: Vec<[Tensor; D_CONV]>,
    /// Current sequence position.
    pos: usize,
}

impl MambaState {
    pub fn new(config: &MambaConfig, batch_size: usize, device: &Device) -> CandleResult<Self> {
        let d_inner = config.d_inner();
        let n_layer = config.n_layer;

        let mut hs = Vec::with_capacity(n_layer);
        let mut prev_xs = Vec::with_capacity(n_layer);

        for _ in 0..n_layer {
            hs.push(Tensor::zeros((batch_size, d_inner, D_STATE), DType::F32, device)?);
            // P5: explicit slot creation with error propagation (no unwrap/expect).
            let s0 = Tensor::zeros((batch_size, d_inner), DType::F32, device)?;
            let s1 = Tensor::zeros((batch_size, d_inner), DType::F32, device)?;
            let s2 = Tensor::zeros((batch_size, d_inner), DType::F32, device)?;
            let s3 = Tensor::zeros((batch_size, d_inner), DType::F32, device)?;
            prev_xs.push([s0, s1, s2, s3]);
        }

        Ok(Self {
            hs,
            prev_xs,
            pos: 0,
        })
    }

    /// Mutable: clears all per-layer SSM hidden states and conv buffers.
    /// Requires mutation because SSM state accumulates across tokens
    /// and must be wiped between conversations to prevent cross-contamination.
    pub fn reset(&mut self, config: &MambaConfig, device: &Device) -> CandleResult<()> {
        let d_inner = config.d_inner();
        for li in 0..config.n_layer {
            self.hs[li] = Tensor::zeros((1, d_inner, D_STATE), DType::F32, device)?;
            for slot in 0..D_CONV {
                self.prev_xs[li][slot] = Tensor::zeros((1, d_inner), DType::F32, device)?;
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// Read SSM hidden state for a specific layer (for probing/analysis).
    ///
    /// Brain analog: reading persistent neural activation in a cortical layer.
    /// Shape: (batch, d_inner, D_STATE). Returns None if layer index out of range.
    pub fn layer_hidden_state(&self, layer: usize) -> Option<&Tensor> {
        self.hs.get(layer)
    }

    /// Number of layers with stored hidden state.
    pub fn num_layers(&self) -> usize {
        self.hs.len()
    }
}

/// Compute hidden state statistics from SSM state at target layers.
///
/// Reads hs from layers in the delta modulation target range (40-60% depth).
/// Computes state churn by comparing current hs against a previous snapshot.
///
/// Brain analog: LC reads aggregate cortical state change (Grella 2024).
/// State churn = unsigned PE = how much the model's internal state changed.
///
/// `state`: current MambaState after forward pass for token t.
/// `prev_hs_snapshot`: mean-pooled hs from previous token (None on first token).
/// `target`: which layers to read (reuses delta_modulation layer targeting).
///
/// Returns: `(HiddenStateStats, snapshot)` — stats for downstream arousal
/// computation, and new snapshot to store for the next token's churn calc.
pub fn compute_hs_stats(
    state: &MambaState,
    prev_hs_snapshot: Option<&[f64]>,
    target: &crate::types::intervention::LayerTarget,
) -> CandleResult<(crate::types::intervention::HiddenStateStats, Vec<f64>)> {
    use candle_core::D;

    let mut current_snapshot = Vec::new();
    let mut layer_count = 0usize;
    let end = target.end_layer.min(state.num_layers().saturating_sub(1));

    // Collect mean-pooled hs across target layers.
    // Each hs[layer] shape: (batch=1, d_inner, D_STATE).
    // Mean over D_STATE → (1, d_inner), flatten → Vec<f32>.
    for layer_idx in target.start_layer..=end {
        if let Some(hs_tensor) = state.layer_hidden_state(layer_idx) {
            let layer_mean = hs_tensor.mean(D::Minus1)?;
            let vals: Vec<f32> = layer_mean.flatten_all()?.to_vec1()?;
            current_snapshot.extend(vals.iter().map(|v| *v as f64));
            layer_count += 1;
        }
    }

    if current_snapshot.is_empty() || layer_count == 0 {
        return Ok((
            crate::types::intervention::HiddenStateStats::default(),
            current_snapshot,
        ));
    }

    // State magnitude: L2 norm of current snapshot, normalized per layer.
    let magnitude: f64 = current_snapshot.iter().map(|v| v * v).sum::<f64>().sqrt()
        / layer_count as f64;

    // State churn: ||current - previous|| / ||current||.
    // High churn = lots of state change = needs compensatory retention.
    let (churn, valid) = match prev_hs_snapshot {
        Some(prev) if prev.len() == current_snapshot.len() && magnitude > 1e-10 => {
            let diff_norm: f64 = current_snapshot
                .iter()
                .zip(prev.iter())
                .map(|(c, p)| (c - p) * (c - p))
                .sum::<f64>()
                .sqrt()
                / layer_count as f64;
            (diff_norm / magnitude, true)
        }
        _ => (0.0, false),
    };

    Ok((
        crate::types::intervention::HiddenStateStats {
            state_churn: churn,
            state_magnitude: magnitude,
            valid,
        },
        current_snapshot,
    ))
}

// ═══════════════════════════════════════════════════════════════════
// MambaBlock — single SSM layer with cognitive delta injection
// ═══════════════════════════════════════════════════════════════════

/// A single Mamba block with optional cognitive delta modulation.
///
/// Brain analog: one cortical layer. Delta (dt) controls how fast the layer's
/// hidden state updates — analogous to noradrenergic gain control within
/// a single cortical area.
struct CognitiveMambaBlock {
    in_proj: Linear,
    conv1d_bias: Tensor,
    conv1d_weights: [Tensor; D_CONV],
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    dt_rank: usize,
    layer_index: usize,
    d_inner: usize,
}

impl CognitiveMambaBlock {
    fn new(layer_index: usize, config: &MambaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d_inner = config.d_inner();
        let dt_rank = config.dt_rank();

        let in_proj = linear_no_bias(config.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + D_STATE * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let out_proj = linear_no_bias(d_inner, config.d_model, vb.pp("out_proj"))?;

        // A_log initialized to log of uniform — matches Mamba reference impl.
        // D initialized to ones (Gu & Dao 2023 default).
        // conv1d uses Kaiming init (standard for 1D conv layers).
        // These inits are only used with VarMap; safetensors loading ignores them.
        let a_log = vb.get_with_hints(
            (d_inner, D_STATE),
            "A_log",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let d = vb.get_with_hints(d_inner, "D", candle_nn::init::Init::Const(1.0))?;

        let conv1d_weight = vb.get_with_hints(
            (d_inner, 1, D_CONV),
            "conv1d.weight",
            candle_nn::init::DEFAULT_KAIMING_UNIFORM,
        )?;
        let conv1d_bias = vb.get_with_hints(
            d_inner,
            "conv1d.bias",
            candle_nn::init::Init::Const(0.0),
        )?;

        // P5: extract conv slices with error propagation.
        let conv1d_w0 = conv1d_weight.i((.., 0, 0))?;
        let conv1d_w1 = conv1d_weight.i((.., 0, 1))?;
        let conv1d_w2 = conv1d_weight.i((.., 0, 2))?;
        let conv1d_w3 = conv1d_weight.i((.., 0, 3))?;
        let conv1d_weights = [conv1d_w0, conv1d_w1, conv1d_w2, conv1d_w3];

        Ok(Self {
            in_proj,
            conv1d_bias,
            conv1d_weights,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            dt_rank,
            layer_index,
            d_inner,
        })
    }

    /// Forward pass with optional cognitive delta modulation.
    ///
    /// `gain_factor`: if Some, multiplies delta before softplus.
    /// > 1.0 = attend more to current input (phasic gain).
    /// < 1.0 = preserve more history (tonic gain).
    /// 1.0 or None = standard unmodulated behavior.
    fn forward(
        &self,
        xs: &Tensor,
        state: &mut MambaState,
        gain_factor: Option<f64>,
    ) -> CandleResult<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;
        let li = self.layer_index;

        // in_proj splits into conv branch + gate branch.
        let mut xs = xs.apply(&self.in_proj)?.chunk(2, D::Minus1)?;
        let proj_for_silu = xs.remove(1);
        state.prev_xs[li][state.pos % D_CONV] = xs.remove(0);

        // Causal conv1d via circular buffer.
        let mut proj_for_conv = self.conv1d_bias.broadcast_as((b_sz, self.d_inner))?;
        for d_c in 0..D_CONV {
            proj_for_conv = (proj_for_conv
                + self.conv1d_weights[d_c]
                    .broadcast_mul(&state.prev_xs[li][(d_c + 1 + state.pos) % D_CONV])?)?;
        }
        let proj_for_conv = candle_nn::ops::silu(&proj_for_conv)?;

        // x_proj → delta (dt_rank), B (D_STATE), C (D_STATE).
        let x_proj = self.x_proj.forward(&proj_for_conv)?;
        let delta = x_proj.narrow(D::Minus1, 0, self.dt_rank)?.contiguous()?;
        let b = x_proj.narrow(D::Minus1, self.dt_rank, D_STATE)?;
        let c = x_proj.narrow(D::Minus1, self.dt_rank + D_STATE, D_STATE)?;

        // dt_proj: expand delta from dt_rank → d_inner.
        let delta = delta.apply(&self.dt_proj)?;

        // ━━━ COGNITIVE DELTA MODULATION (Tầng 2) ━━━━━━━━━━━━━━━━━━━
        // Injection point: after dt_proj linear, before softplus.
        // delta shape: (batch, d_inner).
        //
        // gain > 1.0 → larger delta → exp(delta*A) decays more →
        //   model forgets old state faster, attends to new input (phasic).
        // gain < 1.0 → smaller delta → state persists more →
        //   model preserves history, broad attention (tonic).
        //
        // Brain analog: NE gain control on cortical circuits (Aston-Jones 2005).
        // Math proof: delta scaling = implicit attention temporal decay (ACL 2025).
        let delta = if let Some(gain) = gain_factor {
            if (gain - 1.0).abs() > f64::EPSILON {
                (&delta * gain)?
            } else {
                delta
            }
        } else {
            delta
        };

        // softplus: delta = log(exp(delta) + 1).
        let delta = (delta.exp()? + 1.)?.log()?;

        let a = self.a_log.to_dtype(delta.dtype())?.exp()?.neg()?;
        let d = self.d.to_dtype(delta.dtype())?;

        // Selective scan: h_t = exp(delta * A) * h_{t-1} + delta * B * x_t.
        let delta = delta
            .unsqueeze(D::Minus1)?
            .broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let a = a.broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let b = b.unsqueeze(1)?
            .broadcast_as((b_sz, self.d_inner, D_STATE))?;
        let proj_for_conv_b = proj_for_conv
            .unsqueeze(D::Minus1)?
            .broadcast_as((b_sz, self.d_inner, D_STATE))?;

        state.hs[li] =
            ((&state.hs[li] * (&delta * &a)?.exp()?)? + &delta * &b * &proj_for_conv_b)?;

        // Output: y = C * h + D * x.
        let ss = (state.hs[li]
            .matmul(&c.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?
            + proj_for_conv.broadcast_mul(&d)?)?;

        // Gated output: y * SiLU(gate).
        let ys = (ss * candle_nn::ops::silu(&proj_for_silu))?;
        ys.apply(&self.out_proj)
    }
}

// ═══════════════════════════════════════════════════════════════════
// ResidualBlock — norm + mamba block + residual
// ═══════════════════════════════════════════════════════════════════

/// RMSNorm — standard normalization for Mamba models.
///
/// Used by both CognitiveMambaModel and CognitiveGate (hence pub(crate)).
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn new(d_model: usize, vb: VarBuilder) -> CandleResult<Self> {
        // Init to 1.0 — standard for RmsNorm (matches PyTorch default).
        // vb.get() uses Init::Const(0.0) for VarMap, which would zero out everything.
        let weight = vb.get_with_hints(d_model, "weight", candle_nn::init::Init::Const(1.0))?;
        Ok(Self { weight, eps: 1e-5 })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs_normed.broadcast_mul(&self.weight)
    }
}

/// Residual block: RMSNorm → MambaBlock → residual add.
struct ResidualBlock {
    mixer: CognitiveMambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    fn new(layer_index: usize, config: &MambaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let mixer = CognitiveMambaBlock::new(layer_index, config, vb.pp("mixer"))?;
        let norm = RmsNorm::new(config.d_model, vb.pp("norm"))?;
        Ok(Self { mixer, norm })
    }

    fn forward(
        &self,
        xs: &Tensor,
        state: &mut MambaState,
        gain_factor: Option<f64>,
        bottleneck: Option<&BottleneckSteering>,
    ) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let xs = self.mixer.forward(&xs, state, gain_factor)?;
        // ━━━ BOTTLENECK STEERING (Tầng 4) ━━━━━━━━━━━━━━━━━━━━━━━━
        // At the identified bottleneck layer, scale mixer output channels
        // to widen the information routing constraint.
        // (Mohan et al. 2026: Layer 20, KL=813, scale 5× → +8.27%)
        let xs = if let Some(steering) = bottleneck {
            steering.apply(&xs)?
        } else {
            xs
        };
        residual + xs
    }
}

// ═══════════════════════════════════════════════════════════════════
// CognitiveMambaModel — full model implementing CognitiveModel
// ═══════════════════════════════════════════════════════════════════

/// Complete Mamba model with cognitive delta modulation.
///
/// Brain analog: the full cortical hierarchy. Input tokens → embeddings →
/// sequential processing through layers (with NE modulation at mid-layers) →
/// output logits. The model IS cognitive — delta modulation is not an add-on
/// but part of how the model processes information.
pub struct CognitiveMambaModel {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    config: MambaConfig,
    state: MambaState,
    device: Device,
    /// Optional bottleneck steering (Tầng 4).
    /// Scales mixer output at the identified bottleneck layer.
    bottleneck: Option<BottleneckSteering>,
}

impl CognitiveMambaModel {
    /// Load model from pretrained weights.
    pub fn new(config: MambaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();
        let padded_vocab = config.padded_vocab_size();

        // HF Mamba uses "backbone.embeddings" (plural 's').
        let embedding = candle_nn::embedding(padded_vocab, config.d_model, vb.pp("backbone.embeddings"))?;

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            layers.push(ResidualBlock::new(
                i,
                &config,
                vb.pp(format!("backbone.layers.{i}")),
            )?);
        }

        let norm_f = RmsNorm::new(config.d_model, vb.pp("backbone.norm_f"))?;
        // lm_head: try dedicated weights first, fall back to tied embedding weights.
        // Many Mamba models tie lm_head.weight = embedding.weight to save parameters.
        let lm_head = match linear_no_bias(config.d_model, padded_vocab, vb.pp("lm_head")) {
            Ok(head) => head,
            Err(_) => {
                // Weight tying: reuse embedding weight as lm_head.
                let emb_weight = embedding.embeddings();
                Linear::new(emb_weight.clone(), None)
            }
        };

        let state = MambaState::new(&config, 1, &device)?;

        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
            config,
            state,
            device,
            bottleneck: None,
        })
    }

    /// Set bottleneck steering (Tầng 4).
    /// Scales mixer output at the specified layer to compensate routing bottleneck.
    pub fn set_bottleneck(&mut self, steering: BottleneckSteering) {
        self.bottleneck = Some(steering);
    }

    /// Remove bottleneck steering (return to baseline).
    pub fn clear_bottleneck(&mut self) {
        self.bottleneck = None;
    }

    /// Access SSM state for reading per-layer hidden states after forward pass.
    /// Used by `compute_hs_stats()` to extract cognitive signals from model state.
    pub fn state(&self) -> &MambaState {
        &self.state
    }

    /// Mutable: advances SSM state position and updates per-layer hidden states.
    /// Requires mutation because SSM state accumulates sequentially across tokens
    /// (each token's hidden state depends on all previous tokens via state.hs).
    ///
    /// Applies delta modulation gain_factor at layers within target range.
    /// Layers outside target range run standard (unmodulated) forward.
    fn forward_with_gains(
        &mut self,
        token_id: u32,
        delta_modulation: Option<&DeltaModulation>,
    ) -> CandleResult<Vec<f32>> {
        let token = Tensor::new(&[token_id], &self.device)?;
        let mut xs = self.embedding.forward(&token)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let gain = delta_modulation
                .filter(|dm| dm.target.contains(i) && (dm.gain_factor - 1.0).abs() > f64::EPSILON)
                .map(|dm| dm.gain_factor);

            let steer = self.bottleneck.as_ref().filter(|b| b.should_steer(i));
            xs = layer.forward(&xs, &mut self.state, gain, steer)?;
        }

        self.state.pos += 1;

        let xs = self.norm_f.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;

        // Extract logits as Vec<f32>, truncated to actual vocab_size.
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        Ok(logits_vec[..self.config.vocab_size].to_vec())
    }

    /// Forward one token and capture mixer output at a target layer.
    ///
    /// The mixer output = residual_after - residual_before at the target layer.
    /// Shape: (d_model,). Used for bottleneck calibration (channel variance analysis).
    ///
    /// Returns: (logits, mixer_output_at_target_layer).
    pub fn forward_capture_mixer(
        &mut self,
        token_id: u32,
        target_layer: usize,
    ) -> CandleResult<(Vec<f32>, Vec<f32>)> {
        let token = Tensor::new(&[token_id], &self.device)?;
        let mut xs = self.embedding.forward(&token)?;
        let mut mixer_output = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let xs_before = if i == target_layer {
                Some(xs.clone())
            } else {
                None
            };
            xs = layer.forward(&xs, &mut self.state, None, None)?;
            if let Some(before) = xs_before {
                // mixer_output = xs_after - xs_before (the residual branch contribution)
                let diff = (&xs - &before)?;
                mixer_output = diff.squeeze(0)?.to_vec1()?;
            }
        }

        self.state.pos += 1;
        let xs = self.norm_f.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        Ok((logits_vec[..self.config.vocab_size].to_vec(), mixer_output))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Trait implementations
// ═══════════════════════════════════════════════════════════════════

impl LocalModel for CognitiveMambaModel {
    fn forward(&mut self, tokens: &[u32], _position: usize) -> NoosResult<Vec<f32>> {
        // For prompt processing: forward each token sequentially (fill SSM state).
        // For generation: only the last token matters.
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self
                .forward_with_gains(token, None)
                .map_err(|e| NoosError::Internal(format!("Mamba forward error: {e}")))?;
        }
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn reset_cache(&mut self) {
        // Reset SSM state for new conversation.
        let _ = self.state.reset(&self.config, &self.device);
    }
}

impl CognitiveModel for CognitiveMambaModel {
    fn intervention_depth(&self) -> InterventionDepth {
        InterventionDepth::ActivationAccess
    }

    fn forward_cognitive(
        &mut self,
        tokens: &[u32],
        _position: usize,
        delta_modulation: &DeltaModulation,
    ) -> NoosResult<ForwardResult> {
        let mut logits = Vec::new();

        // Forward each token with cognitive delta modulation.
        for &token in tokens {
            logits = self
                .forward_with_gains(token, Some(delta_modulation))
                .map_err(|e| NoosError::Internal(format!("Mamba cognitive forward error: {e}")))?;
        }

        // Determine which layers were actually modulated.
        let modulated_layers: Vec<usize> = (0..self.config.n_layer)
            .filter(|&i| {
                delta_modulation.target.contains(i)
                    && (delta_modulation.gain_factor - 1.0).abs() > f64::EPSILON
            })
            .collect();

        let modulation_applied = !modulated_layers.is_empty();

        Ok(ForwardResult {
            logits,
            modulation_applied,
            modulated_layers,
            applied_gain_factor: delta_modulation.gain_factor,
            gate_delta_gain: None,
            gate_alpha: None,
            hs_stats: None,
        })
    }

    fn num_layers(&self) -> usize {
        self.config.n_layer
    }
}

// ═══════════════════════════════════════════════════════════════════
// Model Loading — HuggingFace Hub integration
// ═══════════════════════════════════════════════════════════════════

impl CognitiveMambaModel {
    /// Load a pretrained Mamba model from HuggingFace Hub.
    ///
    /// Downloads safetensors weights and tokenizer config automatically.
    /// `model_id`: HuggingFace model ID (e.g., "state-spaces/mamba-130m").
    pub fn from_pretrained(model_id: &str, config: MambaConfig) -> NoosResult<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| NoosError::Internal(format!("HF Hub API init error: {e}")))?;
        let repo = api.model(model_id.to_string());

        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| NoosError::Internal(format!("Failed to download weights: {e}")))?;

        let device = Device::Cpu;
        // P5 exception: candle's mmap API requires unsafe because the memory-mapped file
        // could be modified by another process during use. This is the standard candle
        // pattern for loading safetensors and is localized to this single call.
        // Alternative (safe but slower): candle_core::safetensors::load() copies into memory.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| NoosError::Internal(format!("Failed to load safetensors: {e}")))?
        };

        Self::new(config, vb)
            .map_err(|e| NoosError::Internal(format!("Failed to construct model: {e}")))
    }
}

/// HuggingFace tokenizer wrapper implementing NoosTokenizer.
///
/// Wraps the `tokenizers` crate for text ↔ token conversion.
/// Brain analog: sensory transduction — converting raw input (text)
/// into neural codes (token IDs) that cortical tissue can process.
pub struct HfTokenizer {
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: u32,
}

impl HfTokenizer {
    /// Load tokenizer from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> NoosResult<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| NoosError::Internal(format!("HF Hub API init error: {e}")))?;
        let repo = api.model(model_id.to_string());

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| NoosError::Internal(format!("Failed to download tokenizer: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| NoosError::Internal(format!("Failed to load tokenizer: {e}")))?;

        // EOS token: GPT-NeoX uses <|endoftext|> (token 0).
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(0);

        Ok(Self {
            tokenizer,
            eos_token_id,
        })
    }
}

impl crate::inference::tokenizer::NoosTokenizer for HfTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> NoosResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| NoosError::Internal(format!("Tokenization error: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> NoosResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| NoosError::Internal(format!("Decode error: {e}")))
    }

    fn decode_token(&self, token: u32) -> NoosResult<String> {
        self.decode(&[token])
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

// ═══════════════════════════════════════════════════════════════════
// CognitiveMambaWithGate — Tầng 3 architecture integration
// ═══════════════════════════════════════════════════════════════════

use crate::inference::cognitive_gate::{CognitiveGate, CognitiveGateConfig};

/// Mamba model with integrated CognitiveGate — Tầng 3 architecture.
///
/// Brain analog: cortical hierarchy with embedded thalamic relay station.
/// Pre-gate layers process input normally (with optional external delta
/// modulation from Tầng 2). CognitiveGate reads the cortical state at
/// mid-depth, then post-gate layers receive the gate's learned delta_gain.
///
/// Key insight: Tầng 2 (external) and Tầng 3 (internal) coexist.
/// - Pre-gate layers: external DeltaModulation from cognitive state (Tầng 2)
/// - CognitiveGate: reads hidden state, produces learned modulation
/// - Post-gate layers: gate's learned delta_gain (Tầng 3)
/// - Sampling: CognitiveSampler still applies (Tầng 1)
///
/// All three tầng stack — they don't replace each other.
///
/// Key papers:
/// - Crick 1984 (thalamic gating in cortical hierarchy)
/// - Aston-Jones & Cohen 2005 (LC-NE gain at multiple cortical levels)
/// - HiSPA 2026 (mid-depth layers = critical transport corridor)
pub struct CognitiveMambaWithGate {
    embedding: candle_nn::Embedding,
    pre_gate_layers: Vec<ResidualBlock>,
    cognitive_gate: CognitiveGate,
    post_gate_layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    config: MambaConfig,
    gate_config: CognitiveGateConfig,
    state: MambaState,
    device: Device,
    /// Optional bottleneck steering (Tầng 4).
    bottleneck: Option<BottleneckSteering>,
}

impl CognitiveMambaWithGate {
    /// Create model with cognitive gate from VarBuilder.
    ///
    /// Splits layers at gate_position: layers 0..gate_pos are pre-gate,
    /// layers gate_pos..n_layer are post-gate. CognitiveGate sits between.
    pub fn new(
        config: MambaConfig,
        gate_config: CognitiveGateConfig,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let device = vb.device().clone();
        let padded_vocab = config.padded_vocab_size();

        let embedding =
            candle_nn::embedding(padded_vocab, config.d_model, vb.pp("backbone.embeddings"))?;

        let gate_pos = gate_config.gate_position;

        // Pre-gate layers: 0..gate_pos.
        let mut pre_gate_layers = Vec::with_capacity(gate_pos);
        for i in 0..gate_pos {
            pre_gate_layers.push(ResidualBlock::new(
                i,
                &config,
                vb.pp(format!("backbone.layers.{i}")),
            )?);
        }

        // CognitiveGate — the learnable cognitive layer.
        let cognitive_gate =
            CognitiveGate::new(gate_config.clone(), vb.pp("backbone.cognitive_gate"))?;

        // Post-gate layers: gate_pos..n_layer.
        let mut post_gate_layers = Vec::with_capacity(config.n_layer - gate_pos);
        for i in gate_pos..config.n_layer {
            post_gate_layers.push(ResidualBlock::new(
                i,
                &config,
                vb.pp(format!("backbone.layers.{i}")),
            )?);
        }

        let norm_f = RmsNorm::new(config.d_model, vb.pp("backbone.norm_f"))?;
        let lm_head = match linear_no_bias(config.d_model, padded_vocab, vb.pp("lm_head")) {
            Ok(head) => head,
            Err(_) => {
                let emb_weight = embedding.embeddings();
                Linear::new(emb_weight.clone(), None)
            }
        };

        let state = MambaState::new(&config, 1, &device)?;

        Ok(Self {
            embedding,
            pre_gate_layers,
            cognitive_gate,
            post_gate_layers,
            norm_f,
            lm_head,
            config,
            gate_config,
            state,
            device,
            bottleneck: None,
        })
    }

    /// Set bottleneck steering (Tầng 4).
    pub fn set_bottleneck(&mut self, steering: BottleneckSteering) {
        self.bottleneck = Some(steering);
    }

    /// Remove bottleneck steering.
    pub fn clear_bottleneck(&mut self) {
        self.bottleneck = None;
    }

    /// Mutable: core single-token forward — shared by inference and training paths.
    ///
    /// P3 single source of truth: all forward passes go through this method.
    /// Callers control behavior via parameters:
    /// - `delta_modulation`: external Tầng 2 gain for pre-gate layers (None = no external gain)
    /// - `apply_gate_delta`: whether gate's learned delta_gain modulates post-gate layers
    ///
    /// Brain analog: cortical processing loop — sensory input propagates through layers,
    /// thalamic relay (gate) reads and modulates state, then continues to output.
    /// (Crick 1984: thalamic gating in cortical hierarchy)
    ///
    /// Returns: (logits_tensor, delta_gain, gate_alpha).
    fn forward_one_token(
        &mut self,
        token_id: u32,
        delta_modulation: Option<&DeltaModulation>,
        apply_gate_delta: bool,
    ) -> CandleResult<(Tensor, f64, f64)> {
        let token = Tensor::new(&[token_id], &self.device)?;
        let mut xs = self.embedding.forward(&token)?;

        // Pre-gate layers: apply external delta modulation (Tầng 2) if provided.
        for layer in &self.pre_gate_layers {
            let layer_idx = layer.mixer.layer_index;
            let gain = delta_modulation
                .filter(|dm| {
                    dm.target.contains(layer_idx)
                        && (dm.gain_factor - 1.0).abs() > f64::EPSILON
                })
                .map(|dm| dm.gain_factor);
            let steer = self.bottleneck.as_ref().filter(|b| b.should_steer(layer_idx));
            xs = layer.forward(&xs, &mut self.state, gain, steer)?;
        }

        // CognitiveGate: read SSM state (persistent memory), write to residual stream.
        // SSM state differentiates content types much more than residual stream
        // (probe: SSM sim 0.21-0.55 vs residual 0.57).
        // Brain analog: thalamus reads sustained PFC activity (Goldman-Rakic 1995).
        let gate_layer_idx = self.gate_config.gate_position.saturating_sub(1);
        let ssm_state_raw = self
            .state
            .layer_hidden_state(gate_layer_idx)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "SSM state not available at layer {gate_layer_idx}"
                ))
            })?
            .clone();
        // Mean over D_STATE dimension: (batch, d_inner, D_STATE) → (batch, d_inner).
        let ssm_summary = ssm_state_raw.mean(D::Minus1)?;

        let gate_output = self.cognitive_gate.forward(&ssm_summary, &xs)?;
        xs = gate_output.modulated;
        let learned_delta_gain = gate_output.delta_gain;
        let gate_alpha = gate_output.gate_alpha;

        // Post-gate layers: optionally apply gate's learned delta_gain.
        // During inference: apply_gate_delta=true (Tầng 3 delta internalized).
        // During training: apply_gate_delta=false (gate learns via residual blend;
        //   delta_gain scalar extraction would break gradient chain).
        let post_gain = if apply_gate_delta
            && (learned_delta_gain - 1.0).abs() > f64::EPSILON
        {
            Some(learned_delta_gain)
        } else {
            None
        };
        for layer in &self.post_gate_layers {
            let layer_idx = layer.mixer.layer_index;
            let steer = self.bottleneck.as_ref().filter(|b| b.should_steer(layer_idx));
            xs = layer.forward(&xs, &mut self.state, post_gain, steer)?;
        }

        self.state.pos += 1;

        let xs = self.norm_f.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;

        Ok((logits, learned_delta_gain, gate_alpha))
    }

    /// Mutable: inference forward — returns Vec<f32> logits + gate metrics.
    ///
    /// Uses forward_one_token with full Tầng 2+3 modulation.
    fn forward_with_gate(
        &mut self,
        token_id: u32,
        delta_modulation: Option<&DeltaModulation>,
    ) -> CandleResult<(Vec<f32>, f64, f64)> {
        let (logits, delta_gain, gate_alpha) =
            self.forward_one_token(token_id, delta_modulation, true)?;
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        Ok((
            logits_vec[..self.config.vocab_size].to_vec(),
            delta_gain,
            gate_alpha,
        ))
    }

    /// Mutable: training-mode forward — returns logits as Tensor for gradient flow.
    ///
    /// Uses forward_one_token WITHOUT gate delta application (gradient chain preserved).
    /// Gate learns through residual blend only — sufficient for proving the gate learns.
    ///
    /// Brain analog: synaptic plasticity — gradient-based learning maps to Hebbian
    /// weight updates. The gate's residual connection allows gradients to flow back
    /// from the loss, teaching the gate WHEN and HOW to modulate.
    /// (Hebb 1949: neurons that fire together wire together)
    ///
    /// Returns: (seq_len, vocab_size) Tensor of logits for each input position.
    pub fn forward_train(&mut self, tokens: &[u32]) -> CandleResult<Tensor> {
        let mut all_logits = Vec::with_capacity(tokens.len());

        for &token_id in tokens {
            let (logits, _, _) = self.forward_one_token(token_id, None, false)?;
            let logits = logits.narrow(D::Minus1, 0, self.config.vocab_size)?;
            all_logits.push(logits);
        }

        // Stack all position logits: (seq_len, vocab_size).
        Tensor::cat(&all_logits, 0)
    }

    /// Access the gate config.
    pub fn gate_config(&self) -> &CognitiveGateConfig {
        &self.gate_config
    }

    /// Access the cognitive gate (for weight inspection/analysis).
    pub fn cognitive_gate(&self) -> &CognitiveGate {
        &self.cognitive_gate
    }

    /// Access SSM state (for reading per-layer hidden states after forward pass).
    pub fn state(&self) -> &MambaState {
        &self.state
    }
}

// ═══════════════════════════════════════════════════════════════════
// Probing — hidden state analysis infrastructure
// ═══════════════════════════════════════════════════════════════════

/// Result from probing forward pass — captures activations at every layer.
///
/// Brain analog: multi-electrode recording across cortical layers.
/// Each layer's activation reveals what information is present at that
/// depth in the processing hierarchy.
///
/// Used for understanding what cognitive signals the model already encodes
/// before designing deeper intervention (Tầng 4).
pub struct ProbeResult {
    /// Logits from final layer (standard output).
    pub logits: Vec<f32>,
    /// Residual stream activation at each layer boundary.
    /// Index i = activation AFTER layer i, before layer i+1.
    /// Shape per entry: flattened (d_model,). Length = n_layer.
    pub layer_activations: Vec<Vec<f32>>,
    /// Gate's learned cognitive signal (W_read output).
    /// Shape: flattened (cognitive_dim,).
    /// This is what the gate "sees" in the hidden state.
    pub gate_cog_signal: Vec<f32>,
    /// Gate blend factor for this input.
    pub gate_alpha: f64,
    /// Gate's learned delta gain for this input.
    pub gate_delta_gain: f64,
    /// SSM hidden state at gate layer after processing.
    /// Shape: flattened (d_inner * D_STATE,).
    /// The recurrent memory at the depth where the gate reads.
    pub ssm_state_at_gate: Vec<f32>,
}

impl CognitiveMambaWithGate {
    /// Mutable: probing forward — records activations at every layer for analysis.
    ///
    /// Processes all tokens sequentially (filling SSM state), then captures
    /// the final-token activations at each layer boundary. This reveals what
    /// information the model has built up after processing the full sequence.
    ///
    /// Brain analog: recording neural activity at multiple cortical depths
    /// after stimulus presentation (Hubel & Wiesel 1962: hierarchical feature detection).
    pub fn forward_probe(&mut self, tokens: &[u32]) -> CandleResult<ProbeResult> {
        // Process all tokens to fill SSM state with full context.
        // Only capture activations on the LAST token (represents full-context representation).
        let last_token = *tokens.last().ok_or_else(|| {
            candle_core::Error::Msg("Empty token sequence for probing".to_string())
        })?;

        // Process all tokens except last to build up SSM state.
        for &token_id in &tokens[..tokens.len().saturating_sub(1)] {
            let token = Tensor::new(&[token_id], &self.device)?;
            let mut xs = self.embedding.forward(&token)?;
            for layer in self.pre_gate_layers.iter().chain(self.post_gate_layers.iter()) {
                xs = layer.forward(&xs, &mut self.state, None, None)?;
            }
            self.state.pos += 1;
        }

        // Now process last token with activation capture.
        let mut layer_activations = Vec::with_capacity(self.config.n_layer);

        let token = Tensor::new(&[last_token], &self.device)?;
        let mut xs = self.embedding.forward(&token)?;

        // Pre-gate layers: capture activation after each.
        for layer in &self.pre_gate_layers {
            xs = layer.forward(&xs, &mut self.state, None, None)?;
            let activation: Vec<f32> = xs.squeeze(0)?.to_vec1()?;
            layer_activations.push(activation);
        }

        // CognitiveGate: read SSM state, capture cog_signal and gate metrics.
        let gate_layer_idx = self.gate_config.gate_position.saturating_sub(1);
        let ssm_for_gate = self
            .state
            .layer_hidden_state(gate_layer_idx)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "SSM state not available at layer {gate_layer_idx}"
                ))
            })?
            .clone();
        let ssm_summary = ssm_for_gate.mean(D::Minus1)?;
        let gate_output = self.cognitive_gate.forward(&ssm_summary, &xs)?;
        xs = gate_output.modulated;
        let gate_cog_signal: Vec<f32> = gate_output.cog_signal.squeeze(0)?.to_vec1()?;
        let gate_alpha = gate_output.gate_alpha;
        let gate_delta_gain = gate_output.delta_gain;

        // Capture SSM state at gate layer (last pre-gate layer).
        let gate_layer_idx = self.gate_config.gate_position.saturating_sub(1);
        let ssm_state_at_gate: Vec<f32> = self
            .state
            .layer_hidden_state(gate_layer_idx)
            .map(|t| t.flatten_all().and_then(|f| f.to_vec1()))
            .transpose()?
            .unwrap_or_default();

        // Post-gate layers: capture activation after each.
        for layer in &self.post_gate_layers {
            xs = layer.forward(&xs, &mut self.state, None, None)?;
            let activation: Vec<f32> = xs.squeeze(0)?.to_vec1()?;
            layer_activations.push(activation);
        }

        self.state.pos += 1;

        let xs = self.norm_f.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;

        Ok(ProbeResult {
            logits: logits_vec[..self.config.vocab_size].to_vec(),
            layer_activations,
            gate_cog_signal,
            gate_alpha,
            gate_delta_gain,
            ssm_state_at_gate,
        })
    }
}

// ─── Model Loading for CognitiveMambaWithGate ─────────────────────

impl CognitiveMambaWithGate {
    /// Load pretrained Mamba with a fresh CognitiveGate for training.
    ///
    /// Split VarBuilder pattern: base model weights from safetensors (frozen,
    /// no gradient tracking), gate params in VarMap (gradient tracking via Var).
    ///
    /// Returns (model, gate_varmap). Caller creates optimizer from gate_varmap:
    /// ```ignore
    /// let (model, varmap) = CognitiveMambaWithGate::from_pretrained_with_gate(...)?;
    /// let gate_vars = varmap.all_vars();
    /// let optimizer = AdamW::new_lr(gate_vars, 1e-3)?;
    /// ```
    pub fn from_pretrained_with_gate(
        model_id: &str,
        config: MambaConfig,
        gate_config: CognitiveGateConfig,
    ) -> NoosResult<(Self, candle_nn::VarMap)> {
        let device = Device::Cpu;

        // Base model weights from safetensors (frozen — plain Tensors, no Var tracking).
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| NoosError::Internal(format!("HF Hub API init error: {e}")))?;
        let repo = api.model(model_id.to_string());
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| NoosError::Internal(format!("Failed to download weights: {e}")))?;

        // P5 exception: candle mmap requires unsafe (see CognitiveMambaModel::from_pretrained).
        let base_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| NoosError::Internal(format!("Failed to load safetensors: {e}")))?
        };

        // Gate weights in VarMap (trainable — gradient tracking via candle Var).
        // Gate params initialized fresh: W_gate bias = GATE_INIT_BIAS (-3.0) for safe passthrough.
        let gate_varmap = candle_nn::VarMap::new();
        let gate_vb = VarBuilder::from_varmap(&gate_varmap, DType::F32, &device);

        let padded_vocab = config.padded_vocab_size();
        let gate_pos = gate_config.gate_position;

        // Build model with split VarBuilders.
        let embedding = candle_nn::embedding(
            padded_vocab,
            config.d_model,
            base_vb.pp("backbone.embeddings"),
        )
        .map_err(|e| NoosError::Internal(format!("Failed to load embedding: {e}")))?;

        // Pre-gate layers from base_vb (frozen).
        let mut pre_gate_layers = Vec::with_capacity(gate_pos);
        for i in 0..gate_pos {
            pre_gate_layers.push(
                ResidualBlock::new(i, &config, base_vb.pp(format!("backbone.layers.{i}")))
                    .map_err(|e| {
                        NoosError::Internal(format!("Failed to load pre-gate layer {i}: {e}"))
                    })?,
            );
        }

        // CognitiveGate from gate_vb (trainable).
        let cognitive_gate =
            CognitiveGate::new(gate_config.clone(), gate_vb.pp("cognitive_gate")).map_err(|e| {
                NoosError::Internal(format!("Failed to create cognitive gate: {e}"))
            })?;

        // Post-gate layers from base_vb (frozen).
        let mut post_gate_layers = Vec::with_capacity(config.n_layer - gate_pos);
        for i in gate_pos..config.n_layer {
            post_gate_layers.push(
                ResidualBlock::new(i, &config, base_vb.pp(format!("backbone.layers.{i}")))
                    .map_err(|e| {
                        NoosError::Internal(format!("Failed to load post-gate layer {i}: {e}"))
                    })?,
            );
        }

        let norm_f = RmsNorm::new(config.d_model, base_vb.pp("backbone.norm_f"))
            .map_err(|e| NoosError::Internal(format!("Failed to load norm_f: {e}")))?;

        let lm_head = match linear_no_bias(config.d_model, padded_vocab, base_vb.pp("lm_head")) {
            Ok(head) => head,
            Err(_) => {
                let emb_weight = embedding.embeddings();
                Linear::new(emb_weight.clone(), None)
            }
        };

        let state = MambaState::new(&config, 1, &device)
            .map_err(|e| NoosError::Internal(format!("Failed to create state: {e}")))?;

        let model = Self {
            embedding,
            pre_gate_layers,
            cognitive_gate,
            post_gate_layers,
            norm_f,
            lm_head,
            config,
            gate_config,
            state,
            device,
            bottleneck: None,
        };

        Ok((model, gate_varmap))
    }
}

// ─── Trait implementations for CognitiveMambaWithGate ──────────────

impl LocalModel for CognitiveMambaWithGate {
    fn forward(&mut self, tokens: &[u32], _position: usize) -> NoosResult<Vec<f32>> {
        let mut logits = Vec::new();
        for &token in tokens {
            let (l, _, _) = self
                .forward_with_gate(token, None)
                .map_err(|e| NoosError::Internal(format!("Mamba+Gate forward error: {e}")))?;
            logits = l;
        }
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn reset_cache(&mut self) {
        let _ = self.state.reset(&self.config, &self.device);
    }
}

impl CognitiveModel for CognitiveMambaWithGate {
    fn intervention_depth(&self) -> InterventionDepth {
        InterventionDepth::ArchitectureIntegration
    }

    fn forward_cognitive(
        &mut self,
        tokens: &[u32],
        _position: usize,
        delta_modulation: &DeltaModulation,
    ) -> NoosResult<ForwardResult> {
        let mut logits = Vec::new();
        let mut last_delta_gain = 1.0;
        let mut last_gate_alpha = 0.0;

        for &token in tokens {
            let (l, dg, ga) = self.forward_with_gate(token, Some(delta_modulation)).map_err(
                |e| NoosError::Internal(format!("Mamba+Gate cognitive forward error: {e}")),
            )?;
            logits = l;
            last_delta_gain = dg;
            last_gate_alpha = ga;
        }

        // Pre-gate layers modulated by external DeltaModulation (Tầng 2).
        let pre_gate_modulated: Vec<usize> = (0..self.gate_config.gate_position)
            .filter(|&i| {
                delta_modulation.target.contains(i)
                    && (delta_modulation.gain_factor - 1.0).abs() > f64::EPSILON
            })
            .collect();

        // Post-gate layers modulated by gate's learned delta_gain (Tầng 3).
        let post_gate_modulated: Vec<usize> =
            if (last_delta_gain - 1.0).abs() > f64::EPSILON {
                (self.gate_config.gate_position..self.config.n_layer).collect()
            } else {
                Vec::new()
            };

        let mut all_modulated = pre_gate_modulated;
        all_modulated.extend(post_gate_modulated);
        let modulation_applied = !all_modulated.is_empty();

        Ok(ForwardResult {
            logits,
            modulation_applied,
            modulated_layers: all_modulated,
            applied_gain_factor: delta_modulation.gain_factor,
            gate_delta_gain: Some(last_delta_gain),
            gate_alpha: Some(last_gate_alpha),
            hs_stats: None,
        })
    }

    fn num_layers(&self) -> usize {
        self.config.n_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_config_130m() {
        let config = MambaConfig::mamba_130m();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.n_layer, 24);
        assert_eq!(config.d_inner(), 1536);
        assert_eq!(config.dt_rank(), 48);
    }

    #[test]
    fn mamba_config_370m() {
        let config = MambaConfig::mamba_370m();
        assert_eq!(config.d_model, 1024);
        assert_eq!(config.n_layer, 48);
        assert_eq!(config.d_inner(), 2048);
        assert_eq!(config.dt_rank(), 64);
    }

    #[test]
    fn padded_vocab_size_rounds_up() {
        let config = MambaConfig::mamba_130m();
        assert_eq!(config.padded_vocab_size(), 50280); // Already multiple of 8
    }

    #[test]
    fn state_initializes_to_zeros() {
        let config = MambaConfig::mamba_130m();
        let state = MambaState::new(&config, 1, &Device::Cpu).unwrap();
        assert_eq!(state.hs.len(), 24);
        assert_eq!(state.prev_xs.len(), 24);
        assert_eq!(state.pos, 0);
    }

    #[test]
    fn model_creates_with_varmap() {
        let config = MambaConfig {
            d_model: 32,
            n_layer: 2,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // VarMap auto-initializes tensors — model construction should succeed.
        let model = CognitiveMambaModel::new(config, vb);
        assert!(model.is_ok(), "Model construction should succeed with VarMap");

        let model = model.unwrap();
        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(
            model.intervention_depth(),
            InterventionDepth::ActivationAccess
        );
    }

    #[test]
    fn model_forward_with_random_weights() {
        let config = MambaConfig {
            d_model: 32,
            n_layer: 2,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model = CognitiveMambaModel::new(config, vb).unwrap();

        // Forward pass should produce logits of correct size.
        let logits = model.forward(&[1], 0).unwrap();
        assert_eq!(logits.len(), 100, "Logits should match vocab_size");
    }

    #[test]
    fn delta_modulation_changes_block_output() {
        // THE critical test: prove that different gain factors produce different
        // outputs from a Mamba block. We test at BLOCK level with non-zero input
        // to isolate delta modulation from embedding initialization.
        let config = MambaConfig {
            d_model: 32,
            n_layer: 1,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();

        // Non-zero input (simulating a real embedding output).
        let input = Tensor::ones((1, 32), DType::F32, &device).unwrap();

        // Run with gain = None (neutral).
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block_n = CognitiveMambaBlock::new(0, &config, vb).unwrap();
        let mut state_n = MambaState::new(&config, 1, &device).unwrap();
        let out_n: Vec<f32> = block_n.forward(&input, &mut state_n, None).unwrap()
            .squeeze(0).unwrap().to_vec1().unwrap();

        // Run with gain = 1.5 (strong phasic).
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block_p = CognitiveMambaBlock::new(0, &config, vb).unwrap();
        let mut state_p = MambaState::new(&config, 1, &device).unwrap();
        let out_p: Vec<f32> = block_p.forward(&input, &mut state_p, Some(1.5)).unwrap()
            .squeeze(0).unwrap().to_vec1().unwrap();

        // Run with gain = 0.5 (strong tonic).
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block_t = CognitiveMambaBlock::new(0, &config, vb).unwrap();
        let mut state_t = MambaState::new(&config, 1, &device).unwrap();
        let out_t: Vec<f32> = block_t.forward(&input, &mut state_t, Some(0.5)).unwrap()
            .squeeze(0).unwrap().to_vec1().unwrap();

        // Outputs must be non-zero.
        let nonzero = out_n.iter().any(|v| v.abs() > 1e-8);
        assert!(nonzero, "Block output should be non-zero with ones input");

        // THE PROOF: modulated outputs MUST differ from neutral.
        let p_diff = out_n.iter().zip(out_p.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
        let t_diff = out_n.iter().zip(out_t.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
        let pt_diff = out_p.iter().zip(out_t.iter()).any(|(a, b)| (a - b).abs() > 1e-8);

        assert!(p_diff, "Phasic (gain=1.5) MUST differ from neutral");
        assert!(t_diff, "Tonic (gain=0.5) MUST differ from neutral");
        assert!(pt_diff, "Phasic and tonic MUST differ from each other");
    }

    #[test]
    fn model_cognitive_forward_applies_modulation() {
        let config = MambaConfig {
            d_model: 32,
            n_layer: 4,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model = CognitiveMambaModel::new(config, vb).unwrap();

        let dm = DeltaModulation {
            gain_factor: 1.2,
            target: crate::types::intervention::LayerTarget {
                start_layer: 1,
                end_layer: 2,
                total_layers: 4,
            },
            source: crate::types::intervention::DeltaModulationSource::Combined,
        };

        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();
        assert_eq!(result.logits.len(), 100);
        assert!(result.modulation_applied);
        assert_eq!(result.modulated_layers, vec![1, 2]);
        assert_eq!(result.applied_gain_factor, 1.2);
    }

    // ─── CognitiveMambaWithGate tests (Tầng 3) ───

    fn make_gate_model() -> CognitiveMambaWithGate {
        let config = MambaConfig {
            d_model: 32,
            n_layer: 4,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let gate_config = CognitiveGateConfig {
            cognitive_dim: 8,
            gate_position: 2,
            d_model: 32,
            d_inner: 64,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        CognitiveMambaWithGate::new(config, gate_config, vb)
            .expect("Gate model construction should succeed")
    }

    #[test]
    fn model_with_gate_creates_with_varmap() {
        let model = make_gate_model();
        assert_eq!(model.pre_gate_layers.len(), 2);
        assert_eq!(model.post_gate_layers.len(), 2);
        assert_eq!(model.num_layers(), 4);
        assert_eq!(model.vocab_size(), 100);
    }

    #[test]
    fn model_with_gate_forward_produces_logits() {
        let mut model = make_gate_model();
        let logits = model.forward(&[1], 0).unwrap();
        assert_eq!(logits.len(), 100, "Logits should match vocab_size");
    }

    #[test]
    fn model_with_gate_reports_architecture_integration() {
        let model = make_gate_model();
        assert_eq!(
            model.intervention_depth(),
            InterventionDepth::ArchitectureIntegration
        );
    }

    #[test]
    fn gate_and_external_delta_coexist() {
        let mut model = make_gate_model();

        // External delta targets layer 0-1 (pre-gate).
        let dm = DeltaModulation {
            gain_factor: 1.3,
            target: crate::types::intervention::LayerTarget {
                start_layer: 0,
                end_layer: 1,
                total_layers: 4,
            },
            source: crate::types::intervention::DeltaModulationSource::Combined,
        };

        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();

        // Pre-gate layers (0, 1) should be in modulated list (external delta).
        assert!(
            result.modulated_layers.contains(&0),
            "Layer 0 (pre-gate) should be modulated by external delta"
        );
        assert!(
            result.modulated_layers.contains(&1),
            "Layer 1 (pre-gate) should be modulated by external delta"
        );

        // Gate metadata should be present.
        assert!(result.gate_delta_gain.is_some());
        assert!(result.gate_alpha.is_some());

        // If gate's learned delta_gain != 1.0, post-gate layers should be modulated too.
        let gate_gain = result.gate_delta_gain.unwrap();
        if (gate_gain - 1.0).abs() > f64::EPSILON {
            assert!(
                result.modulated_layers.contains(&2) || result.modulated_layers.contains(&3),
                "Post-gate layers should be modulated by learned delta_gain"
            );
        }
    }

    #[test]
    fn forward_result_includes_gate_metadata() {
        let mut model = make_gate_model();
        let dm = DeltaModulation::default();

        let result = model.forward_cognitive(&[1], 0, &dm).unwrap();

        assert!(
            result.gate_delta_gain.is_some(),
            "Gate model should report gate_delta_gain"
        );
        assert!(
            result.gate_alpha.is_some(),
            "Gate model should report gate_alpha"
        );

        let alpha = result.gate_alpha.unwrap();
        assert!(
            alpha >= 0.0 && alpha <= 1.0,
            "gate_alpha should be in [0, 1], got {}",
            alpha
        );
    }

    #[test]
    fn gate_model_reset_clears_state() {
        let mut model = make_gate_model();

        // Forward a few tokens to accumulate state.
        model.forward(&[1], 0).unwrap();
        model.forward(&[2], 1).unwrap();

        // Reset should succeed and clear position.
        model.reset_cache();
        assert_eq!(model.state.pos, 0);
    }

    // ─── Training tests (Phase 3.3) ───

    /// Helper: create gate model with ALL params in VarMap (enables gradient tracking).
    /// Returns (model, varmap) for training tests.
    fn make_trainable_gate_model() -> (CognitiveMambaWithGate, candle_nn::VarMap) {
        let config = MambaConfig {
            d_model: 32,
            n_layer: 4,
            vocab_size: 100,
            pad_vocab_size_multiple: 8,
        };
        let gate_config = CognitiveGateConfig {
            cognitive_dim: 8,
            gate_position: 2,
            d_model: 32,
            d_inner: 64,
        };
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = CognitiveMambaWithGate::new(config, gate_config, vb)
            .expect("Trainable gate model construction should succeed");
        (model, varmap)
    }

    #[test]
    fn forward_train_returns_correct_shape() {
        let (mut model, _varmap) = make_trainable_gate_model();
        let tokens = &[1u32, 2, 3, 4];
        let logits = model.forward_train(tokens).unwrap();

        // Should be (seq_len, vocab_size).
        assert_eq!(logits.dims(), &[4, 100]);
    }

    #[test]
    fn gradient_flows_through_gate() {
        // THE critical training test: prove backward() succeeds through gate.
        let (mut model, varmap) = make_trainable_gate_model();
        let tokens = &[1u32, 2, 3];
        let targets = Tensor::new(&[2u32, 3, 4], &Device::Cpu).unwrap();

        let logits = model.forward_train(tokens).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();

        // backward() must succeed — proves gradient chain is intact.
        let grads = loss.backward().unwrap();

        // Gate params should have non-None gradients.
        let gate_has_grads = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.contains("cognitive_gate"))
            .any(|(_, var)| grads.get(var).is_some());

        assert!(
            gate_has_grads,
            "Gate params should receive gradients via backward()"
        );
    }

    #[test]
    fn gate_params_change_after_step() {
        // Prove optimizer actually updates gate weights.
        use candle_nn::optim::{AdamW, Optimizer};

        let (mut model, varmap) = make_trainable_gate_model();

        // Snapshot gate params before training.
        let before: Vec<(String, Vec<f32>)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.contains("cognitive_gate"))
            .map(|(name, var)| {
                let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
                (name.clone(), vals)
            })
            .collect();

        assert!(!before.is_empty(), "Should have gate params in varmap");

        // Create optimizer for gate params only.
        let gate_vars: Vec<candle_core::Var> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.contains("cognitive_gate"))
            .map(|(_, var)| var.clone())
            .collect();

        let mut optimizer = AdamW::new_lr(gate_vars, 1e-2).unwrap();

        // One training step.
        let tokens = &[1u32, 2, 3];
        let targets = Tensor::new(&[2u32, 3, 4], &Device::Cpu).unwrap();
        let logits = model.forward_train(tokens).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        optimizer.backward_step(&loss).unwrap();

        // Snapshot gate params after training.
        let after: Vec<(String, Vec<f32>)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.contains("cognitive_gate"))
            .map(|(name, var)| {
                let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1().unwrap();
                (name.clone(), vals)
            })
            .collect();

        // At least one gate param must have changed.
        let any_changed = before.iter().zip(after.iter()).any(|((_, b), (_, a))| {
            b.iter().zip(a.iter()).any(|(bv, av)| (bv - av).abs() > 1e-10)
        });

        assert!(
            any_changed,
            "Gate params should change after one optimizer step"
        );
    }
}
