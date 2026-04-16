//! Unified World Model — single generative model (Friston 2010, Pessoa 2023).
//!
//! Three lifetime tiers:
//! - **Per-Turn**: reconstructed each message (affect, gate, sensoryPE)
//! - **Per-Conversation**: maintained in-memory (turn, RPE, dynamics)
//! - **Learned**: cross-conversation (weights, threats, strategies)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;


use super::belief::SharedBeliefState;
use super::gate::GateResult;

/// NE gain modulation mode (Aston-Jones & Cohen 2005).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GainMode {
    /// High confidence, focused exploitation, narrow signal.
    Phasic,
    /// Low confidence, broad exploration, wide signal.
    Tonic,
    /// Balanced.
    #[default]
    Neutral,
}

/// Response strategy type detected from LLM output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResponseStrategy {
    ClarifyFirst,
    StepByStep,
    StructuredAnalysis,
    ExecuteTask,
    DirectAnswer,
}

/// Strategy confidence level (Graybiel 2008 habit formation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrategyConfidence {
    /// 12+ observations, 75%+ success. Use ALWAYS.
    Habit,
    /// 8+ observations, 65%+ success. Use "prefer".
    Strong,
    /// 5+ observations, 50%+ success. Use "consider".
    Moderate,
    /// Below thresholds.
    Weak,
}

/// Strategy compliance assessment (Sperry 1950 efference copy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyCompliance {
    Compliant,
    DeviatedBetter,
    DeviatedWorse,
    NoRecommendation,
}

/// EMA-tracked success rate for a topic cluster.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuccessEntry {
    pub success_rate: f64,
    pub count: u32,
}

/// Strategy competition result (Cisek 2010 urgency-gated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCompetition {
    pub winner: Option<StrategyCompetitionEntry>,
    pub runner_up: Option<StrategyCompetitionEntry>,
    pub margin: f64,
    pub urgency: f64,
    pub should_commit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCompetitionEntry {
    pub name: String,
    pub score: f64,
    pub confidence: Option<StrategyConfidence>,
}

/// Conversation dynamics state (DPC 2024, Murray 2014).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConversationRegime {
    #[default]
    Opening,
    Exploration,
    DeepDive,
    ProblemSolving,
    Divergent,
}

/// Level 3 temporal hierarchy state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsState {
    pub regime: ConversationRegime,
    pub depth: f64,
    pub turns_in_regime: u32,
    pub accumulated_turn_pe: f64,
}

impl Default for DynamicsState {
    fn default() -> Self {
        Self {
            regime: ConversationRegime::Opening,
            depth: 0.0,
            turns_in_regime: 0,
            accumulated_turn_pe: 0.0,
        }
    }
}

/// Cross-conversation learned state — persisted between sessions.
///
/// Contains only non-cortical learning traces actually populated by the live
/// pipeline: LC gain mode + tick, and per-cluster response-success /
/// per-strategy EMA. Topic cluster keys are opaque indices
/// (from `detector::extract_topics` as an interim heuristic), not cognitive
/// topic models — model's attention handles topic knowledge.
///
/// **Audit 2026-04-14** removed 4 dead fields + 3 dead types from this
/// struct: `threats` + `ThreatEntry` (only populated by a test, never by
/// live code after `ThreatAssociations` was deleted in Phase 1),
/// `layer_precision` + `PrecisionEntry` (never read or written),
/// `response_calibration` + `CalibrationEntry` (never read or written),
/// `composition_effectiveness` (never read or written). Serde's default
/// behaviour silently drops unknown fields on deserialisation, so old
/// exported snapshots containing those fields still load into the new
/// struct without error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedState {
    pub gain_mode: GainMode,
    pub tick: u32,

    /// Strategy success per topic cluster.
    pub response_success: HashMap<String, SuccessEntry>,
    /// Strategy success per topic × strategy.
    pub response_strategies: HashMap<String, HashMap<String, SuccessEntry>>,
}

impl Default for LearnedState {
    fn default() -> Self {
        Self {
            gain_mode: GainMode::Neutral,
            tick: 0,
            response_success: HashMap::new(),
            response_strategies: HashMap::new(),
        }
    }
}

/// Unified World Model — single generative model.
///
/// Extends SharedBeliefState with gate, sensory PE, and cross-conversation learned state.
/// All cognitive modules read and write this single state object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModel {
    // ── SharedBeliefState (embedded) ──
    pub belief: SharedBeliefState,

    // ── Tier 1: Per-Turn ──
    pub gate: GateResult,
    /// 0-1, feature-specific prediction error.
    pub sensory_pe: f64,

    /// 0-1, allostatic body budget (Barrett 2017, Sterling 2012).
    /// Depletes on high PE/stress, replenishes on success/rest.
    /// Low budget → perceive more threats, trust model less (Principle 4+8).
    pub body_budget: f64,

    // ── Tier 2: Per-Conversation ──
    pub idle_cycles: u32,
    pub turns_since_switch: u32,
    /// 0-1, allostatic demand (Barrett 2017).
    pub resource_pressure: f64,
    /// 0-1, predicted quality of last response.
    pub last_response_prediction: f64,
    /// -1 to +1, Schultz reward prediction error.
    pub response_rpe: f64,
    /// Rolling window for PE volatility (Behrens 2007).
    pub pe_history: Vec<f64>,
    /// 0-1, estimated PE volatility.
    pub pe_volatility: f64,
    pub last_response_strategy: Option<ResponseStrategy>,
    pub last_response_length: Option<usize>,
    pub last_response_question_ratio: Option<f64>,
    /// Pre-computed strategy competition (Kok 2012).
    pub pre_computed_competition: Option<StrategyCompetition>,
    pub pre_computed_for_cluster: Option<String>,
    pub last_composition: Option<String>,
    /// Topic → turn number when last discussed (Clark 1996 common ground).
    pub discussed_topics: HashMap<String, u32>,
    /// Efference copy: recommended strategy before LLM call.
    pub recommended_strategy: Option<ResponseStrategy>,
    pub last_compliance: Option<StrategyCompliance>,
    pub dynamics: Option<DynamicsState>,

    // ── Tier 3: Cross-Conversation ──
    pub learned: LearnedState,
}

impl WorldModel {
    /// Create a fresh world model for a new conversation.
    pub fn new(conversation_id: String) -> Self {
        Self {
            belief: SharedBeliefState::new(conversation_id),
            gate: GateResult::default(),
            sensory_pe: 0.0,
            body_budget: 1.0, // Full budget at start (Sterling 2012 allostatic baseline)
            idle_cycles: 0,
            turns_since_switch: 0,
            resource_pressure: 0.0,
            last_response_prediction: 0.5,
            response_rpe: 0.0,
            pe_history: Vec::new(),
            pe_volatility: 0.0,
            last_response_strategy: None,
            last_response_length: None,
            last_response_question_ratio: None,
            pre_computed_competition: None,
            pre_computed_for_cluster: None,
            last_composition: None,
            discussed_topics: HashMap::new(),
            recommended_strategy: None,
            last_compliance: None,
            dynamics: None,
            learned: LearnedState::default(),
        }
    }
}

// ── Constants ──────────────────────────────────────────────────────────

/// PE volatility tracking window (Behrens 2007 ACC volatility).
pub const PE_VOLATILITY_WINDOW: usize = 5;
/// Response success EMA rate (moderate learning speed, CR3).
pub const RESPONSE_SUCCESS_EMA: f64 = 0.25;
/// Habit formation: min observations (Graybiel 2008 dorsolateral striatum).
pub const HABIT_MIN_COUNT: u32 = 12;
/// Habit formation: min success rate (75%+ = reliable habit).
pub const HABIT_MIN_SUCCESS: f64 = 0.75;
/// Strong strategy: min observations.
pub const STRONG_MIN_COUNT: u32 = 8;
/// Strong strategy: min success rate.
pub const STRONG_MIN_SUCCESS: f64 = 0.65;
/// Moderate strategy: min observations.
pub const MODERATE_MIN_COUNT: u32 = 5;
/// Moderate strategy: min success rate.
pub const MODERATE_MIN_SUCCESS: f64 = 0.5;
/// Avoid strategy: max success rate (below this = consistently bad).
pub const AVOID_MAX_SUCCESS: f64 = 0.4;
/// Avoid strategy: min observations (need enough data before avoiding).
pub const AVOID_MIN_COUNT: u32 = 5;

/// Classify strategy confidence from observation count and success rate.
pub fn classify_strategy_confidence(success_rate: f64, count: u32) -> StrategyConfidence {
    if count >= HABIT_MIN_COUNT && success_rate >= HABIT_MIN_SUCCESS {
        StrategyConfidence::Habit
    } else if count >= STRONG_MIN_COUNT && success_rate >= STRONG_MIN_SUCCESS {
        StrategyConfidence::Strong
    } else if count >= MODERATE_MIN_COUNT && success_rate >= MODERATE_MIN_SUCCESS {
        StrategyConfidence::Moderate
    } else {
        StrategyConfidence::Weak
    }
}

/// Whether a strategy should be avoided (low success + enough data).
pub fn should_avoid_strategy(success_rate: f64, count: u32) -> bool {
    count >= AVOID_MIN_COUNT && success_rate <= AVOID_MAX_SUCCESS
}

/// Recommend best learned strategy for a topic cluster (Principle 5: classification emerges).
///
/// Queries learned.response_strategies for the topic cluster, finds the
/// highest-confidence strategy that isn't "Avoid", and returns it.
///
/// Brain analog: dorsomedial striatum (Daw 2005) — model-based strategy
/// selection informed by learned action-outcome associations.
///
/// Returns None if no strategy has enough observations (Weak signal).
pub fn get_recommended_strategy(
    topic_cluster: &str,
    learned: &LearnedState,
) -> Option<(ResponseStrategy, StrategyConfidence)> {
    let strategies = learned.response_strategies.get(topic_cluster)?;

    let mut best: Option<(ResponseStrategy, StrategyConfidence, f64)> = None;

    for (strategy_key, entry) in strategies {
        // Skip strategies with insufficient data or poor track record.
        if should_avoid_strategy(entry.success_rate, entry.count) {
            continue;
        }

        let confidence = classify_strategy_confidence(entry.success_rate, entry.count);
        if matches!(confidence, StrategyConfidence::Weak) {
            continue;
        }

        // Parse strategy key back to enum.
        let strategy = match strategy_key.as_str() {
            "ClarifyFirst" => ResponseStrategy::ClarifyFirst,
            "StepByStep" => ResponseStrategy::StepByStep,
            "StructuredAnalysis" => ResponseStrategy::StructuredAnalysis,
            "ExecuteTask" => ResponseStrategy::ExecuteTask,
            "DirectAnswer" => ResponseStrategy::DirectAnswer,
            _ => continue, // Unknown strategy key — skip
        };

        // Pick highest success rate among non-Weak strategies.
        let dominated = best
            .as_ref()
            .is_some_and(|(_, _, best_rate)| entry.success_rate <= *best_rate);
        if !dominated {
            best = Some((strategy, confidence, entry.success_rate));
        }
    }

    best.map(|(strategy, confidence, _)| (strategy, confidence))
}
