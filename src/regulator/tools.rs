//! Tool-call observation channel (Path A, 0.3.0).
//!
//! Modern LLM agents make tool calls (function calls, API calls,
//! retrieval). Without visibility into these, the regulator cannot
//! detect a very common failure mode: the agent calls the same tool
//! over and over, burning cost without producing progress.
//!
//! This module tracks per-turn tool-call sequences. It exposes a
//! detector `detected_loop()` that fires `Some((tool, count))` when the
//! last [`TOOL_LOOP_THRESHOLD`] consecutive tool calls are the same
//! tool name — the signature pattern of "agent is stuck in a retry
//! loop on a single tool that is failing / not making progress."
//!
//! Counts reset on every [`LLMEvent::TurnStart`] so a new turn gets a
//! fresh view — accumulation across turns would conflate "legitimately
//! hit this tool 3 times across a multi-step plan" with "stuck in a
//! one-tool loop." The loop-detection unit of account is the turn.
//!
//! `ToolResult` events carry `duration_ms` + `success` for observability
//! hooks (accessible via [`ToolStatsAccumulator::total_calls`] and
//! related getters); they do not influence loop detection in 0.3.0.
//!
//! [`LLMEvent::TurnStart`]: super::LLMEvent::TurnStart

use std::collections::HashMap;

// ── Constants ──────────────────────────────────────────────────────────

/// Number of consecutive calls to the same tool that trip the loop
/// circuit-break. Chosen to balance "false alarms on a legitimate
/// 3-retry strategy" against "waste from a runaway loop."
///
/// 5 is conservative: a well-behaved agent almost never calls the
/// same tool 5× in a row without interleaving a reasoning or
/// different-tool step. Anthropic's agentic-retrieval loop
/// research observes loops typically exceed 10 repetitions when
/// they occur, so 5 catches them with margin.
pub const TOOL_LOOP_THRESHOLD: usize = 5;

// ── Records ────────────────────────────────────────────────────────────

/// One observed tool call.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Tool name as reported by the agent (the caller owns the naming
    /// convention — `"search"`, `"db.query"`, `"exec_python"`, etc.).
    pub tool_name: String,
    /// Optional JSON-serialised args. Opaque to the regulator; stored
    /// only for observability / downstream inspection by the app.
    pub args_json: Option<String>,
    /// Index within the current turn (0-based). Resets on `TurnStart`.
    pub turn_local_index: usize,
}

/// One observed tool result.
#[derive(Debug, Clone)]
pub struct ToolResultRecord {
    pub tool_name: String,
    pub success: bool,
    pub duration_ms: u64,
    pub error_summary: Option<String>,
    pub turn_local_index: usize,
}

// ── Accumulator ────────────────────────────────────────────────────────

/// Per-turn tool-call history + loop detector.
///
/// Reset via [`ToolStatsAccumulator::reset_turn`] at every
/// `LLMEvent::TurnStart`. The regulator drives that lifecycle; callers
/// don't interact with this type directly.
#[derive(Debug, Clone, Default)]
pub struct ToolStatsAccumulator {
    /// All tool calls observed in the current turn, in emission order.
    calls: Vec<ToolCallRecord>,
    /// All tool results observed in the current turn, in emission order.
    results: Vec<ToolResultRecord>,
}

impl ToolStatsAccumulator {
    /// Fresh accumulator with empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear per-turn history. Called on `LLMEvent::TurnStart`.
    pub fn reset_turn(&mut self) {
        self.calls.clear();
        self.results.clear();
    }

    /// Record a tool call.
    pub fn record_call(&mut self, tool_name: String, args_json: Option<String>) {
        let turn_local_index = self.calls.len();
        self.calls.push(ToolCallRecord {
            tool_name,
            args_json,
            turn_local_index,
        });
    }

    /// Record a tool result.
    pub fn record_result(
        &mut self,
        tool_name: String,
        success: bool,
        duration_ms: u64,
        error_summary: Option<String>,
    ) {
        let turn_local_index = self.results.len();
        self.results.push(ToolResultRecord {
            tool_name,
            success,
            duration_ms,
            error_summary,
            turn_local_index,
        });
    }

    /// Total tool calls observed this turn.
    pub fn total_calls(&self) -> usize {
        self.calls.len()
    }

    /// Total tool results observed this turn.
    pub fn total_results(&self) -> usize {
        self.results.len()
    }

    /// Per-tool call counts this turn. Useful for observability —
    /// `{"db.query": 3, "search": 1}` style reporting.
    pub fn counts_by_tool(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for call in &self.calls {
            *counts.entry(call.tool_name.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Detect a consecutive-same-tool loop.
    ///
    /// Returns `Some((tool_name, count))` when the last [`TOOL_LOOP_THRESHOLD`]
    /// calls were all the same tool. `count` is the number of consecutive
    /// same-tool calls counting back from the most recent.
    ///
    /// Returns `None` when:
    /// - Fewer than `TOOL_LOOP_THRESHOLD` calls have been observed.
    /// - The last `TOOL_LOOP_THRESHOLD` calls include at least two
    ///   distinct tools (the agent interleaved, which breaks a loop).
    pub fn detected_loop(&self) -> Option<(String, usize)> {
        if self.calls.len() < TOOL_LOOP_THRESHOLD {
            return None;
        }
        let last_tool = &self.calls.last()?.tool_name;
        // Count back from the end while the tool name matches. If we
        // hit >= TOOL_LOOP_THRESHOLD before seeing a different tool (or
        // running out), we have a loop.
        let mut consecutive = 0usize;
        for call in self.calls.iter().rev() {
            if &call.tool_name == last_tool {
                consecutive += 1;
            } else {
                break;
            }
        }
        if consecutive >= TOOL_LOOP_THRESHOLD {
            Some((last_tool.clone(), consecutive))
        } else {
            None
        }
    }

    /// Total wall-clock time spent in tool results this turn, in
    /// milliseconds. Zero when no results have been observed.
    pub fn total_duration_ms(&self) -> u64 {
        self.results.iter().map(|r| r.duration_ms).sum()
    }

    /// Count of failed results this turn.
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_loop() {
        let acc = ToolStatsAccumulator::new();
        assert!(acc.detected_loop().is_none());
        assert_eq!(acc.total_calls(), 0);
    }

    #[test]
    fn below_threshold_has_no_loop() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..(TOOL_LOOP_THRESHOLD - 1) {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_none());
    }

    #[test]
    fn threshold_same_tool_detects_loop() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        let (tool, count) = acc.detected_loop().expect("loop should fire");
        assert_eq!(tool, "search");
        assert_eq!(count, TOOL_LOOP_THRESHOLD);
    }

    #[test]
    fn interleaved_tools_break_loop() {
        let mut acc = ToolStatsAccumulator::new();
        // 4 search + 1 db.query + 4 search ⇒ trailing same-tool run is 4,
        // below threshold.
        for _ in 0..4 {
            acc.record_call("search".into(), None);
        }
        acc.record_call("db.query".into(), None);
        for _ in 0..4 {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_none());
    }

    #[test]
    fn reset_turn_clears_state() {
        let mut acc = ToolStatsAccumulator::new();
        for _ in 0..TOOL_LOOP_THRESHOLD {
            acc.record_call("search".into(), None);
        }
        assert!(acc.detected_loop().is_some());
        acc.reset_turn();
        assert!(acc.detected_loop().is_none());
        assert_eq!(acc.total_calls(), 0);
    }

    #[test]
    fn counts_by_tool_aggregates_correctly() {
        let mut acc = ToolStatsAccumulator::new();
        acc.record_call("search".into(), None);
        acc.record_call("search".into(), None);
        acc.record_call("db.query".into(), None);
        let counts = acc.counts_by_tool();
        assert_eq!(counts.get("search"), Some(&2));
        assert_eq!(counts.get("db.query"), Some(&1));
    }

    #[test]
    fn duration_and_failure_counters() {
        let mut acc = ToolStatsAccumulator::new();
        acc.record_result("search".into(), true, 100, None);
        acc.record_result("db.query".into(), false, 250, Some("timeout".into()));
        acc.record_result("search".into(), true, 50, None);
        assert_eq!(acc.total_duration_ms(), 400);
        assert_eq!(acc.failure_count(), 1);
        assert_eq!(acc.total_results(), 3);
    }
}
