# Nous ↔ Application Contract

Specifies the semantic agreement between Nous (library) and the application that embeds it. Type signatures describe shape; this document describes **meaning**. Without a shared interpretation of `body_budget = 0.7` or `signals.conservation > X`, two apps using the same Nous version will diverge.

Scope: `CognitiveSession` API (`src/session.rs`), `CognitiveSignals` (`src/cognition/signals.rs`), `LearnedState` (`src/types/world.rs`).

Versioned against: Phase 7 (2026-04-14). Break-compat changes to this contract should bump the library minor version.

---

## 1. CognitiveSignals — reading Nous state

Signals are the primary application-facing interface. Each is a **decision-oriented scalar**, not a status report. Apps use them to gate behavior.

### 1.1 `conservation: f64` (range [0, 1])

**Meaning**: how strongly the system recommends reducing effort / deferring expensive operations. Derived from `body_budget` depletion, `resource_pressure` competition, and sustained-arousal penalty (remaining capacity).

**Operating ranges** (calibrated 2026-04-14, phase 6):

- **Healthy operation** (response quality mostly good, even under cost): conservation stays in **0.05 – 0.30**. `body_budget` maintained by positive RPE from successful responses.
- **Struggling operation** (sustained stress AND poor response quality): conservation reaches **0.50+** after ~30-60 turns. Positive RPE absent → `body_budget` drops below threshold → `budget_factor` contributes meaningfully.

The design intent: conservation fires when cost AND poor outcomes BOTH apply, not cost alone. A costly-but-succeeding session doesn't trigger conservation — the allostatic system knows it's working.

**Suggested app interpretation** (not enforced):

| Range | Suggested app response |
|-------|------------------------|
| 0.00 – 0.15 | Full exploration. Baseline / recovering / succeeding. |
| 0.15 – 0.30 | Normal operation. Some arousal penalty from sustained stress. |
| 0.30 – 0.50 | Consider shallower paths. Stress sustained, some budget pressure. |
| 0.50+ | Conservation mode. Cost AND outcomes both bad. Defer non-critical work, stop agent loops, surface to user. |

**Thresholds are app choices, not Nous guarantees.** Nous does not promise that a given value means the same thing across two runs — it reflects the convolution of several internal signals. Apps that hard-code thresholds should document them and ideally measure the observed range on their own workload before setting decision points.

**Calibration note**: the threshold where `budget_factor` activates is at `threshold_body_budget_conservation` (base 0.70, adaptive). See `src/cognition/adaptive_thresholds.rs` for the calibration rationale; the bug-fix trail that made this signal usable lives in the project's private session memos.

### 1.1.1 When does using `signals.conservation` actually help your app?

Empirical guidance from `examples/task_eval_budget_sweep.rs` (24-query mixed workload across budget caps 4.0 → 20.0):

- **Tight budget regime** (≤ ~0.33 effort units per query): Nous's conservation + reward-learning combo wins on BOTH absolute quality AND quality-per-cost vs a competent app-level cost-tracker. Use Nous unconditionally.
- **Mid budget regime** (~0.33-0.67 effort units per query): Nous wins on absolute quality (peaked +5.59 vs cost-tracker baseline at budget=8 over 24 queries), but a cost-only tracker may edge on quality-per-cost. Pick by what you optimize for.
- **Loose budget regime** (≥ ~0.67 effort units per query): both agents converge — there's no budget pressure for conservation to act on. Nous's allostatic overhead is unjustified by quality lift in this regime.

The general shape: Nous's quality advantage is largest in the moderately-tight regime where the cost-tracker baseline downshifts prematurely (at budget/2) but Nous's depletion signal correctly recognizes that high-quality responses are replenishing capacity. Apps should benchmark on their own workload distribution to find their operating regime. The full sweep table lives in the project's private session memos.

**Invariant Nous guarantees**: within a session, `conservation` rises monotonically when stressed inputs arrive and `idle_cycle` isn't called. It falls after `idle_cycle`. The direction of change is meaningful; the absolute value is relative.

### 1.2 `salience: f64` (range [0, 1])

**Meaning**: how attention-worthy the current input is. Combines arousal, gate classification (URGENT / NOVEL / ROUTINE), and sensory prediction error.

**Suggested app interpretation**:

| Range | Suggested app response |
|-------|------------------------|
| 0.0 – 0.3 | Routine. Fast path OK. |
| 0.3 – 0.6 | Standard processing. |
| 0.6 – 0.8 | Prioritize. Deeper reasoning, preserve context. |
| 0.8 – 1.0 | Urgent / novel. Top-priority handling, minimize delay. |

**Invariant**: high salience → Nous internally suppresses breadth-oriented signals (P10 gating). The signal is produced after that internal suppression has settled.

### 1.3 `confidence: f64` (range [0, 1])

**Meaning**: how reliable Nous thinks its own current assessment is. Derived from gate classification confidence, reduced by PE volatility (unstable environment → trust less).

**Suggested app interpretation**:

| Range | Suggested app response |
|-------|------------------------|
| 0.0 – 0.4 | Hedge. Present uncertainty explicitly, avoid committing to single plan. |
| 0.4 – 0.7 | Proceed normally. |
| 0.7 – 1.0 | High trust. Commit, move fast. |

**What `confidence` is NOT**: this is not the model's confidence in its own output (that's a different signal, requires logit entropy from inference). This is Nous's meta-confidence in its own classification of the turn.

### 1.4 `strategy: Option<ResponseStrategy>`

**Meaning**: `Some(strategy)` = Nous has learned that `strategy` has succeeded on semantically-similar prior turns (cluster hash match via `detector::build_topic_cluster`). `None` = insufficient data, no recommendation.

**Contract**: `Some` is emitted only after the reward learning loop has accumulated ≥ RECOMMENDATION_MIN_SAMPLES samples for this cluster AND success rate exceeds the threshold. Values earlier are `None`.

**App responsibility**: treat as recommendation, not mandate. Apps may override. If apps override and the override succeeds/fails, `process_response(resp, quality)` still updates the learned state for future recommendations.

**Cluster stability pitfall** (surfaced 2026-04-14 in `task_eval_abstention.rs`): the cluster hash comes from `extract_meaningful_words(user_query, min_length=3)` then `build_topic_cluster` (sort, truncate to 2 topics). Words shorter than 3 characters get filtered. **If your user_query templates contain numeric IDs or short tokens that affect cluster contents, training and eval clusters can silently diverge.** Example: `"Help me debug issue 999"` keeps `"999"` (3 chars) in topics; `"Help me debug issue 0"` drops `"0"` (1 char). Different cluster hashes → no recommendation match across sessions.

Mitigation: keep app-controlled IDs out of the user-facing template (handle them metadata-side), or use IDs ≤ 2 chars consistently so they're always filtered, or use ≥ 3-char IDs consistently. Ad-hoc mixing breaks cross-session matching.

**Minimum-observations threshold pitfall** (surfaced 2026-04-15 in `task_eval_real_llm_multi_signal.rs`): `RECOMMENDATION_MIN_SAMPLES` above is not a single constant — it's the `MODERATE_MIN_COUNT = 5` in `types/world.rs` (plus `MODERATE_MIN_SUCCESS = 0.5`). **For `signals.strategy` to be `Some`, each cluster must have at least 5 successful observations of the same strategy before the eval starts.** Apps warming up `LearnedState` for N clusters need ≥ 5N warmup turns (one per cluster per round × 5 rounds, or 5 turns per cluster). Under-warming produces `signals.strategy = None` for every eval turn, silently disabling Nous's cross-session reward learning advantage. Observed: `task_eval_real_llm_multi_signal.rs` v2 with 2 rounds produced count=2 per cluster → `strategy` was None on every turn across 3 seeds; v3 with 6 rounds → `strategy` fires correctly on pre-trained clusters.

Mitigation: verify `warm.response_strategies.values().any(|m| m.values().any(|e| e.count >= 5))` before the eval runs, OR warm each expected cluster explicitly ≥ 5 times. For interactive apps that can't pre-train, expect `signals.strategy = None` for the first ~5 turns per new topic cluster — this is by design (habit-formation threshold per Graybiel 2008, not a bug).

### 1.5 `gain_mode: GainMode` (Phasic / Tonic / Neutral)

**Meaning**: the current LC-NE mode.
- **Phasic**: locked onto a salient event. Prefer focused / narrow processing.
- **Tonic**: sustained elevated baseline. Prefer exploratory / broad processing.
- **Neutral**: baseline. No mode preference.

**Maps to**: temperature modulation in Tầng 1 sampling. If the app uses `turn.sampling`, gain_mode is already applied there — reading `signals.gain_mode` is for logging/inspection.

### 1.6 `recent_quality: f64` (range [0, 1])

**Meaning**: EMA of recent response qualities reported via `process_response(resp, quality)`. Predicted quality for the next response.

**Caveat**: this EMA is only as good as the `quality` values the app has been reporting. If the app passes `1.0` constantly (because it has no quality signal), this is noise.

### 1.7 `rpe: f64` (range [-1, +1])

**Meaning**: reward prediction error on the most recent turn. `quality - recent_quality_before_update`. Positive = exceeded expectation. Negative = underperformed.

**App use**: combine with `recent_quality` to detect failure patterns. Example: `recent_quality < 0.4 AND rpe < 0.0` → strategy is failing consistently, consider switching.

### 1.8 `valence: AffectValence` (Positive / Negative / Neutral)

**Meaning**: dominant valence of the user's recent messages (from regex heuristics in `emotional.rs`, with plans to replace with SSM readout).

**Caveat**: this is interim heuristic, NOT a claim to implement amygdala valence discrimination. Text-pattern-based.

---

## 2. Body Budget — the allostatic resource

`body_budget ∈ [0, 1]` tracks allostatic resources. Exposed on `TurnResult.body_budget` and embedded in `signals.conservation`.

### 2.1 What depletes `body_budget`

- **Arousal-driven depletion** (inside `perceive`): sustained high arousal from user messages.
- **Cost-driven depletion** (requires `track_cost`): application reports effort via `session.track_cost(cost)`, where `cost ∈ [0, 1]` is the normalized work done for the completed turn.

### 2.2 What replenishes `body_budget`

- **Idle cycles**: `session.idle_cycle()` runs between-turn maintenance. Each call nudges the budget back toward 1.0.

### 2.3 The closed-loop contract

For `body_budget` to be meaningful, the application **must** call `track_cost` with honest values after each turn. If the app doesn't:

- Stress events will still deplete the budget (via `perceive`), so the signal will drift meaningfully.
- BUT the resource-management story ("Nous senses its own cost") breaks — nothing ties `body_budget` to actual resource consumption.
- `signals.conservation` will underestimate real depletion when the app is doing expensive work but not reporting it.

**Rule of thumb for `cost` value**:

| Operation | Suggested `cost` |
|-----------|-----------------|
| Cached response, no LLM call | 0.0 – 0.1 |
| Simple single-model call | 0.2 – 0.4 |
| Long reasoning chain / multiple tool calls | 0.5 – 0.8 |
| Exhaustive search / many API calls | 0.8 – 1.0 |

App chooses its own normalization. Nous uses the scalar as an effort signal (`COST_DEPLETION_RATE = 0.02` per unit — defined in `src/session.rs`).

### 2.4 What `body_budget` is NOT

- Not a token count. Not an API-cost counter.
- Not a user-facing metric (users don't have body budgets).
- Not a hard cap on anything — it's a *signal* for apps to act on.

---

## 3. LearnedState — ownership and persistence

`LearnedState` is a plain serializable struct (serde). It contains:

- Pavlovian threat associations (per topic-cluster threat EMA)
- Response strategy success rates (per topic-cluster per strategy)
- Response calibration (EMA of predicted vs observed quality)
- LC gain mode + current arousal

### 3.1 Who owns it

**Nous owns**: the *content* — the computation of what goes into each field.

**Application owns**: the *persistence* — serializing to disk / database, restoring at session start, deleting on user request, ensuring privacy compliance.

Nous never writes to the filesystem or network. Nous exposes `export_learned() → LearnedState` and `import_learned(LearnedState)`. Everything in between is the application's responsibility.

### 3.2 Privacy obligations (application side)

`LearnedState` is user-specific. Topic clusters are opaque hashes but derived from the user's actual messages. Treat `LearnedState` as PII-equivalent:

- Scope persistence by user identity, not globally.
- Offer deletion in line with your privacy policy.
- Do not ship `LearnedState` between users.

Nous provides no cross-user isolation — that is an application concern.

### 3.3 Versioning

`LearnedState` uses serde derives. Breaking changes to its shape will break old snapshots. Current policy (as of Phase 7):

- Adding fields with `#[serde(default)]` is backward-compatible.
- Renaming or removing fields is breaking.
- Applications SHOULD tag persisted snapshots with the Nous library version and reject mismatched imports rather than silently losing data.

Nous provides no migration facility today.

---

## 4. Closed-loop requirements for Phase 7 value claims

The allostatic claims (reward learning, resource-aware agents, memory-aware sessions) rely on the application participating in the closed loop. For each claim to hold:

| Phase 7 claim | Requires app to call |
|---------------|----------------------|
| `signals.conservation` reflects real cost | `track_cost(cost)` after each turn |
| `signals.strategy` recommends effective strategies | `process_response(response, quality)` after each response |
| `signals.recent_quality` / `rpe` reflect reality | `process_response(response, quality)` with honest `quality` |
| Cross-session learning persists | `export_learned()` at session end, `with_learned()` at next session start |
| `signals.conservation` decays during idle | `idle_cycle()` between turns (when no user input for a while) |

**If the app skips any of these, the corresponding signal silently degrades** — it still produces a value, but the value is no longer grounded in reality. This is a fail-open design (P5): better to return a degraded signal than to crash. But apps that want the allostatic guarantees must honor the loop.

---

## 5. Threading and concurrency

`CognitiveSession` is **not** `Send + Sync`. It holds interior state that mutates per call.

- One session per conversation.
- One thread per session.
- Sharing a session across threads requires external synchronization and is not a supported use case today.

Applications with multiple concurrent conversations own their own session-per-conversation bookkeeping.

---

## 6. What Nous does NOT promise

To bound expectations:

- **No fairness claim about signals.** `conservation > 0.5` does not predict task failure at a calibrated rate. Task-eval hasn't been run.
- **No time bounds.** Nous is designed for <25ms per turn but this is not contractual. Worst-case convergence is 5 iterations + clamp.
- **No ordering guarantee between sessions.** Two sessions with the same inputs and the same imported `LearnedState` will produce the same signals. But the moment `track_cost` or `process_response` quality values differ, state diverges.
- **No recovery from invalid input.** Clamping bounds are safety rails (CR4), not correctness guarantees. Passing `quality = 2.0` gets clamped to 1.0 silently — but downstream RPE math still runs on the clamped value.

---

## 7. See also

- `src/session.rs` — the API surface these semantics describe.
- `src/cognition/signals.rs` — signal computation, constants, gating.
- `src/types/world.rs` — `LearnedState` shape.
- Project-level identity (proven vs aspirational) — see the repo's
  `CLAUDE.md` on GitHub for the honest-scope table.

## 8. Path 2 Regulator contract (Sessions 16-20, 2026-04-15)

The contract above describes Path 1 (`CognitiveSession`). Path 2
(`Regulator`, `src/regulator/`) wraps Path 1 and adds an event-driven
interface for callers that want LLM-operational signals instead of
text-pattern-derived ones. Semantic differences worth knowing:

- **Input modality** — Path 2 consumes `LLMEvent` (token stream,
  corrections, cost, quality feedback) instead of raw user text. The
  wrapped `CognitiveSession` still runs, so Path 1 signals continue to
  update; Path 2 *adds* signals without replacing them.
- **New decisions** — `Regulator::decide()` returns a `Decision` enum
  with a locked P10 priority order: `CircuitBreak(CostCapReached) >
  CircuitBreak(QualityDeclineNoRecovery) > ScopeDriftWarn >
  ProceduralWarning > Continue`. See `src/regulator/mod.rs`
  `Regulator::decide` doc for the authoritative ordering.
- **Persistence** — `Regulator::export()` returns a `RegulatorState`
  (defined in `src/regulator/state.rs`) that wraps `LearnedState` plus
  a Path-2-only `correction_patterns: HashMap<String, CorrectionPattern>`
  field. `#[serde(default)]` on the new field means pre-Session-20
  snapshots load cleanly; old `RegulatorState` JSON produced before the
  `correction_patterns` field existed restores with an empty map.
- **Cost flow** — when a `LLMEvent::Cost` event is ingested, the
  Regulator both records raw counters in `CostAccumulator` AND calls
  `session.track_cost(normalize_cost(tokens_out, wallclock_ms))`,
  feeding Path 1 body-budget depletion from Path 2's observed cost.
  The Path 1 closed-loop contract in §2.3 above is satisfied
  automatically when callers use the Path 2 event surface.
- **P9b compliance** — Path 2 avoids English regex throughout. Pattern
  extraction from corrections uses opaque names
  (`corrections_on_{cluster}`) and raw-text passthrough for app / LLM
  interpretation. Confidence fallback uses language-neutral structural
  signals (length, `?` density) when logprobs are absent. See
  `docs/regulator-design.md` §11a for per-session implementation
  notes.

### 8.1 Regulator lifetime — per-task vs per-user (Session 24)

A single `Regulator` instance accumulates task-scoped state
(`CostAccumulator`, `ScopeTracker`, `TokenStatsAccumulator`,
`pending_response`) across every event. User-scoped state
(`LearnedState` + correction patterns) also accumulates, but on a
different lifetime.

Apps that serve **one task per regulator** (a conversation, a
long-running batch job, a multi-turn tool-use loop) keep one regulator
instance for the duration. Cost / quality / scope signals correctly
reflect the whole task.

Apps that serve **many independent tasks per user** (a retrieval agent
answering 50 unrelated queries, a per-user dashboard cycling through
widgets) must reset task-scoped state between tasks, otherwise the
first task that trips `CircuitBreak(QualityDeclineNoRecovery)` will
halt every subsequent task. The canonical reset is a round-trip
through `export()` / `import()`:

```rust
let snapshot = regulator.export();
regulator = Regulator::import(snapshot).with_cost_cap(COST_CAP);
```

`import()` rehydrates with `CostAccumulator::new()` (default cap), so
callers using a non-default cap must re-apply `with_cost_cap(...)`
every time. See `examples/task_eval_real_llm_regulator.rs` for the
canonical per-query reset pattern.

### 8.2 `decide()` timing — pre- vs post-generation (Session 23)

`decide()` is idempotent within a turn, but the `Decision` it returns
depends on which events have fired. Apps that want
`Decision::ProceduralWarning` to surface **before generation** must
call `decide()` after `TurnStart` and BEFORE `TurnComplete`. Rationale:

- After `TurnStart`, the scope tracker has task keywords but no
  response keywords, so `drift_score` returns `None` and
  `ScopeDriftWarn` skips (P10 priority: `ScopeDriftWarn >
  ProceduralWarning`).
- `ProceduralWarning` is cluster-keyed, not response-keyed, so it
  fires as soon as `TurnStart` sets the cluster.

Apps that probe `decide()` only after `TurnComplete` will see
`ScopeDriftWarn` dominate and hide the procedural warning. This isn't
a bug in `decide()` — it's a direct consequence of the locked priority
order plus when the scope tracker populates its response side.

### 8.3 Correction pattern persistence trade-off (Sessions 20, 24)

`Regulator::export()` preserves correction patterns only for clusters
that have reached `MIN_CORRECTIONS_FOR_PATTERN = 3`. Below-threshold
records (a cluster with 2 corrections) do NOT survive export — on
import they devolve to zero.

This is a documented trade-off (see `Regulator::export` Rustdoc) that
keeps the snapshot focused on stable learned rules. The practical
consequence: **pattern formation requires within-task correction
accumulation**, not cross-task. A user who makes 1 correction per
session for 3 sessions never reaches threshold if the agent
exports/imports between sessions; a user who makes 3 corrections in
one session does.

Apps that need cross-task correction accumulation below the threshold
should either:

- Not reset between tasks (keep one regulator across tasks),
  accepting that `CircuitBreak` predicates see cumulative quality /
  cost and may halt early.
- Store raw correction events externally and replay them into the
  regulator on startup (bypasses the built-in persistence entirely).

See `examples/regulator_correction_memory_demo.rs` for the within-task
accumulation path and `examples/task_eval_real_llm_regulator.rs`
(module doc, `procedural_warnings = 0` narration) for the cross-task
caveat.

### 8.4 `QualityFeedback` is load-bearing

Without a `QualityFeedback` event after each turn, the strategy
learning path inside the wrapped `CognitiveSession` never fires (§2.3
above). The buffered `TurnComplete` response sits un-consolidated;
`response_strategies` in `LearnedState` stays empty; `ProceduralWarning`
based on cluster memory still works, but Path 1's strategy
recommendations (exposed on `turn.signals.strategy`) don't come online.

This was implicit under Path 1 (`CognitiveSession::process_response` is
a single call with both response text and quality). It's explicit
under Path 2: `TurnComplete` delivers the text, `QualityFeedback`
delivers the score, and only the pair together consolidates.

If your app has no reliable quality signal, emit a conservative
neutral (`0.5`) rather than skipping the event — the consolidation
fires unconditionally on feedback, and a neutral score keeps the EMAs
parked near their prior instead of starving them entirely.
