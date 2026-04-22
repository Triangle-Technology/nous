# Calibrating the scope-drift threshold for your workload

`Regulator::with_scope_drift_threshold(f64)` defaults to **0.5**. That
cut-off was tuned against a 10-case hand-crafted audit (Session 18) and
holds up well on terse models (Claude Haiku, Ollama phi3). On verbose
models that tend to expand task keywords with peripheral vocabulary
(Gemini, GPT-4o-mini in chatty mode), the default can produce
false-positive `ScopeDriftWarn` emissions on on-topic responses scoring
in the `[0.5, 0.7)` band.

This doc walks through the data-driven process for deciding whether
your workload is better served by a non-default threshold, using the
`shadow_replay` CLI shipped in the crate.

## When to calibrate

Calibrate once you have at least **a few hundred real turns of shadow
data** from your target model mix. Calibrating against the default test
cases in the crate is not useful — the synthetic audit is already
optimised for them.

Signs the default needs adjustment:

- `Decision::ScopeDriftWarn` fires on responses your reviewers rate
  on-topic. Raise the threshold.
- Truly drifted responses ship to users without warning. Lower the
  threshold (rare — the default is already generous).
- Verbose-model pedagogical answers (multiple paragraphs of background
  before the concrete answer) routinely trigger warnings.

## Prerequisites

1. **Shadow data in JSONL form.** Each row carries `turn_id`,
   `event_idx`, `event_type`, `payload`. See the
   `examples/shadow_replay.rs` module doc for the exact shape. If your
   data lives in Cloudflare D1, Postgres, or a log file, a small `jq`
   reshape is all that stands between you and the CLI.
2. **At least 200 turns of data** for a meaningful distribution. Below
   that, histogram bin counts are noisy; percentile estimates are
   unreliable.
3. **A diverse model mix** matching production. If you only sample one
   model, you'll calibrate for that model only.

## Step-by-step workflow

### Step 1: Export shadow data

Format depends on where you stored it. A D1 export looks like:

```bash
wrangler d1 execute YOUR_DB --remote --json \
  --command "SELECT turn_id, event_idx, event_type, payload_json
             FROM llm_events
             WHERE turn_ts >= '2026-04-20'
             ORDER BY turn_id, event_idx" \
  | jq -c '.[0].results[]
           | {turn_id, event_idx, event_type,
              payload: (.payload_json | fromjson)}' \
  > shadow.jsonl
```

Postgres / MySQL / SQLite equivalents are the same shape — emit one
JSON object per event with the four fields above.

### Step 2: Run shadow_replay with threshold 0.0

Setting the threshold to `0.0` forces every non-empty turn to emit
`ScopeDriftWarn` regardless of how low the drift score is, so the CLI
captures the **complete distribution** rather than only the
above-default tail.

```bash
cargo run --release --example shadow_replay -- \
  --scope-threshold=0.0 --summary-only < shadow.jsonl \
  > summary.json
```

`--summary-only` skips the per-event output so the only line on stdout
is the aggregate. Output is a single JSON object with a
`drift_score_stats` field.

### Step 3: Inspect `drift_score_stats`

A typical result on 300 turns of shadow data:

```json
{
  "_summary": {
    "turns": 300,
    "events": 1207,
    "decisions": { "Continue": 707, "ScopeDriftWarn": 298, "...": 2 },
    "drift_score_stats": {
      "count": 298,
      "min": 0.142,
      "mean": 0.548,
      "p50": 0.555,
      "p95": 0.889,
      "max": 1.0,
      "histogram": [5, 18, 42, 78, 54, 62, 27, 10, 1, 1],
      "threshold_sweep": [
        ["0.3", 248], ["0.4", 190], ["0.5", 148], ["0.6", 70],
        ["0.7", 31], ["0.8", 12], ["0.9", 2]
      ]
    }
  }
}
```

Read the threshold sweep as: *"at threshold X, Y of my 298 non-empty
turns would fire `ScopeDriftWarn`."* The default (0.5) fires on 148
turns — nearly half. If human review shows most of those are on-topic,
the default is too aggressive for this workload.

### Step 4: Decide the new default

Pick the lowest threshold that fires on the turns your reviewers
*actually* want flagged. Three concrete heuristics:

- **Match human labels on a sample.** Pick 50 high-drift turns, have a
  reviewer tag them as "off-topic" or "on-topic but verbose". The
  right threshold puts the inflection between those two classes.
- **Target a fixed flag rate.** If you want roughly 10% of turns to
  fire, look at the sweep and pick the threshold whose count is ~10%
  of `turns`. In the example above, 30 / 300 = 10% sits at 0.7.
- **Use the p95.** For noisy verbose workloads, p95 often coincides
  with "genuinely drifted" — responses beyond the 95th percentile are
  hard to miss even to a casual reviewer.

The sample above suggests **0.7** for this model mix. The default 0.5
flags 148 turns (49%); 0.7 flags 31 (10%). Both numbers need to be
validated against real human review before shipping.

### Step 5: Apply in code

Once you have a number:

```rust
let regulator = Regulator::for_user("alice")
    .with_scope_drift_threshold(0.7)
    .with_cost_cap(10_000);
```

Python:

```python
regulator = Regulator.for_user("alice")
regulator.with_scope_drift_threshold(0.7)
regulator.with_cost_cap(10_000)
```

Node:

```typescript
const r = Regulator.forUser('alice')
r.withScopeDriftThreshold(0.7)
r.withCostCap(10_000)
```

### Step 6: Verify with `metrics_snapshot`

Confirm the new value is live in observability:

```rust
assert_eq!(
    regulator.metrics_snapshot().get("noos.scope_drift_threshold"),
    Some(&0.7),
);
```

## Alternative modes

### Simulating a proposed default without editing code

To preview what a specific threshold would do on historical data
without touching application code:

```bash
cargo run --release --example shadow_replay -- \
  --scope-threshold=0.7 < shadow.jsonl \
  > simulation.jsonl
```

The per-event output lets you diff against the production run
(threshold 0.5) — count the decisions that flipped, spot-check which
turns moved from `ScopeDriftWarn` to `Continue`.

### Comparing two models

Run the CLI twice with the traffic partitioned by provider:

```bash
jq -c 'select(.payload.provider == "gemini")' shadow.jsonl \
  | cargo run --release --example shadow_replay -- \
      --scope-threshold=0.0 --summary-only

jq -c 'select(.payload.provider == "anthropic")' shadow.jsonl \
  | cargo run --release --example shadow_replay -- \
      --scope-threshold=0.0 --summary-only
```

Compare the two `drift_score_stats.mean` values. If Gemini runs at
mean 0.55 and Anthropic at mean 0.32, the single-threshold policy is
compromising somewhere — either configure two `Regulator` instances
keyed on provider, or pick a threshold that works on the worse
distribution and accept a slightly higher FP rate on the better one.

## Re-calibration cadence

The threshold is not a one-shot decision. Re-run the workflow when:

- You switch model families (Anthropic → Google, etc.).
- You change the system prompt in a way that changes response style
  (e.g. from terse-answer to pedagogical-tutor).
- Your review team starts flagging either too many false positives
  (raise threshold) or missed drifts (lower it).
- You cross roughly 10× the shadow-data volume used for the last
  calibration — more data sharpens the distribution tails and may
  reveal the current choice is too far from the optimum.

## Related tunables

Threshold is not the only knob. The `Regulator` also exposes:

- `with_cost_cap(u32)` — upper bound on cumulative `tokens_out` per
  regulator lifetime. Hard stop when exceeded and recent quality is
  poor.
- `with_implicit_correction_window(Duration)` — temporal + topic
  proximity window for auto-detecting retries as corrections.

Both are covered in [`regulator-guide.md`](regulator-guide.md).
Calibrating them follows a similar shadow-replay loop, but the
relevant summary fields are `circuit_break_reasons` and
`turns_with_procedural_warning` rather than `drift_score_stats`.

## FAQ

**Why not ship a smarter default?** Model behaviour varies too much
across providers and system-prompt styles for a single value to be
right everywhere. The `0.5` default is a defensible median for terse
models; a smarter default would require user-supplied workload
metadata we don't want to collect.

**Is a non-keyword drift metric coming?** Embedding-based drift is
tracked as post-0.4.0 work. The keyword metric has documented failure
modes (synonym mismatch, rank-21 truncation — see
`src/regulator/scope.rs` adversarial tests) that embedding scoring
would address. Trade-off: embedding adds an LLM call (or local model
load) per turn, inflating the 44ns-per-event overhead by ~6 orders of
magnitude.

**Can I calibrate without shipping shadow infrastructure first?** Yes —
you can reuse an existing observability pipeline's event stream.
`shadow_replay` accepts any JSONL shape that matches the `LLMEvent`
contract; you don't need to stand up a dedicated D1 table. See the
OTel GenAI mapping in `src/regulator/otel.rs` for one concrete bridge.
