# Single-Algorithm TUI Dashboard — Design

**Date**: 2026-06-10
**Status**: Approved (brainstorming complete with browser mockups, awaiting implementation plan)
**Scope**: Restyle the single-algorithm training TUI in `display.py` from one monochrome plain-text panel into a colored three-section dashboard (header / Optimization+Validation columns / footer), fix its known rendering bugs, and additively extend `compute_eval_summary` with cost/DV min·max and per-burn DV1/DV2/DV3 stats. Islands-mode layout untouched; its validation detail inherits the richer content through the shared formatter.

## 1. Motivation

The single-algorithm `LiveDisplay` panel (`_build_panel`, display.py:147-241) predates the islands work and never got its styling pass: it is entirely monochrome (no `border_style`, no `Text` styles) while `_update_islands` (243-345) uses green/cyan/bold panels. It also carries four real defects, confirmed by rendering the live path with synthetic records:

1. **Flat sparklines render blank** — `_sparkline` (24-31) maps the minimum bucket to a space, so a constant series (capture 100%) shows nothing and converged cost curves visually truncate.
2. **Sparkline columns misaligned by one char** — cost rows use 9-char labels + `:>10s` values, Capture/Diversity use 11-char labels + `:>9` fields (170-173).
3. **ETA wrong after resume** — progress = `gen / n_gens` with absolute gen but `_start_time` from this process; `set_start_gen` exists on the protocol but only the islands path consumes it (267).
4. **Best-params line prints every parameter untruncated** (234-238) — for NN runs that is hundreds-to-thousands of weights, blowing the panel (Live crops with ellipsis).

Beyond bugs, per-generation data already in the logger buffer is invisible: worst/median/σ cost, the full `all_costs` population distribution, `pool_metrics` (adaptive seed curator state), `gen_elapsed_s`. And the validation detail block lacks the spread (min/max) and the per-burn DV decomposition the final records carry.

## 2. Goals and Non-Goals

**Goals:**

1. Dashboard layout matching the approved browser mockup (`.superpowers/brainstorm/51349-1781120774/content/tui-final-v2.html`): header panel (identity + timing + progress), side-by-side Optimization (cyan) and Validation (green) panels, one-line footer.
2. Fix all four defects above; alignment guaranteed structurally via `Table.grid`, not hand-padding.
3. Show the new data: pop size, worst+σ, population cost histogram, gen wall time, seed-pool refresh gen.
4. Validation panel: aligned `min · p50 · p95 · max` grid for Cost, total DV, and DV1/DV2/DV3 (terminal-maneuver burns: periapsis / circularization / inclination — same semantics and naming as `chart` panel 16, charts.py:1115-1119).
5. Additive-only payload changes: existing JSONL consumers, islands panels, and the PDF report keep working unmodified.
6. Degradation behavior unchanged: `NoopDisplay` on `--no-tui` / non-TTY; `Live` at `refresh_per_second=2`; one content update per generation.

**Non-Goals:**

1. Islands-mode layout changes (its panels/columns/migration view stay as-is; only the shared validation formatter's content gets richer).
2. RL display (`rl/display.py`), report PDF charts, logger record schema changes beyond the additive `validation_summary` fields.
3. `rich.progress.Progress` adoption — the repo convention is hand-rolled unicode rendering inside `Live`; the new bar is a styled `Text` line (`━`/`╸` glyphs), not a Progress widget.
4. Terminal-width adaptive layouts — `Columns` wraps naturally on narrow terminals; no breakpoint logic.

## 3. Layout

```
╭─ ftc · qpso │ Gen 740/2000 │ pop 64 │ elapsed 10m 12s │ 1.21 gen/s │ ETA 17m 22s ─╮
│ ━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37%                   │
╰───────────────────────────────────────────────────────────────────────────────────╯
╭─ Optimization ───────────────────────╮ ╭─ Validation (1000 sims · g733) ──────────╮
│ Best    3.3450e+02  █▅▃▂▂▁▁▁▁▁▁▁     │ │ Best  RMS 1.6840e+02  g704               │
│ Mean    4.8007e+02  █▅▃▁▁▁▁▁▁▁▁▁     │ │ Last  RMS 1.8120e+02  REJECTED           │
│ Worst   2.1300e+04  σ 3.1e+03        │ │ Cap   968/1000 (96.8%)                   │
│ Capture   100%  ▄▄▄▄▄▄▄▄▄▄▄▄         │ │        min      p50      p95      max    │
│ Divers    0.42  ▄▁█▄▂▅▆▃▂▄▃▅         │ │ Cost   38.1    112.4    387.2   1.2e4    │
│ Pop cost  ▂▆█▅▂▁·▁  3e2→2e4 log      │ │ DV     62.0    118.2    342.0   980.4    │
│ gen wall 0.83s · pool refresh g720   │ │ DV1    41.3     84.3    211.0   512.0    │
╰──────────────────────────────────────╯ │ DV2     8.0     22.1     88.7   190.2    │
                                         │ DV3     0.4      9.4     61.2   145.0    │
                                         │ Apo    p50 41.2 · p95 96.0 km            │
                                         │ Q      max 204.9   2.1% > 200            │
                                         │ G      max 11.2    0% > 15               │
                                         ╰──────────────────────────────────────────╯
 Stagnant 36 gens · improved g704 · best: gain 1.24, tau 8.31, thr 2.05 (+14 more)
```

**Header panel** (border green, like the islands header): `{scheme} · {algorithm}` bold, `Gen g/N`, `pop n`, `elapsed`, `rate gen/s`, `ETA` — all via the existing `_format_duration` (display.py:91-101). Second line: progress bar (`━` filled blue, `╸` tip, dim `━` remainder) + bold percent. Rate and ETA are resume-aware: `rate = (gen - start_gen) / elapsed`, `ETA = (n_gen - gen) / rate`, exactly the islands header math (267-271).

**Optimization panel** (border cyan). A `Table.grid` with three columns (label, value, sparkline/extra):
- `Best` / `Mean`: `_format_cost` value + 30-char cost sparkline (cyan).
- `Worst`: value + dim `σ {std_cost:.1e}` (new data).
- `Capture`: percent (green) + sparkline (green).
- `Divers`: 2-dp value + sparkline (magenta).
- `Pop cost`: 16-bin log-spaced histogram of finite `all_costs` (blue) + dim range caption `lo→hi log`; non-finite entries appended as dim `∞×k` when present; `·` glyph for empty bins.
- Dim status row: `gen wall {gen_elapsed_s}s · pool refresh g{last_curation_gen}` (each fragment only when its field is present in the latest record).

**Validation panel** (border green). Built from the latest record carrying `validation`/`validation_summary` (same selection logic as today, display.py:190-222):
- `Best`/`Last` headline rows (green RMS for best; PROMOTED green / REJECTED yellow).
- `Cap n/N (pct)` (green when ≥ previous best capture, plain otherwise — simple, no history scan: green when pct ≥ 95, else plain).
- The stats grid (`Table.grid`, right-aligned numeric columns): header row dim `min p50 p95 max`; rows `Cost`, `DV`, `DV1`, `DV2`, `DV3` — values from the extended summary (section 4.2). Cost max colored yellow when > 10× p95 (outlier hint).
- `Apo` row: `p50 · p95 km` (existing fields).
- Constraint rows `Q`/`G`/`HL`: `max` + violation fragment — red `{viol_pct}% > {limit}` when viol_pct > 0, dim `0% > {limit}` otherwise (same data as `_format_validation_summary` today, display.py:39-88).
- Pre-gate / validation-off placeholder: dim `waiting for first validation…` (+ explanation line) when no record carries `validation`.

**Footer** (plain line, no panel): yellow `Stagnant N gens` (or `No improvement yet`), dim `· improved g{N}`, dim best-params preview — first 3 params + `(+N more)` for non-NN schemes; `"{n} NN params (best_model.json)"` when `scheme == "neural_network"`.

## 4. Architecture

### 4.1 `display.py` changes

- `_build_panel` is replaced by `_build_dashboard(logger, current_run) -> Group` composed of `_build_header(...) -> Panel`, `_build_optimization_panel(records) -> Panel`, `_build_validation_panel(records) -> Panel | None`, `_build_footer(records) -> Text` — each independently testable, all pure functions of the logger buffer + constructor state. `update()` dispatch unchanged.
- New constructor args: `algorithm: str = ""` (shown in the header; train.py passes `config.optimizer.algorithm`) and pop size derived per-update from `len(record["all_costs"])` (no constructor arg; falls back to omitting the fragment when absent).
- `_sparkline` fix: flat series renders the mid glyph `▄` (replace the `span = 1.0` branch's output, display.py:24-31). Width stays 30 in the Optimization panel.
- New `_cost_histogram(all_costs, bins=16) -> tuple[str, str]` (glyph string, dim caption): log10-spaced bins between finite min/max (guarding min==max and empty), `·` for zero bins, same `▁▂▃▄▅▆▇█` ramp; returns the `∞×k` suffix in the caption when non-finite entries exist.
- Resume-aware ETA: `train.py` calls `display.set_start_gen(start_gen)` on the single-algo path (mirroring the islands call); header math uses `_start_gen` like `_update_islands` does.
- `_format_validation_summary` (39-88) is retired in favor of `_validation_summary_rows(summary) -> list[tuple[str, list[str], str]]` (label, cell strings, style hint `""|"dim"|"red"|"yellow"`) — a data-shaping helper consumed by BOTH the new single-algo validation panel and the islands per-island detail panels (which keep their Panel/Columns layout but render the rows through the same helper, picking up the new min/max + DV1-3 rows). The islands panels' *structure* is untouched.
- Single-algo panel title loses the vestigial `Run 1/1` fragment (`n_runs` stays in the constructor signature for islands compatibility).

### 4.2 `compute_eval_summary` extension (`report.py:188`)

Additive fields only:

- `cost: {p50, p95, rms}` → add `min`, `max` (over the same `per_sim_costs`).
- `captured.dv: {p50, p95, mean}` → add `min`, `max` (over the same clipped DV array — min/max share the existing `[DV_FLOOR, DV_CAP]` clip for row consistency; a pathological outlier therefore shows as the cap value, same as p95 today).
- New `captured.dv1`, `captured.dv2`, `captured.dv3`: `{min, p50, p95, mean, max}` over `|cap[:, _FR_DV1..3]|` (absolute values, matching `chart_burn_dv_histograms`' `np.abs` convention, charts.py:1115-1119). Index constants imported from `charts` (`_FR_DV1 = 37`, `_FR_DV2 = 38`, `_FR_DV3 = 39`).
- `captured is None` when `n_captured == 0`, unchanged — the TUI grid rows render dim `—` placeholders in that case.

Consumers verified additive-safe: the JSONL `validation_summary` payload grows; `display.py` formatters read known keys; `report.py`'s own PDF table and `print_eval_summary` are untouched (they may adopt the new fields later, out of scope).

### 4.3 `train.py` touchpoints

Two one-liners: `create_display(..., algorithm=config.optimizer.algorithm)` and `display.set_start_gen(start_gen)` after resume restore on the single-algo path. The islands path already passes everything it needs.

## 5. Edge States

| State | Rendering |
|---|---|
| Empty logger buffer | Header panel renders (gen 0, no rate/ETA fragments) above a single dim `Waiting for first generation…` line; no Optimization/Validation panels. |
| No validation yet / `validation_n_sims = 0` | Validation panel body: dim `waiting for first validation…` + explanation line. |
| `captured is None` (zero captures in validation) | Stats grid rows show dim `—`; Cap row red `0/N (0%)`. |
| Flat metric series | `▄` midline sparkline (fix #1). |
| `< 30` records | Sparkline naturally shorter, left-aligned (unchanged behavior, now consistent across rows). |
| `all_costs` all non-finite | Histogram row: dim `no finite costs (∞×N)`. |
| NN scheme | Footer shows `{n} NN params (best_model.json)`; no param listing. |
| Resume | Header rate/ETA over `(gen - start_gen)`; first post-resume update may show rate `0.00` for one tick (same guard as islands, display.py:267). |
| Narrow terminal | `Columns` wraps the two panels vertically; no special handling. |

## 6. Testing

- Update `tests/test_training_display.py`: the `_build_panel` pin (test:31) moves to `_build_dashboard` and the section builders.
- New unit tests (pure, no Rust): `_sparkline` flat-series midline; `_cost_histogram` (log binning, empty/flat/non-finite inputs, `·` empty bins, `∞×k` caption); `_validation_summary_rows` (full summary, `captured=None`, missing constraint limits); footer param truncation (3 + `(+N more)`, NN suppression); header ETA math with `start_gen` offset.
- `compute_eval_summary` tests: new keys present, dv1/2/3 stats match hand-computed values on a synthetic `(n, 52)` final-records matrix, `captured=None` path, additivity (old keys byte-identical for the same input).
- Render snapshot test: build the dashboard from synthetic records on `Console(record=True, width=120)`, assert key fragments present (`pop 64`, `ETA`, `DV2`, `REJECTED`, histogram glyphs) — content assertions, not full-frame golden (Rich version drift would make a byte-golden brittle).
- Islands regression: existing islands display tests must pass with the shared `_validation_summary_rows` swap.

## 7. Out of Scope

- RL display unification (different lifecycle conventions, noted during exploration).
- `report.py` PDF adoption of the new summary fields.
- The pre-existing islands `_update_islands` origin-stats indentation wart (display.py:312) — separate cosmetic fix, not blocking this layout.
