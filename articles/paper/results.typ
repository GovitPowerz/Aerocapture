// Compile-time accessors over the committed bundle products, so the headline tables of
// paper.typ read data/*.json instead of quoting transcribed literals (issue #120).
//   run(key)      -> a results.json run summary (n = 1000 final-evaluation pool)
//   paired(key)   -> a results.json paired comparison
//   conf(label)   -> a confirmatory_eval.json cell (10 x 100 000 frozen pool)
//   finalist(l)   -> a quant/finalists_results.json row (re-quote pool, n = 1000)
//   marg(label)   -> a confirmatory_marginal.json cell (10 x 100 000, per-scenario noise)
//   ou(label)     -> a quote_marginal.json cell (paired n = 1000; regime: "marginal" = one noise
//                    path per scenario, "frozen" = the shared path)
//   slope(cell, ceiling, eval)  -> a heat_load_slope.json row: the #192 ceiling leg `cell` ("mamba_p962" /
//                    "dense_p515") fine-tuned under ceiling "q25" / "q27" / "q30", scored at its "own"
//                    ceiling or at the "v4" limit (paired n = 1000, the marginal regime of ou())
//   slope_summary(cell)  -> its CVaR95 slope per MJ/m2 and the source seeds' CVaR95 spread
//   centered(label, regime)          -> a centered_depth.json cell (9M stress pool, n = 10 000, high
//                                       regime; regime "per_draw" or "legacy", the file's pair, asserted; #156)
//   centered_paired(a, b, regime)    -> its paired seed-versus-baseline deltas under one regime
//   bench(label)  -> a compute_benchmark.json row (one idle core, wall-clock; NN-mamba, NN-dense,
//                    FTC, FNPAG): ms_per_sim, n_guidance_updates, us_per_update_incl_sim, and
//                    FNPAG's ms_per_replan_derived
//   probe_ref(p, ref, arm)  -> an Appendix B probe's higher-budget reference row against its
//                    retrained in-regime baseline arm (data/probes/<p>_probe_results.json): p95 of
//                    the reference, three-seed mean p95 of the arm and its sigma_run
//   span(xs)      -> the min--max range of a list of numbers, as math (f: fixed or signed, d decimals)
// The regime is part of the number (ADR-0003 / ADR-0006): a results.json run enters a table only
// through legacy_regime(key) or per_draw_regime(key), each asserting the run's noise_seeding, so a
// shared-path (development-regime) run cannot enter a per-scenario table or the reverse. The
// confirmatory_eval.json cells are legacy throughout; confirmatory_marginal.json is per_draw
// throughout, asserted at load (issues #137, #174). quote_marginal.json
// (issue #157) pins legacy seeding for both of its regimes and reaches the per-scenario one through
// a per-seed override of simulation.random_seed, asserted at load on the protocol record its source
// writes (`regimes`, `seed_pool`; issue #166), so the assert tests the scoring script, not the extract;
// heat_load_slope.json (issue #192) carries the same marginal protocol record and is asserted the same way.

#let results = json("data/results.json")
#let confirmatory = json("data/confirmatory_eval.json")
#let finalists = json("data/quant/finalists_results.json").finalists
#let marginal = json("data/confirmatory_marginal.json")
#assert(marginal.noise_seeding == "per_draw", message: "confirmatory_marginal.json is not the per_draw regime")
#let quotes = json("data/quote_marginal.json")
#assert(quotes.regimes.keys() == ("frozen", "marginal") and quotes.regimes.values().all(r => r.noise_seeding == "legacy")
  and quotes.regimes.frozen.per_seed_override == none and quotes.regimes.marginal.per_seed_override != none,
  message: "quote_marginal.json does not carry the frozen / marginal pair of legacy-seeded regimes")
#let heat_load = json("data/heat_load_slope.json")
#assert(heat_load.regime.noise_seeding == "legacy" and heat_load.regime.per_seed_override != none
  and heat_load.regime.per_seed_override == quotes.regimes.marginal.per_seed_override
  and heat_load.seed_pool == quotes.seed_pool,
  message: "heat_load_slope.json is not the marginal regime of quote_marginal.json (legacy seeding, per-scenario random_seed override, the same seed pool)")
#let centered_depth = json("data/centered_depth.json")
#assert(centered_depth.regimes.keys() == ("per_draw", "legacy")
  and centered_depth.regimes.per_draw.at("monte_carlo.noise_seeding") == "per_draw"
  and centered_depth.regimes.legacy.at("monte_carlo.noise_seeding") == "legacy",
  message: "centered_depth.json does not carry the per_draw / legacy regime pair")
#let benchmark = json("data/compute_benchmark.json")
#assert(benchmark.single_core, message: "compute_benchmark.json is not the single-core benchmark the prose quotes")

#let run(key) = results.runs.at(key)
#let paired(key) = results.paired.at(key)
#let conf(label) = {
  let cell = confirmatory.cells.find(c => c.label == label)
  assert(cell != none, message: "confirmatory_eval.json has no cell " + label)
  cell
}
#let finalist(label) = {
  let row = finalists.find(f => f.label == label)
  assert(row != none, message: "finalists_results.json has no row " + label)
  row
}

// A per-scenario confirmatory cell, pooled over the full pool (asserted): n, capture % from
// n_captured / n (the source's capture_pct is rounded to 2 decimals, 100.0 for 5 losses), the
// scenarios lost, the pooled CVaR95 / CVaR99.9 / worst case / constraint-violation %, and the
// CVaR99.9 standard error over the replicates.
#let marg(label) = {
  let cell = marginal.cells.find(c => c.label == label)
  assert(cell != none, message: "confirmatory_marginal.json has no cell " + label)
  let p = cell.pooled
  assert(p.n == marginal.n_replicates * marginal.n_per_replicate, message: label + " does not cover the full confirmatory pool")
  (n: p.n, capture_pct: 100 * p.n_captured / p.n, lost: p.n - p.n_captured, cvar95: p.cvar95, cvar999: p.cvar999,
    cvar999_se: cell.replicate_stats.cvar999.se, max: p.max, viol_pct: p.viol_pct)
}
// A paired n = 1000 cell of the OU-marginal campaign: capture %, CVaR95 of the correction DV over
// captured scenarios, heat-load violation %, under the per-scenario ("marginal") or the
// shared-path ("frozen") regime.
#let ou(label, regime: "marginal") = {
  let key = label + "/" + regime
  assert(key in quotes.cells, message: "quote_marginal.json has no cell " + key)
  let c = quotes.cells.at(key)
  (capture_pct: c.capture_pct, cvar95: c.dv_cvar95, viol_pct: c.heat_load_viol_pct)
}
// A #192 ceiling leg scored on the paired n = 1000 pool: the ceiling it flew under (kJ/m2), capture %,
// CVaR95 and worst case of the correction DV, heat-load p95 / max (MJ/m2) and the violation %.
#let slope(cell, ceiling, eval: "own") = {
  let key = "hs_" + cell + "_" + ceiling + "/" + eval
  assert(key in heat_load.rows, message: "heat_load_slope.json has no row " + key)
  let r = heat_load.rows.at(key)
  (ceiling: r.ceiling_kj_m2, capture_pct: r.capture_pct, cvar95: r.dv_cvar95, max: r.dv_max,
    heat_load_p95: r.heat_load_p95_mj_m2, heat_load_max: r.heat_load_max_mj_m2, viol_pct: r.heat_load_viol_pct)
}
#let slope_summary(cell) = {
  assert(cell in heat_load.summary, message: "heat_load_slope.json has no summary for " + cell)
  heat_load.summary.at(cell)
}
// A centered high-regime cell at sizing depth, under one of the file's two regimes: n, capture %
// (+ CI), the conditional DV statistics (+ CIs) of run_stats.
#let centered(label, regime) = {
  let cell = centered_depth.cells.at(regime).find(c => c.label == label)
  assert(cell != none, message: "centered_depth.json has no " + regime + " cell " + label)
  cell
}
// Its paired a-versus-b record: capture-rate and conditional-CVaR95 deltas with bootstrap CIs
// (negative CVaR95 delta = a's tail is better), the both-captured mean delta, win rate, Wilcoxon p.
#let centered_paired(a, b, regime) = {
  let p = centered_depth.paired.at(regime).find(p => p.a == a and p.b == b)
  assert(p != none, message: "centered_depth.json has no " + regime + " pair " + a + " - " + b)
  p
}
#let bench(label) = {
  let row = benchmark.schemes.find(s => s.label == label)
  assert(row != none, message: "compute_benchmark.json has no row " + label)
  row
}
#let probe_ref(probe, ref, arm) = {
  let d = json("data/probes/" + probe + "_probe_results.json")
  assert(ref in d.references and arm in d.arms, message: probe + "_probe_results.json has no reference " + ref + " or arm " + arm)
  (ref: d.references.at(ref).dv_p95, base: d.arms.at(arm).dv_p95.mean, sd: d.arms.at(arm).dv_p95.std)
}
// Mean and sample sd (n - 1) of a list of numbers.
#let mean_sd(xs) = {
  let mean = xs.sum() / xs.len()
  (mean: mean, sd: calc.sqrt(xs.map(x => calc.pow(x - mean, 2)).sum() / (xs.len() - 1)))
}

// The legacy noise regime, asserted for a results.json run (the regime is part of the number).
#let legacy_regime(key) = {
  let r = run(key)
  assert(r.noise_seeding == "legacy" and not r.legacy_prefix_regime, message: key + " is not a legacy-regime run")
  r
}
// The per-scenario (per_draw) regime, asserted the same way: the v4 performance-table rows read
// results.json through this accessor only, so a shared-path run cannot enter them.
#let per_draw_regime(key) = {
  let r = run(key)
  assert(r.noise_seeding == "per_draw" and not r.legacy_prefix_regime, message: key + " is not a per-scenario (per_draw) run")
  r
}

// Fixed-point string with d decimals, half away from zero, U+2212 minus.
#let fixed(x, d: 1) = {
  let base = calc.pow(10, d)
  let scaled = int(calc.round(calc.abs(x) * base))
  let frac = str(calc.rem(scaled, base))
  while frac.len() < d { frac = "0" + frac }
  (if x < 0 { "\u{2212}" } else { "" }) + str(calc.quo(scaled, base)) + (if d > 0 { "." + frac } else { "" })
}
// Same with an explicit sign, for paired deltas.
#let signed(x, d: 1) = (if x < 0 { "\u{2212}" } else { "+" }) + fixed(calc.abs(x), d: d)
// "lo, hi" of a two-element interval, signed.
#let ci(iv, d: 1) = signed(iv.at(0), d: d) + ", " + signed(iv.at(1), d: d)
// "min--max" of a list of numbers, each end formatted by f (fixed or signed) with d decimals.
#let span(xs, f: fixed, d: 0) = [$#f(calc.min(..xs), d: d)$--$#f(calc.max(..xs), d: d)$]
// One significant digit times a power of ten, as an equation (x > 0).
#let sci(x) = {
  assert(x > 0, message: "sci() needs x > 0, got " + repr(x) + " (a zero rate needs prose, not a power of ten)")
  let e = calc.floor(calc.log(x, base: 10))
  let m = int(calc.round(x / calc.pow(10.0, e)))
  if m == 10 { m = 1; e += 1 }
  $#m times 10^(#e)$
}
// Wilcoxon p: the saturated normal-approximation statistic (~1e-165 at sign unanimity) is shown
// as "< 1e-15" as the tbl-paired caption states; resolved values keep one significant digit
// below 0.01 and two decimals above.
#let pval(p) = if p < 1e-100 { $< 10^(-15)$ } else if p < 0.01 { sci(p) } else { $#fixed(p, d: 2)$ }
