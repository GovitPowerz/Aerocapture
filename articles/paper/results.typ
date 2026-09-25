// Compile-time accessors over the committed bundle products, so the headline tables of
// paper.typ read data/*.json instead of quoting transcribed literals (issue #120).
//   run(key)      -> a results.json run summary (n = 1000 final-evaluation pool)
//   paired(key)   -> a results.json paired comparison
//   conf(label)   -> a confirmatory_eval.json cell (10 x 100 000 frozen pool)
//   finalist(l)   -> a quant/finalists_results.json row (re-quote pool, n = 1000)
//   marg(label)   -> a confirmatory_marginal.json cell (10 x 100 000, per-scenario noise)
//   ou(label)     -> a quote_marginal.json cell (paired n = 1000; regime: "marginal" = one noise
//                    path per scenario, "frozen" = the shared path)
// Every run and confirmatory cell of results.json / confirmatory_eval.json used by the headline
// tables was flown under the legacy (shared-path) noise regime; legacy_regime() asserts it so a
// re-quoted run cannot enter a table whose caption states that regime. confirmatory_marginal.json
// is per_draw throughout, asserted at load (ADR-0003 / ADR-0006, issue #137). quote_marginal.json
// (issue #157) pins legacy seeding for both of its regimes and reaches the per-scenario one through
// a per-seed override of simulation.random_seed, asserted at load.

#let results = json("data/results.json")
#let confirmatory = json("data/confirmatory_eval.json")
#let finalists = json("data/quant/finalists_results.json").finalists
#let marginal = json("data/confirmatory_marginal.json")
#assert(marginal.noise_seeding == "per_draw", message: "confirmatory_marginal.json is not the per_draw regime")
#let quotes = json("data/quote_marginal.json")
#assert(quotes.regimes.keys() == ("frozen", "marginal") and quotes.regimes.values().all(r => r.noise_seeding == "legacy")
  and quotes.regimes.frozen.per_seed_override == none and quotes.regimes.marginal.per_seed_override != none,
  message: "quote_marginal.json does not carry the frozen / marginal pair of legacy-seeded regimes")

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
