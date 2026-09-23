// Compile-time accessors over the committed bundle products, so the headline tables of
// paper.typ read data/*.json instead of quoting transcribed literals (issue #120).
//   run(key)      -> a results.json run summary (n = 1000 final-evaluation pool)
//   paired(key)   -> a results.json paired comparison
//   conf(label)   -> a confirmatory_eval.json cell (10 x 100 000 frozen pool)
//   finalist(l)   -> a quant/finalists_results.json row (re-quote pool, n = 1000)
//   marg(label)   -> a confirmatory_marginal.json cell (10 x 100 000, per-scenario noise)
// Every run and confirmatory cell of results.json / confirmatory_eval.json used by the headline
// tables was flown under the legacy (shared-path) noise regime; legacy_regime() asserts it so a
// re-quoted run cannot enter a table whose caption states that regime. confirmatory_marginal.json
// is per_draw throughout, asserted at load (ADR-0003 / ADR-0006, issue #137).

#let results = json("data/results.json")
#let confirmatory = json("data/confirmatory_eval.json")
#let finalists = json("data/quant/finalists_results.json").finalists
#let marginal = json("data/confirmatory_marginal.json")
#assert(marginal.noise_seeding == "per_draw", message: "confirmatory_marginal.json is not the per_draw regime")

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

// A per-scenario confirmatory cell, pooled over the 10^6: capture % from n_captured / n (the
// source's capture_pct is rounded to 2 decimals, 100.0 for 5 losses), the scenarios lost, the
// pooled CVaR95 / CVaR99.9 / worst case and the CVaR99.9 standard error over the ten replicates.
#let marg(label) = {
  let cell = marginal.cells.find(c => c.label == label)
  assert(cell != none, message: "confirmatory_marginal.json has no cell " + label)
  let p = cell.pooled
  (capture_pct: 100 * p.n_captured / p.n, lost: p.n - p.n_captured, cvar95: p.cvar95, cvar999: p.cvar999,
    cvar999_se: cell.replicate_stats.cvar999.se, max: p.max)
}
// Mean and sample sd (n - 1) of one marg() field over the three fine-tune seeds of the deployed Mamba.
#let mamba_seeds(field) = {
  let xs = ("ou_marginal/ft_mamba_p962", "ou_marginal/ft_mamba_p962_s2", "ou_marginal/ft_mamba_p962_s3").map(l => marg(l).at(field))
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
// Wilcoxon p: the saturated normal-approximation statistic (~1e-165 at sign unanimity) is shown
// as "< 1e-15" as the tbl-paired caption states; resolved values keep one significant digit
// below 0.01 and two decimals above.
#let pval(p) = if p < 1e-100 { $< 10^(-15)$ } else if p < 0.01 {
  let e = calc.floor(calc.log(p, base: 10))
  let m = int(calc.round(p / calc.pow(10.0, e)))
  if m == 10 { m = 1; e += 1 }
  $#m times 10^(#e)$
} else { $#fixed(p, d: 2)$ }
