// Compile-time accessors over the committed bundle products, so the headline tables of
// paper.typ read data/*.json instead of quoting transcribed literals (issue #120).
//   run(key)      -> a results.json run summary (n = 1000 final-evaluation pool)
//   paired(key)   -> a results.json paired comparison
//   conf(label)   -> a confirmatory_eval.json cell (10 x 100 000 frozen pool)
//   finalist(l)   -> a quant/finalists_results.json row (re-quote pool, n = 1000)
// Every run and confirmatory cell used by the headline tables was flown under the legacy
// (shared-path) noise regime; legacy_regime() asserts it so a re-quoted run cannot enter a
// table whose caption states that regime (ADR-0003 / ADR-0006).

#let results = json("data/results.json")
#let confirmatory = json("data/confirmatory_eval.json")
#let finalists = json("data/quant/finalists_results.json").finalists

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
// below 1e-3 and two decimals above.
#let pval(p) = if p < 1e-100 { $< 10^(-15)$ } else if p < 1e-3 {
  let e = calc.floor(calc.log(p, base: 10))
  let m = int(calc.round(p / calc.pow(10.0, e)))
  if m == 10 { m = 1; e += 1 }
  $#m times 10^(#e)$
} else { $#fixed(p, d: 2)$ }
