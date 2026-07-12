// Appendix A: per-scheme mission reports. Two pages per scheme.
// Data: figures/appendix/<slug>/{*.svg, stats.json} (built by scripts/collect_appendix.py).
// Report-style panels reuse the training pipeline's charts.py output verbatim.

#let apx = "figures/appendix/"

#let fnum(x, d: 1) = if x == none { "--" } else { calc.round(x * 1.0, digits: d) }

// One right-aligned stats row: label + up to four value cells.
#let srow(label, ..vals) = (
  [#label], ..vals.pos().map(v => align(right)[#v])
)

#let scheme_report(slug, title) = {
  let s = json(apx + slug + "/stats.json")
  let cap = s.captured
  let con = s.constraints

  // ---- Page 1: corridor behaviour ----
  // Panels capped at 86% of the (wide, Mamba-3-style) text block so the three
  // corridor views still share one page.
  pagebreak(weak: true)  // each card starts on a fresh page (no blank if already at top)
  heading(level: 2, title)
  align(center, image(apx + slug + "/corridor_pdyn.svg", width: 86%))
  align(center, image(apx + slug + "/corridor_inclination.svg", width: 86%))
  align(center, image(apx + slug + "/corridor_bank.svg", width: 86%))
  pagebreak()

  // ---- Page 2: cost + constraints + stats ----
  heading(level: 2, title + " (continued)")
  image(apx + slug + "/dv_cdf.svg", width: 100%)
  v(4pt)
  grid(columns: 3, gutter: 5pt,
    image(apx + slug + "/heat_flux.svg", width: 100%),
    image(apx + slug + "/g_load.svg", width: 100%),
    image(apx + slug + "/heat_load.svg", width: 100%))
  v(8pt)

  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    inset: (x: 6pt, y: 3.5pt),
    table.hline(stroke: 0.7pt),
    table.header([*Statistic*], [*p50*], [*p95*], [*mean/max*], [*note*]),
    table.hline(stroke: 0.35pt),
    ..srow([Correction Δv (m/s)], fnum(cap.dv.p50), fnum(cap.dv.p95), fnum(cap.dv.mean), [mean]),
    ..srow([Δv CVaR95 / p99 / max], fnum(s.dv_cvar95), fnum(s.dv_p99), fnum(cap.dv.max), [tail]),
    ..srow([dv1 periapsis raise], fnum(cap.dv1.p50), fnum(cap.dv1.p95), fnum(cap.dv1.max), [m/s]),
    ..srow([dv2 circularization], fnum(cap.dv2.p50), fnum(cap.dv2.p95), fnum(cap.dv2.max), [m/s]),
    ..srow([dv3 plane change], fnum(cap.dv3.p50), fnum(cap.dv3.p95), fnum(cap.dv3.max), [m/s]),
    ..srow([Apoapsis error (km)], fnum(cap.apoapsis.p50), fnum(cap.apoapsis.p95), fnum(cap.apoapsis.mean), [mean]),
    ..srow([Periapsis error (km)], fnum(cap.periapsis.p50), fnum(cap.periapsis.p95), fnum(cap.periapsis.mean), [mean]),
    ..srow([Inclination error (deg)], fnum(cap.inclination.p50, d: 2), fnum(cap.inclination.p95, d: 2), fnum(cap.inclination.mean, d: 2), [mean]),
    ..srow([Heat flux (kW/m²)], fnum(con.heat_flux.p50), fnum(con.heat_flux.p95), fnum(con.heat_flux.max), [viol #fnum(con.heat_flux.viol_pct)%]),
    ..srow([G-load (g)], fnum(con.g_load.p50, d: 2), fnum(con.g_load.p95, d: 2), fnum(con.g_load.max, d: 2), [viol #fnum(con.g_load.viol_pct)%]),
    ..srow([Heat load (MJ/m²)], fnum(con.heat_load.p50 / 1000, d: 1), fnum(con.heat_load.p95 / 1000, d: 1), fnum(con.heat_load.max / 1000, d: 1), [viol #fnum(con.heat_load.viol_pct)%]),
    table.hline(stroke: 0.35pt),
    ..srow([Capture rate], [], [], [], [#fnum(s.capture_rate * 100)%]),
    table.hline(stroke: 0.7pt),
  )
}

#scheme_report("nn_mamba", "NN -- Mamba (962 params)")
#scheme_report("nn_lstm", "NN -- LSTM (1082 params)")
#scheme_report("nn_gru", "NN -- GRU (1014 params)")
#scheme_report("nn_dense", "NN -- Dense (515 params)")
#scheme_report("ftc", "FTC (joint reference)")
#scheme_report("fnpag", "FNPAG")
#scheme_report("predguid", "PredGuid (joint reference)")
#scheme_report("energyctl", "Energy controller (joint reference)")
#scheme_report("eqglide", "Equilibrium glide")
#scheme_report("piecewise", "Piecewise constant")
