#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

#align(center)[
  #text(size: 22pt, weight: "bold")[Warm-Start Snapshot]
  #v(0.2cm)
  #text(size: 14pt)[#meta.scheme]
  #v(0.1cm)
  #text(size: 9pt, fill: luma(110))[#meta.arch_summary]
]

#v(0.4cm)
#line(length: 100%, stroke: 0.5pt + luma(180))
#v(0.3cm)

== Configuration

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 6pt, y: 3pt),
  align: (left, left),
  [*Mode*],                    [#meta.config.mode],
  [*Output parameterization*], [#meta.config.output_parameterization],
  [*Supervisor schemes*],      [#meta.config.supervisor_schemes.join(", ")],
  [*n_warm_seeds*],            [#meta.config.n_warm_seeds],
  [*n_epochs*],                [#meta.config.n_epochs],
  [*bptt_length*],             [#meta.config.bptt_length],
  [*bound_multiplier*],        [#meta.config.bound_multiplier],
  [*adaptive_bounds*],         [#meta.config.adaptive_bounds],
  [*base_mc_seed*],            [#meta.config.base_mc_seed],
)

== Supervised pretrain (Adam MSE)

*#meta.loss_summary*  -- corpus: #meta.n_chunks BPTT chunks.

#image(dir + "/mse_convergence.svg", width: 100%)

== Supervisor selection

Per-seed best-of: the supervisor with the lowest captured DV wins; seeds with
no captures across any scheme are dropped. Total winners:
#meta.n_selected_total / #meta.min_corpus_required corpus-required.

#image(dir + "/supervisor_selection.svg", width: 100%)

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: none,
  inset: (x: 6pt, y: 4pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 1.2pt),
  [*Scheme*], [*Supervised*], [*Captured*], [*Capture rate*], [*Selected*], [*Median DV*],
  table.hline(stroke: 0.4pt),
  ..meta.supervisors.map(r => (
    [#r.scheme], [#r.n_supervised], [#r.n_captured], [#r.capture_rate], [#r.n_selected], [#r.median_dv],
  )).flatten(),
  table.hline(stroke: 1.2pt),
)

== PSO/GA/DE search-space bounds (per layer slab)

Adaptive bounds are 2× max-abs(slab) with a floor at Xavier × bound_multiplier.
Wider bars = larger slab; PSO explores [-half_width, +half_width] per param.

#image(dir + "/bound_widening.svg", width: 100%)

== Gen-0 validation baseline (warm-started chromosome)

Evaluated on the reserved validation seed pool (val_seeds), directly comparable
to the validation gate later in training.

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 6pt, y: 3pt),
  align: (left, right),
  [*n_sims*],       [#meta.baseline.n_sims],
  [*Capture rate*], [#meta.baseline.capture_rate],
  [*RMS cost*],     [#meta.baseline.rms_cost],
  [*Mean cost*],    [#meta.baseline.mean_cost],
  [*Median cost*],  [#meta.baseline.median_cost],
  [*p95 cost*],     [#meta.baseline.p95_cost],
  [*Worst cost*],   [#meta.baseline.worst_cost],
)

#if meta.eval_summary_lines.len() > 0 [
  == Final evaluation (warm-started chromosome on val seeds)

  Mirrors the end-of-training final-eval block so the warm-start metrics are
  directly comparable. Numbers come from the SAME MC run as the baseline above
  -- this view exposes per-axis DV / apoapsis / heat-flux statistics.

  #block(
    fill: luma(245),
    inset: 8pt,
    radius: 4pt,
    width: 100%,
    text(font: "Courier New", size: 9pt)[
      #for line in meta.eval_summary_lines [
        #line \
      ]
    ],
  )
]

#if meta.compare.has_data [
  #pagebreak()
  == Trajectory comparison: supervisor vs warm-started NN

  Side-by-side view of supervisor (#meta.compare.primary_supervisor)
  trajectories vs the warm-started NN, on BOTH the training pool
  (`WARM_START_SEED_OFFSET`) and the reserved validation pool
  (`VALIDATION_SEED_OFFSET`). Same dispersion draws per seed within a pool, so
  the only thing changing between supervisor and NN rows is the guidance scheme.

  Spaghetti coloring: blue = captured + constraints OK, orange = captured +
  constraint violation, red = crash / hyperbolic / timeout. Envelopes use ALL
  trajectories; spaghetti alpha scales as 1/√n so dense pools stay readable.

  #table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    inset: (x: 6pt, y: 4pt),
    align: (left, left, right, right),
    table.hline(stroke: 1.2pt),
    [*Pool*], [*Side*], [*Captured*], [*Capture rate*],
    table.hline(stroke: 0.4pt),
    ..meta.compare.rows.map(r => (
      [#r.pool], [#r.side_label], [#r.n_captured / #r.n_sims], [#r.capture_rate_pct],
    )).flatten(),
    table.hline(stroke: 1.2pt),
  )

  #for row in meta.compare.rows [
    #if row.panels.len() > 0 [
      === #row.pool pool -- #row.side_label

      // 2x2 grid of corridor panels + altitude/heat-flux time below.
      #grid(
        columns: (1fr, 1fr),
        gutter: 4pt,
        image(dir + "/" + row.panels.at(0), width: 100%),  // corridor_pdyn
        image(dir + "/" + row.panels.at(1), width: 100%),  // corridor_inclination
      )
      #v(4pt)
      #image(dir + "/" + row.panels.at(2), width: 100%)    // corridor_bank (full width)
      #v(4pt)
      #grid(
        columns: (1fr, 1fr),
        gutter: 4pt,
        image(dir + "/" + row.panels.at(3), width: 100%),  // altitude_time
        image(dir + "/" + row.panels.at(4), width: 100%),  // heat_flux_time
      )
      #v(10pt)
    ] else if "error" in row [
      === #row.pool pool -- #row.side_label

      #text(fill: red)[Failed: #row.error]
      #v(10pt)
    ]
  ]
]
