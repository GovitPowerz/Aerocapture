#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

// Cover Page
#v(3cm)
#align(center)[
  #text(size: 28pt, weight: "bold")[Aerocapture RL Training Report]
  #v(0.5cm)
  #text(size: 18pt)[neural_network (#meta.at("algorithm", default: "PPO"))]
  #v(0.3cm)
  #text(size: 12pt, fill: luma(100))[#meta.at("mission", default: "RL") -- #meta.at("date", default: "")]
  #v(1cm)
  #table(
    columns: (auto, auto),
    stroke: none,
    inset: (x: 12pt, y: 5pt),
    align: (left, right),
    table.hline(stroke: 1.2pt),
    [*Updates*], [#meta.at("n_updates", default: "N/A")],
    [*Final Eval Sims*], [#if meta.at("has_final_eval", default: false) { meta.at("final_eval_n_sims", default: "1000") } else { "skipped" }],
    table.hline(stroke: 1.2pt),
  )
]
#pagebreak()

// Part 1: RL Convergence
#section-heading("Part 1: RL Convergence")

#full-width-chart(dir + "/rl_return.svg")
#full-width-chart(dir + "/rl_dv.svg")
#full-width-chart(dir + "/rl_entropy.svg")
#full-width-chart(dir + "/rl_value_loss.svg")
#full-width-chart(dir + "/rl_capture.svg")
#full-width-chart(dir + "/rl_val.svg")

#pagebreak()

// Part 2: Mission Performance
#section-heading("Part 2: Mission Performance")

#if meta.at("has_trajectories", default: false) {
  full-width-chart(dir + "/corridor_pdyn.svg")
  full-width-chart(dir + "/corridor_inclination.svg")
  full-width-chart(dir + "/corridor_bank.svg")
  full-width-chart(dir + "/altitude_time.svg")
  full-width-chart(dir + "/heat_flux_time.svg")
  full-width-chart(dir + "/gload_time.svg")
  full-width-chart(dir + "/bank_angle_time.svg")
  full-width-chart(dir + "/nav_density_ratio.svg")
} else {
  align(center)[
    #v(2cm)
    #text(fill: luma(120), size: 12pt)[Trajectory data not available -- time-domain panels omitted.]
    #v(2cm)
  ]
}

#if meta.at("has_final_eval", default: false) {
  full-width-chart(dir + "/cost_objective.svg")
  full-width-chart(dir + "/dv_distribution.svg")
  full-width-chart(dir + "/dv_individual_burns.svg")

  if meta.at("has_trajectories", default: false) {
    full-width-chart(dir + "/entry_conditions.svg")
    full-width-chart(dir + "/exit_conditions.svg")
  }

  v(0.5cm)
  text(size: 12pt, weight: "bold")[Performance Summary]
  v(0.3cm)
  let summary = json(dir + "/summary_table.json")
  performance-table(summary.rows, violation-rows: summary.at("violation_rows", default: ()))

  pagebreak()
  section-heading("Dispersion Correlations")
  image(dir + "/dispersion_grid.svg", width: 100%)
} else {
  align(center)[
    #v(2cm)
    #text(fill: luma(120), size: 12pt)[Final evaluation not available -- distribution panels omitted.]
    #v(2cm)
  ]
}

// Part 3: Sensitivity Analysis (optional)
#if meta.at("has_sensitivity", default: false) {
  pagebreak()
  section-heading("Part 3: Sensitivity Analysis")

  if meta.at("has_morris", default: false) {
    full-width-chart(dir + "/morris_scatter.svg")

    v(0.5cm)
    text(size: 12pt, weight: "bold")[Morris Screening Results]
    v(0.3cm)
    let morris_data = json(dir + "/morris_table.json")
    morris-table(morris_data.rows)
  }
  if meta.at("has_sobol", default: false) {
    full-width-chart(dir + "/sobol_bars.svg")
  }
  if meta.at("has_sobol_heatmap", default: false) {
    full-width-chart(dir + "/sobol_heatmap.svg")
  }
}
