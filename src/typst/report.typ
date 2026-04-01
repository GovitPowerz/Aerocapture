#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

// Cover Page
#cover-page(meta)

// Part 1: Training Convergence
#section-heading("Part 1: Training Convergence")

#full-width-chart(dir + "/convergence.svg")

#full-width-chart(dir + "/diversity_cost.svg")

#if meta.at("has_cost_distribution", default: false) {
  full-width-chart(dir + "/cost_distribution.svg")
}

#full-width-chart(dir + "/parameter_evolution.svg")

#if meta.at("has_seed_pool", default: false) {
  full-width-chart(dir + "/seed_pool.svg")
}

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
    #text(fill: luma(120), size: 12pt)[Trajectory data not available — time-domain panels omitted.]
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

  // Performance Summary Table
  v(0.5cm)
  text(size: 12pt, weight: "bold")[Performance Summary]
  v(0.3cm)
  let summary = json(dir + "/summary_table.json")
  performance-table(summary.rows, violation-rows: summary.at("violation_rows", default: ()))

  // Dispersion Grid
  pagebreak()
  section-heading("Dispersion Correlations")
  image(dir + "/dispersion_grid.svg", width: 100%)
} else {
  align(center)[
    #v(2cm)
    #text(fill: luma(120), size: 12pt)[Final evaluation not available — distribution panels omitted.]
    #v(2cm)
  ]
}
