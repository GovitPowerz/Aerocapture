// Shared styling for Aerocapture reports

#let page-style = (
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 1.5cm, right: 1.5cm),
)

#let section-heading(title) = {
  v(0.5cm)
  line(length: 100%, stroke: 0.5pt + luma(180))
  v(0.3cm)
  text(size: 16pt, weight: "bold")[#title]
  v(0.3cm)
}

#let full-width-chart(path) = {
  image(path, width: 100%)
  v(0.3cm)
}

#let half-width-pair(left-path, right-path) = {
  grid(
    columns: (1fr, 1fr),
    column-gutter: 0.5cm,
    image(left-path, width: 100%),
    image(right-path, width: 100%),
  )
  v(0.3cm)
}

#let cover-page(meta) = {
  v(3cm)
  align(center)[
    #text(size: 28pt, weight: "bold")[Aerocapture Training Report]
    #v(0.5cm)
    #text(size: 18pt)[#meta.scheme]
    #v(0.3cm)
    #text(size: 12pt, fill: luma(100))[#meta.mission — #meta.date]
    #v(1cm)
    #table(
      columns: (auto, auto),
      stroke: none,
      inset: (x: 12pt, y: 5pt),
      align: (left, right),
      table.hline(stroke: 1.2pt),
      [*Best Cost*], [#meta.best_cost],
      [*Capture Rate*], [#meta.capture_rate],
      table.hline(stroke: 0.4pt),
      [*Generations*], [#meta.total_generations],
      [*Final Eval Sims*], [#meta.n_sims],
      table.hline(stroke: 0.4pt),
      [*Config Hash*], [#text(size: 8pt, font: "Courier New")[#meta.config_hash]],
      table.hline(stroke: 1.2pt),
    )
  ]
  pagebreak()
}

#let performance-table(data, violation-rows: ()) = {
  let headers = ("Parameter", "Mean", "Std", "Min", "p5", "p25", "p50", "p75", "p95", "Max")
  let n-cols = headers.len()

  // Booktabs style: no vertical strokes, horizontal rules only, full page width
  table(
    columns: (1fr, ..range(n-cols - 1).map(_ => auto)),
    stroke: none,
    inset: (x: 6pt, y: 4pt),
    align: (left, ..range(n-cols - 1).map(_ => right)),

    // Header row
    table.hline(stroke: 1.2pt),
    ..headers.map(h => text(weight: "bold", size: 8pt)[#h]),
    table.hline(stroke: 0.6pt),

    // Data rows
    ..data.flatten().map(cell => text(size: 8pt)[#cell]),

    // Constraint violation section
    ..if violation-rows.len() > 0 {
      (
        table.hline(stroke: 0.6pt),
        ..violation-rows.flatten().map(cell => text(size: 8pt, fill: if cell != "" { luma(80) } else { white })[#cell]),
      )
    } else {
      ()
    },

    // Bottom rule
    table.hline(stroke: 1.2pt),
  )
}
