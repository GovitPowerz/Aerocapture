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
      stroke: 0.5pt + luma(200),
      inset: 8pt,
      align: (left, right),
      [*Best Cost*], [#meta.best_cost],
      [*Capture Rate*], [#meta.capture_rate],
      [*Generations*], [#meta.total_generations],
      [*Final Eval Sims*], [#meta.n_sims],
      [*Config Hash*], [#text(size: 8pt, font: "Courier New")[#meta.config_hash]],
    )
  ]
  pagebreak()
}

#let performance-table(data) = {
  let headers = ("Parameter", "Mean", "Std", "Min", "p5", "p25", "p50", "p75", "p95", "Max")
  table(
    columns: headers.len(),
    stroke: 0.5pt + luma(200),
    inset: 6pt,
    align: (left, ..range(headers.len() - 1).map(_ => right)),
    ..headers.map(h => text(weight: "bold", size: 8pt)[#h]),
    ..data.flatten().map(cell => text(size: 8pt)[#cell]),
  )
}
