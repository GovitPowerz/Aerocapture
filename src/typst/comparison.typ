#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

#v(2cm)
#align(center)[
  #text(size: 24pt, weight: "bold")[Cross-Scheme Comparison]
  #v(0.3cm)
  #text(size: 12pt, fill: luma(100))[#meta.date]
]
#v(1cm)

#full-width-chart(dir + "/comparison_convergence.svg")

#v(0.5cm)
#text(size: 12pt, weight: "bold")[Final Metrics]
#v(0.3cm)
#let metrics = json(dir + "/comparison_table.json")
#table(
  columns: metrics.headers.len(),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  ..metrics.headers.map(h => text(weight: "bold", size: 9pt)[#h]),
  ..metrics.rows.flatten().map(cell => text(size: 9pt)[#cell]),
)
