#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/summary.json")

#set page(..page-style)
#set text(size: 9pt)

#align(center)[
  #text(size: 20pt, weight: "bold")[NN Input Behavior Report]
  #v(0.15cm)
  #text(size: 13pt)[#meta.scheme]
  #v(0.1cm)
  #text(size: 9pt, fill: luma(110))[
    DV threshold #meta.dv_threshold m/s · blue (low DV) #meta.n_blue / red (high DV) #meta.n_red · #meta.n_sims sims
  ]
]
#v(0.3cm)

== Per-input summary (sorted by saturation)

#text(size: 8pt, fill: luma(110))[
  %|v|>1 = fraction of samples outside the network's expected [-1, 1] range.
  sep = |mean(red) - mean(blue)| / pooled-sigma (how strongly the input separates costly from cheap runs).
]
#v(0.15cm)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: none,
  inset: (x: 5pt, y: 2.5pt),
  align: (left, center, right, right, right, right, right),
  table.hline(stroke: 1.0pt),
  [*input*], [*in mask*], [*%|v|>1*], [*sep*], [*p1*], [*p50*], [*p99*],
  table.hline(stroke: 0.4pt),
  ..meta.inputs.map(r => (
    [#r.name],
    if r.in_mask [yes] else [#text(fill: luma(150))[--]],
    [#calc.round(r.frac_out_of_range * 100, digits: 1)%],
    [#calc.round(r.separation, digits: 2)],
    [#calc.round(r.p1, digits: 2)],
    [#calc.round(r.p50, digits: 2)],
    [#calc.round(r.p99, digits: 2)],
  )).flatten(),
  table.hline(stroke: 1.0pt),
)

#pagebreak()

== Input panels (time | energy estimated), most-saturated first

#for r in meta.inputs [
  #text(weight: "bold")[\[#r.index\] #r.name]
  #h(1fr)
  #text(size: 8pt, fill: luma(110))[
    sat #calc.round(r.frac_out_of_range * 100, digits: 0)% · sep #calc.round(r.separation, digits: 2)#if not r.in_mask [ · (unused)]
  ]
  #grid(columns: (1fr, 1fr), gutter: 5pt,
    image(dir + "/" + r.time_svg, width: 100%),
    image(dir + "/" + r.energy_svg, width: 100%),
  )
  #v(0.15cm)
]
