# Changes since arxiv-v3

The committed `paper.pdf` is the arxiv-v3 build. The Typst source carries the changes below
for the next version; the PDF is recompiled only when that version pass is complete.

- 2026-09-16 (#108, ADR-0006): the abstract and the conclusion lead with the per-scenario-noise
  result (fine-tuned Mamba, CVaR99.9 163.2 +- 1.3 m/s over three seeds, 99.996% capture of 10^6,
  73 m/s below FNPAG and the best dense network) and state the shared-path result (123.3 +- 0.1 at
  100% capture) as the historical headline that Appendix E corrects. Sections 5-7 and their tables
  still quote the shared-path regime and say so; per-scenario re-quotes of those sections are not
  part of this change.
