# Changes since arxiv-v3

The committed `paper.pdf` is recompiled from the Typst source below (`make -C articles/paper pdf`,
last on 2026-09-23 with #134); the arxiv-v3 build is the `arxiv-v3` tag.

- 2026-09-16 (#108, ADR-0006): the abstract and the conclusion lead with the per-scenario-noise
  result (fine-tuned Mamba, three-seed means: CVaR99.9 163.2 +- 1.3 m/s, 99.996% capture of 10^6,
  73 m/s below FNPAG and the best dense network) and state the shared-path result (123.3 +- 0.1 at
  100% capture) as the historical headline that Appendix E corrects. Sections 5-7 and their tables
  still quote the shared-path regime and say so; per-scenario re-quotes of those sections are not
  part of this change.
- 2026-09-22 (#101): Section 5's reinforcement-learning sentence is re-quoted from
  protocol-matched cells. The old sentence ("636 m/s mean for the dense PPO policy and 513 for
  the recurrent one", footnoted as predating simulator fixes) mislabelled its artifacts: both
  bundled legacy RL cells (`legacy/neural_network_rl`, `legacy/neural_network_gru_ppo`) are one
  Dense -> GRU(16) -> Dense architecture on a 23-input mask at 15M / 60M env steps, matched to no
  population cell. The new baseline (`experiments/paper/18_rl_baseline.sh`, configs under
  `configs/training/paper/rl/`) trains PPO from scratch and PPO warm-started from the per-scenario
  champions `ou_marginal/ft_dense_p515` and `ou_marginal/ft_gru_p1014` under their exact
  observation contract, scaffolding, per_draw regime and reserved pools; the quote reads the new
  `rl/*` and `ou_marginal/*` keys of `data/results.json` (2M pool, n = 1000, four `ppo_*` paired
  tables). The footnote is gone. The README results card gains the two dense PPO rows on the
  10^6 confirmatory pools (`experiments/ou_marginal/confirmatory_marginal.json`) and a
  contributions block; the review workflow artifacts moved from `articles/paper/` to
  `docs/paper/reviews/`.
- 2026-09-22 (#120): the headline tables read the bundle at compile time. `results.typ`
  wraps `data/results.json`, `data/confirmatory_eval.json` and `data/quant/finalists_results.json`;
  every cell of the performance table, the paired-comparison table and the quantization
  finalists table is an accessor call (the performance table's Viol. column stays transcribed:
  the bundle carries no violation field). A colophon reads `data/provenance.json` (paper-inputs
  digest, Release tag, toolchain versions, noise regime) and the git head `make pdf` passes in.
  One cell moved: the QAT fine-tune CVaR99.9 is 122.85 pooled, which rounds to 122.9 (the
  transcription had rounded it down); the two prose quotes follow. Every other converted cell
  reproduces its literal.
- 2026-09-23 (#134): every prose quote of the 99.996% capture rate (abstract, Section 7,
  conclusion) is labelled as a three-fine-tune-seed mean, the label the CVaR99.9 beside it
  already carried; Section 7 and the conclusion also give the deployed seed's 99.9995%, the
  value in the performance table. Per-seed captures on the 10^6 confirmatory pool:
  999995 / 999970 / 999920 (s1 deployed / s2 / s3). No number changes.
