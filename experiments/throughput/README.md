# Throughput and scaling study

What the batched simulator and the training loop cost on one machine (#114). Results live in
`throughput.json` plus two figures, `fig_scaling.svg` and `fig_generation_profile.svg`. The
write-up with the accelerator assessment is [`docs/performance.md`](../../docs/performance.md).

Regenerate everything with one command. It takes about 20 minutes on an M4 Pro; build the release
CLI and the extension from the same commit first, and leave the machine otherwise idle:

```bash
./build.sh
```

```bash
uv run python experiments/throughput/throughput.py
```

`--stages scaling seams grid memory profile fnpag_step benches` runs a subset (each stage rewrites only
its own block of `throughput.json`) and `--plot-only` redraws the figures from the JSON. Every
block records the machine, the commit, the date and `sources_clean`, which is false when `src/`,
`configs/`, `data/`, the paper bundle, the Python lockfile or the driver differed from that
commit (markdown excluded). Numbers from different machines are not comparable, so none are
mixed here.

| stage | measures | seam |
|---|---|---|
| `scaling` | sims/s at 1, 2, 4, 8, P-core count and all threads; n = 1000 seeds of the final-eval pool, 3 repeats interleaved across thread counts | `run_grid`, one individual |
| `seams` | per-sim cost at 1 thread: `run_grid`, `run_batch` (one TOML patch + parse + SimData build per sim), the Rust CLI (process start, config and table load, CSV output) | all three |
| `grid` | `run_grid` wall at the paper population (512 individuals, in-memory weights) vs seeds per individual; the linear fit's intercept is the SimData build, its slope the cells | `AerocaptureProblem` |
| `memory` | peak RSS a `run_grid` call adds, a fresh process per grid shape | `run_grid` |
| `profile` | wall time per generation of the real GA loop, split by loop phase and Rust vs Python, plus the cProfile top-5 hotspots | the training loop |
| `fnpag_step` | FNPAG per-sim cost at its GA-tuned predictor step and at the 2.0 s default, 1 thread, n = 50 | `run_grid`, one individual |
| `benches` | runs `cargo bench --bench tick` (its built-in rows plus the four deployed cells, whose TOMLs the driver writes with the Python deploy routing and passes through `AEROCAPTURE_TICK_CONFIGS`) and the Mamba f64 forward pass of `quant_forward`, and records criterion's estimates | Rust, one thread |

The cells are the paper's deployed operating points from the committed bundle
(`articles/paper/data/runs/`: FTC and FNPAG from `classical_baselines/`, dense-515 and
Mamba-962 from `headline/`), so `scaling`, `seams`, `memory`, `fnpag_step` and `benches` run on
a fresh clone. `grid` and `profile` resume the headline run's last checkpoint from the local
`training_output/mamba_p962_long/` and are skipped (their committed blocks kept) without it.
They need it because the population sets the trajectory lengths and so the cost: a converged
population flies the full ~700 s flights a long campaign pays for. The profile
redirects every write, including the NN deploy copy `save_checkpoint` makes at `[data]
neural_network`, into a scratch directory. The checkpoint carries a 2-seed curated list; an
allocation with another `n_sims` clears it so the first generation bootstraps the right width.
The noise regime is the default `per_draw` (ADR-0006).

Timings are not a CI gate because they are noisy. The deterministic companion is
`tests/test_thread_invariance.py`: `run_grid` output is byte-identical at any thread count. The
per-scheme guidance bench also runs on its own (built-in rows only):

```bash
cargo bench --bench tick --manifest-path src/rust/Cargo.toml
```
