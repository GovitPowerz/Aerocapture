# Rebuild the paper's figures, results.json and PDF from the committed bundle.
#
#   make -C articles/paper                 figures (default): the 18 fig_*.svg from data/ + the bundle
#   make -C articles/paper paper           fetch-logs (if absent) -> results -> figures -> provenance -> pdf
#   make -C articles/paper check           SHA256SUMS + results.json schema + figures/results.json unchanged in git
#   make -C articles/paper -B figures      force every figure (CI does this, then `check`)
#   make -C articles/paper -n paper        print what would run
#
# Bundle-only by default: no Rust build, no simulator, no training_output/. The
# `mc-*` targets at the bottom re-fly cells and are opt-in (hours of compute; never
# CI). Every recipe runs from the repo root; the scripts resolve REPO from their own
# path and typst resolves figure paths relative to paper.typ. GNU make 3.81 (macOS).

HERE := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ifneq ($(CURDIR),$(HERE))
$(error run as: make -C articles/paper <target>)
endif
ROOT := $(abspath $(HERE)/../..)
UVPY := cd $(ROOT) && uv run python
SHASUM := shasum -a 256

SCRIPTS := scripts
DATA := data
RUNS := $(DATA)/runs
FIGS := figures
PDF ?= articles/paper/paper.pdf

.DEFAULT_GOAL := figures
.PHONY: figures results fetch-logs provenance pdf paper check check-sums check-logs check-schema sums help \
        need-sim mc-data mc-far-tail mc-confirmatory mc-corridor mc-appendix mc-robustness mc-robustness-retrain \
        mc-objective-centering mc-compute-benchmark mc-requote mc-quant probes

# ---------------------------------------------------------------------------
# Figures: figures/fig_<name>.svg <- scripts/fig_<name>.py + figlib + fonts + data
# ---------------------------------------------------------------------------
FIG_NAMES := ablation arch_tail classical_vs_nn corridor cost_transform curation islands joint_reference \
             loss_vs_tail objective_centering optimizer output_param pareto plateau robustness seed_strategy \
             survival training_n_sims
FIGURES := $(FIG_NAMES:%=$(FIGS)/fig_%.svg)
FONTS := fonts/STIXTwoText.ttf fonts/STIXTwoText-Italic.ttf
FIGLIB := $(SCRIPTS)/figlib.py $(FONTS)

figures: $(FIGURES)

$(FIGS)/fig_%.svg: $(SCRIPTS)/fig_%.py $(FIGLIB)
	$(UVPY) articles/paper/$<

# Per-figure data inputs (the recipe comes from the pattern rule above). A missing
# input is a make error, never a thinner figure.
SWEEP_MANIFESTS := $(ROOT)/configs/training/sweep/manifest.json $(ROOT)/configs/training/sweep/manifest_floor.json
SWEEP_PARQUETS := $(wildcard $(RUNS)/architecture_sweep/*/final_eval.parquet)

$(FIGS)/fig_ablation.svg: $(RUNS)/headline/mamba_p962/ablation_results.json
$(FIGS)/fig_arch_tail.svg: $(DATA)/confirmatory_eval.json
$(FIGS)/fig_classical_vs_nn.svg: $(DATA)/confirmatory_eval.json $(DATA)/compute_benchmark.json
$(FIGS)/fig_corridor.svg: $(DATA)/corridor.npz
$(FIGS)/fig_cost_transform.svg: $(DATA)/far_tail_eval.json
$(FIGS)/fig_curation.svg: $(DATA)/results.json $(DATA)/far_tail_eval.json
$(FIGS)/fig_islands.svg:
$(FIGS)/fig_joint_reference.svg: $(DATA)/results.json
$(FIGS)/fig_loss_vs_tail.svg: $(DATA)/results.json $(DATA)/far_tail_eval.json
$(FIGS)/fig_objective_centering.svg: $(DATA)/objective_centering.json $(DATA)/robustness_retrain.json
$(FIGS)/fig_optimizer.svg: $(DATA)/results.json
$(FIGS)/fig_output_param.svg: $(DATA)/results.json
$(FIGS)/fig_pareto.svg: $(SWEEP_MANIFESTS) $(SWEEP_PARQUETS)
$(FIGS)/fig_plateau.svg: $(DATA)/plateau.json
$(FIGS)/fig_robustness.svg: $(DATA)/robustness_stress.json
$(FIGS)/fig_seed_strategy.svg: $(DATA)/results.json $(DATA)/sigma_extras.json
$(FIGS)/fig_survival.svg: $(DATA)/confirmatory_eval.json
$(FIGS)/fig_training_n_sims.svg: $(DATA)/results.json

# results.json is a committed bundle product consumed by the figures as a plain file
# (deliberately no rule: a fresh clone's mtimes must not trigger a rebuild that needs
# the 195 MB run logs; a missing file is "No rule to make target"). `make results`
# regenerates it explicitly, and only with the logs present.

# ---------------------------------------------------------------------------
# results.json <- aggregate_results.py over the bundle parquets + the run logs
# ---------------------------------------------------------------------------
RUN_LOGS := $(wildcard $(RUNS)/*/*/run.jsonl.gz)

results:
	$(UVPY) articles/paper/scripts/aggregate_results.py --require-logs

# The 95 run.jsonl.gz (~195 MB) are a Release asset (tag arxiv-v2), gitignored. Network.
fetch-logs:
	@if [ -z "$(RUN_LOGS)" ]; then bash $(SCRIPTS)/fetch_run_logs.sh; else echo "run logs present: $(words $(RUN_LOGS)) files"; fi

# ---------------------------------------------------------------------------
# Provenance, PDF, the one top-level target
# ---------------------------------------------------------------------------
provenance:
	$(UVPY) articles/paper/scripts/write_provenance.py

# SOURCE_DATE_EPOCH = the HEAD commit time, so the PDF's creation date is reproducible.
# `make pdf PDF=/tmp/paper.pdf` compiles without touching the committed PDF (CI).
pdf:
	cd $(ROOT) && SOURCE_DATE_EPOCH=$$(git log -1 --format=%ct) typst compile articles/paper/paper.typ $(PDF)

# Recursive so the log wildcard is re-evaluated after fetch-logs.
paper: fetch-logs
	$(MAKE) results
	$(MAKE) figures
	$(MAKE) provenance
	$(MAKE) pdf

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
check: check-sums check-schema
	cd $(ROOT) && git diff --exit-code --stat -- articles/paper/figures articles/paper/data/results.json

# data/SHA256SUMS covers every tracked file under data/ except itself and provenance.json.
check-sums:
	cd $(DATA) && $(SHASUM) --check --quiet SHA256SUMS

# The run logs (after fetch-logs): the extracted Release asset, file by file.
check-logs:
	cd $(DATA) && $(SHASUM) --check --quiet SHA256SUMS.runlogs

check-schema:
	$(UVPY) articles/paper/scripts/check_results_schema.py

# Regenerate both checksum files after a deliberate bundle change (12_collect_results.sh, mc-* targets).
sums:
	cd $(DATA) && git ls-files -z . | tr '\0' '\n' | grep -v -e '^SHA256SUMS' -e '^provenance.json$$' | LC_ALL=C sort | xargs $(SHASUM) > SHA256SUMS
	cd $(DATA) && find runs -name run.jsonl.gz | LC_ALL=C sort | xargs $(SHASUM) > SHA256SUMS.runlogs

help:
	@sed -n '2,8p' $(lastword $(MAKEFILE_LIST))

# ---------------------------------------------------------------------------
# FROZEN inputs: committed data files with NO producer in the tree. They were
# hand-extracted once (the producing commit and the method are recorded here);
# a missing one is an error, never a silent re-derivation (the recipe only checks
# existence so `make -B` still works).
#   plateau.json                     b85db68  running-min validation RMS per generation for headline/dense_p515 and
#                                             headline/dense_p972, read off the 4-segment training JSONLs (fig_plateau)
#   sigma_extras.json                7234f40  n=1000 final-eval stats of the 16_sigma_extras.sh seed repeats
#                                             (ga/cmaes x fixed/rotating, s1-s3); the eval step lived in a plan doc
#   selection_gate.json              20482aa  validation-gate query count for the pool-roles table
#   nominal_floor.json               fd396b4  undispersed-nominal correction-DV floor (section on the objective)
#   fnpag_failure_classification.json f73a454 FNPAG confirmatory failed seeds: timeout vs physical
#   s2_failure_classification.json   2b683e0  Mamba s2 confirmatory failed seeds: timeout vs physical (61/1e6 physical)
#   quant/ticks_per_sim.json         d9a413e  guidance ticks per sim behind the Appendix C compute table
# ---------------------------------------------------------------------------
FROZEN := $(DATA)/plateau.json $(DATA)/sigma_extras.json $(DATA)/selection_gate.json $(DATA)/nominal_floor.json \
          $(DATA)/fnpag_failure_classification.json $(DATA)/s2_failure_classification.json $(DATA)/quant/ticks_per_sim.json
$(FROZEN):
	@test -f $@ || { echo "error: $@ is a FROZEN input with no producer in the tree (see the FROZEN block in articles/paper/Makefile)"; exit 1; }

# ---------------------------------------------------------------------------
# mc-data: OPT-IN producers that re-fly deployed cells. Never default, never CI.
# Need training_output/ (the cells are not committed), the PyO3 build (./build.sh),
# and every committed number was produced under the legacy noise regime (the
# scripts pin it; ADR-0003/0006). Wall times are single-machine estimates.
# ---------------------------------------------------------------------------
need-sim:
	@test -d $(ROOT)/training_output || { echo "error: training_output/ missing -- mc-* targets re-fly cells (experiments/paper/README.md)"; exit 1; }
	@$(UVPY) -c "import aerocapture_rs" 2>/dev/null || { echo "error: aerocapture_rs not built (./build.sh)"; exit 1; }

mc-data: mc-far-tail mc-confirmatory mc-corridor mc-appendix mc-robustness mc-robustness-retrain \
         mc-objective-centering mc-compute-benchmark mc-requote mc-quant probes

# far_tail_eval.json -- n=10000 reserved 2M pool per cell (~minutes/cell, FNPAG ~50x slower).
# The cell list is label:toml[:bundle_key]; the committed file records only labels, so pass CELLS=...
# (the 26 committed labels: `python -c "import json;print([c['label'] for c in json.load(open('data/far_tail_eval.json'))['cells']])"`).
mc-far-tail: need-sim
	@test -n "$(CELLS)" || { echo "error: CELLS='label:toml ...' required (see the comment above mc-far-tail)"; exit 1; }
	$(UVPY) articles/paper/scripts/far_tail_eval.py --n-sims 10000 --cells $(CELLS)

# confirmatory_eval.json -- 10 x 100000 scenarios per cell, [2^31, 2^32) pool: HOURS per cell.
# The committed file is self-describing (label, toml, bundle_key per cell), so the cell list comes from it.
mc-confirmatory: need-sim
	$(UVPY) articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
	    $$($(UVPY) -c "import json; print(' '.join(':'.join(filter(None, (c['label'], c['toml'], c.get('bundle_key')))) for c in json.load(open('articles/paper/data/confirmatory_eval.json'))['cells']))")

# corridor.npz -- 300000 randomized piecewise-constant sims (~1 h) + boundary traces.
mc-corridor: need-sim
	$(UVPY) articles/paper/scripts/collect_corridor.py --n-sims 300000

# figures/appendix/<scheme>/{7 SVGs, stats.json} -- 1000 sims x 10 schemes with trajectories (~10 min).
mc-appendix: need-sim
	$(UVPY) articles/paper/scripts/collect_appendix.py --n-sims 1000

# robustness_stress.json / robustness_retrain.json / objective_centering.json -- 1000 sims per cell on the 9M stress pool.
mc-robustness: need-sim
	$(UVPY) articles/paper/scripts/robustness_stress.py --n-sims 1000
mc-robustness-retrain: need-sim
	$(UVPY) articles/paper/scripts/robustness_retrain_eval.py --n-sims 1000
mc-objective-centering: need-sim
	$(UVPY) articles/paper/scripts/objective_centering_eval.py --n-sims 1000

# compute_benchmark.json -- 200 single-threaded sims per scheme; wall-clock, never diffable.
mc-compute-benchmark: need-sim
	$(UVPY) articles/paper/scripts/compute_benchmark.py --n-sims 200

# runs/headline/mamba_p962/fresh_pool_requote.json -- the headline cell on the untouched 8M pool (1000 sims).
mc-requote: need-sim
	$(UVPY) articles/paper/scripts/fresh_pool_requote.py training_output/mamba_p962_long --toml configs/training/sweep/mamba_p962.toml --n-sims 1000

# quant/{quantization_results,finalists_results,bench_forward}.json + figures/quantization_sweep.svg:
# copied from the quantization campaign's training_output by its runner.
mc-quant: need-sim
	cd $(ROOT) && ./experiments/paper/17_quantization.sh collect

# probes/<x>_probe_results.json <- the probe drivers' training_output/<x>_probe/probe_results.json.
probes: $(DATA)/probes/cfc_probe_results.json $(DATA)/probes/mamba3_probe_results.json $(DATA)/probes/xlstm_probe_results.json
$(DATA)/probes/%_probe_results.json: $(ROOT)/training_output/%_probe/probe_results.json
	cp $< $@
