# Reachable-corridor visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `fig_corridor` with a first-principles reachable capture corridor traced by a large dispersed randomized piecewise-constant Monte-Carlo.

**Architecture:** A rewritten collector (`collect_corridor.py`) runs a parameterized, batched, memory-bounded MC of randomized piecewise-constant bank profiles under full dispersions, folding each capturing trajectory's (energy, pdyn) points into 2-D count histograms, then reads per-energy-bin p99.5 (upper) / p0.5 (lower) percentiles and smooths them. A rewritten `fig_corridor.py` draws the shaded band + deployed-policy ensemble + nominal. Collector-vs-figure split as elsewhere.

**Tech Stack:** Python (numpy, scipy.ndimage), `aerocapture_rs` PyO3 bindings (`run_batch`/`run_mc`), matplotlib + figlib, Typst.

## Global Constraints

- Collector reads `training_output/` + `configs/`; figure/Typst read only committed `articles/paper/` files (established pattern).
- Bank sampling: N piecewise segments each uniform in [0°, 180°] (positive/in-plane, no roll reversals). N and n_sims are CLI parameters (defaults 10, 300000).
- Environment dispersions ON (the piecewise base config's regime); per-sim `monte_carlo.seed` from `CORRIDOR_SEED_OFFSET = 10_000_000` (disjoint from reserved pools ≤ 9M). Bank draws from fixed `CORRIDOR_BANK_SEED = 20260706` (reproducible).
- Upper boundary = p99.5 of pdyn per energy bin among captures (`ifinal==3 & ecc<1`); lower = p0.5 among captures with `apoapsis_alt_km < 5000`; then `gaussian_filter1d(sigma=2.5)`.
- Column indices — final record (52,): `FR_ECC=9`, `FR_APO=15`, `FR_IFINAL=31`. Trajectory (N,17): `TC_ENERGY=8`, `TC_PDYN=9`.
- Base config: `configs/training/msr_aller_piecewise_constant_train.toml` (inherits `common.toml` + `missions/mars.toml`, `guidance.type = "piecewise_constant"`).
- Deployed-ensemble overlay pinned to the committed bundle model `articles/paper/data/runs/headline/mamba_p962/best_model.json` (frozen deploy; training_output can drift).
- The user runs the heavy MC or the assistant runs it (~3–5 min at 300k); commit only `corridor.npz` + SVGs, never `training_output/` or raw trajectories.

---

### Task 1: Rewrite the corridor collector

**Files:**
- Rewrite: `articles/paper/scripts/collect_corridor.py`

**Interfaces:**
- Consumes: `aerocapture_rs.run_batch/run_mc`; `aerocapture.training.{evaluate.make_reserved_seeds, evaluate.FINAL_EVAL_SEED_OFFSET, reference._MC_DISPERSION_DOMAINS, report._resolve_eval_toml, toml_utils.load_toml_with_bases}`; `scipy.ndimage.gaussian_filter1d`.
- Produces: `articles/paper/data/corridor.npz` with keys `energy_bins, lower_pdyn, upper_pdyn, nominal_energy, nominal_pdyn, ens_energy (object), ens_pdyn (object), n_sims, n_segments, apoapsis_max_km, upper_pct, lower_pct`. CLI: `--n-sims --n-segments --apoapsis-max-km --n-energy-bins --n-pdyn-buckets --chunk-size --upper-pct --lower-pct --smooth-sigma --ensemble-sims`.

- [ ] **Step 1: Replace `collect_corridor.py` entirely with the new collector**

```python
"""Collect the reachable-corridor data (articles/paper/data/corridor.npz).

Traces the aerocapture capture corridor in the (orbital energy, dynamic pressure)
plane from a large DISPERSED Monte-Carlo of RANDOMIZED piecewise-constant bank
profiles (positive / in-plane, no roll reversals). Per energy bin: the upper
boundary is the p99.5 of pdyn among CAPTURING trajectories (crash-side limit),
the lower boundary the p0.5 among captures below --apoapsis-max-km (escape-side
limit); both Gaussian-smoothed. Overlays the deployed Mamba ensemble + nominal.

Collector-vs-figure split: reads training_output/ + configs; the committed
corridor.npz is the durable artifact.

Usage:
    uv run python articles/paper/scripts/collect_corridor.py \
        [--n-sims 300000] [--n-segments 10] [--apoapsis-max-km 5000] \
        [--n-energy-bins 200] [--n-pdyn-buckets 400] [--chunk-size 20000] \
        [--upper-pct 99.5] [--lower-pct 0.5] [--smooth-sigma 2.5] [--ensemble-sims 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

BASE_TOML = REPO / "configs/training/msr_aller_piecewise_constant_train.toml"
MAMBA_RUN = REPO / "training_output/mamba_p962_long"
MAMBA_TOML = REPO / "configs/training/sweep/mamba_p962.toml"
MAMBA_BUNDLE = REPO / "articles/paper/data/runs/headline/mamba_p962"
OUT = REPO / "articles/paper/data/corridor.npz"

CORRIDOR_SEED_OFFSET = 10_000_000   # env-dispersion seeds, disjoint from reserved pools (<= 9M)
CORRIDOR_BANK_SEED = 20260706       # fixed -> reproducible bank draws

FR_ECC, FR_APO, FR_IFINAL = 9, 15, 31   # final record (52,)
TC_ENERGY, TC_PDYN = 8, 9               # trajectory (N, 17)

E_LO, E_HI = -6.0, 5.0
PDYN_MAX = 2.8   # kPa top histogram edge (captures do not dive past this; deep divers crash)
DOWNSAMPLE = 3   # ensemble trajectory point stride


def _percentile_from_hist(hist, p_centers, pct):
    """Per energy bin (row): pdyn at cumulative fraction pct/100; NaN if the bin is empty."""
    out = np.full(hist.shape[0], np.nan)
    frac = pct / 100.0
    for e in range(hist.shape[0]):
        tot = hist[e].sum()
        if tot == 0:
            continue
        cum = np.cumsum(hist[e]) / tot
        out[e] = p_centers[min(int(np.searchsorted(cum, frac)), len(p_centers) - 1)]
    return out


def _smooth(e_centers, y, sigma):
    from scipy.ndimage import gaussian_filter1d

    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return y
    y = y.copy()
    y[~valid] = np.interp(e_centers[~valid], e_centers[valid], y[valid])
    return gaussian_filter1d(y, sigma=sigma)


def build_corridor(args):
    import aerocapture_rs

    e_edges = np.linspace(E_LO, E_HI, args.n_energy_bins + 1)
    p_edges = np.linspace(0.0, PDYN_MAX, args.n_pdyn_buckets + 1)
    e_centers = (e_edges[:-1] + e_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2
    hist_up = np.zeros((args.n_energy_bins, args.n_pdyn_buckets), dtype=np.int64)
    hist_lo = np.zeros((args.n_energy_bins, args.n_pdyn_buckets), dtype=np.int64)
    rng = np.random.default_rng(CORRIDOR_BANK_SEED)

    done = n_cap = 0
    while done < args.n_sims:
        m = min(args.chunk_size, args.n_sims - done)
        banks = rng.uniform(0.0, 180.0, size=(m, args.n_segments))
        ov = []
        for j in range(m):
            d = {"simulation.n_sims": 1,
                 "monte_carlo.seed": CORRIDOR_SEED_OFFSET + done + j,
                 "guidance.piecewise_constant.n_segments": args.n_segments}
            for i in range(args.n_segments):
                d[f"guidance.piecewise_constant.bank_angle_{i}"] = float(banks[j, i])
            ov.append(d)
        batch = aerocapture_rs.run_batch(toml_path=str(BASE_TOML.resolve()), overrides_list=ov,
                                         include_trajectories=True, sim_timeout_secs=5.0)
        recs = np.asarray(batch.final_records)
        cap = (recs[:, FR_IFINAL] == 3) & (recs[:, FR_ECC] < 1.0)
        apo = recs[:, FR_APO]
        n_cap += int(cap.sum())
        for j in np.nonzero(cap)[0]:
            t = np.asarray(batch.trajectories[j])
            ei = np.clip(np.digitize(t[:, TC_ENERGY], e_edges) - 1, 0, args.n_energy_bins - 1)
            pi = np.clip(np.digitize(t[:, TC_PDYN], p_edges) - 1, 0, args.n_pdyn_buckets - 1)
            np.add.at(hist_up, (ei, pi), 1)
            if apo[j] < args.apoapsis_max_km:
                np.add.at(hist_lo, (ei, pi), 1)
        done += m
        del batch
        print(f"  {done}/{args.n_sims} sims, capture {100 * n_cap / max(done, 1):.1f}%", end="\r")
    print()
    upper = _smooth(e_centers, _percentile_from_hist(hist_up, p_centers, args.upper_pct), args.smooth_sigma)
    lower = _smooth(e_centers, _percentile_from_hist(hist_lo, p_centers, args.lower_pct), args.smooth_sigma)
    pop = int((hist_up.sum(axis=1) > 0).sum())
    print(f"corridor: {n_cap} captures, {pop}/{args.n_energy_bins} energy bins populated")
    return e_centers, lower, upper, n_cap


def build_overlay(n_ens):
    """Deployed Mamba dispersed ensemble (energy, pdyn per trajectory) + undispersed nominal."""
    import aerocapture_rs
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    eval_toml, scaffolding = _resolve_eval_toml(MAMBA_TOML, MAMBA_RUN)
    pin = dict(scaffolding)
    bundle_model = MAMBA_BUNDLE / "best_model.json"
    if bundle_model.exists():
        pin["data.neural_network"] = str(bundle_model.resolve())
    base_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_seed, FINAL_EVAL_SEED_OFFSET, n_ens)
    ov = [{"simulation.n_sims": 1, "monte_carlo.seed": s, **pin} for s in seeds]
    batch = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=ov,
                                     include_trajectories=True, sim_timeout_secs=5.0)
    ens_e, ens_p = [], []
    for t in batch.trajectories:
        a = np.asarray(t)[::DOWNSAMPLE]
        ens_e.append(a[:, TC_ENERGY])
        ens_p.append(a[:, TC_PDYN])
    nom_ov = {"simulation.n_sims": 1,
              **{f"monte_carlo.{dom}.level": "off" for dom in _MC_DISPERSION_DOMAINS}, **pin}
    nom = aerocapture_rs.run_mc(toml_path=str(eval_toml.resolve()), overrides=nom_ov,
                                include_trajectories=True, sim_timeout_secs=5.0)
    nt = np.asarray(nom.trajectories[0])
    return (np.array(ens_e, dtype=object), np.array(ens_p, dtype=object),
            nt[:, TC_ENERGY], nt[:, TC_PDYN])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=300_000)
    p.add_argument("--n-segments", type=int, default=10)
    p.add_argument("--apoapsis-max-km", type=float, default=5000.0)
    p.add_argument("--n-energy-bins", type=int, default=200)
    p.add_argument("--n-pdyn-buckets", type=int, default=400)
    p.add_argument("--chunk-size", type=int, default=20_000)
    p.add_argument("--upper-pct", type=float, default=99.5)
    p.add_argument("--lower-pct", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=2.5)
    p.add_argument("--ensemble-sims", type=int, default=200)
    args = p.parse_args()

    e_centers, lower, upper, n_cap = build_corridor(args)
    ens_e, ens_p, nom_e, nom_p = build_overlay(args.ensemble_sims)
    np.savez_compressed(
        OUT, energy_bins=e_centers, lower_pdyn=lower, upper_pdyn=upper,
        nominal_energy=nom_e, nominal_pdyn=nom_p, ens_energy=ens_e, ens_pdyn=ens_p,
        n_sims=args.n_sims, n_segments=args.n_segments, apoapsis_max_km=args.apoapsis_max_km,
        upper_pct=args.upper_pct, lower_pct=args.lower_pct)
    print(f"wrote {OUT.relative_to(REPO)}: {n_cap} captures, ensemble {len(ens_e)}, nominal {len(nom_e)} pts")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Prove the per-sim bank overrides take effect (full lift-up vs full lift-down)**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
import sys; sys.path.insert(0,'src/python'); import numpy as np, aerocapture_rs
toml='configs/training/msr_aller_piecewise_constant_train.toml'
def peak_pdyn(bank):
    ov=[{'simulation.n_sims':1,'monte_carlo.seed':500,'guidance.piecewise_constant.n_segments':10,
         **{f'guidance.piecewise_constant.bank_angle_{i}':bank for i in range(10)}}]
    b=aerocapture_rs.run_batch(toml_path=toml,overrides_list=ov,include_trajectories=True,sim_timeout_secs=5.0)
    return float(np.asarray(b.trajectories[0])[:,9].max())
up, down = peak_pdyn(0.0), peak_pdyn(180.0)
print('peak pdyn: full lift-up(0deg)=%.3f  full lift-down(180deg)=%.3f'%(up,down))
assert down > up*1.3, 'bank override had no effect -- WRONG override path'
print('OK: bank overrides take effect')
"
```
Expected: two clearly different peak-pdyn values (full lift-down dives much deeper), and `OK: bank overrides take effect`. If it asserts, the override path is wrong — stop and fix before the sweep.

- [ ] **Step 3: Smoke-run the collector at small scale and verify a populated corridor with width**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python articles/paper/scripts/collect_corridor.py --n-sims 8000 --chunk-size 4000 --ensemble-sims 40`
Expected: prints a capture rate strictly between 0% and 100% (random banks ⇒ some capture, some don't), `corridor: <N> captures, <pop>/200 energy bins populated` with `pop` ≥ ~120, then `wrote articles/paper/data/corridor.npz`.

- [ ] **Step 4: Verify the boundaries are ordered and finite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
import numpy as np
d=np.load('articles/paper/data/corridor.npz',allow_pickle=True)
e,lo,hi=d['energy_bins'],d['lower_pdyn'],d['upper_pdyn']
m=~np.isnan(lo)&~np.isnan(hi)
print('bins',len(e),'| upper>=lower on',int((hi[m]>=lo[m]).sum()),'/',int(m.sum()))
print('pdyn range: lower[%.3f..%.3f] upper[%.3f..%.3f]'%(np.nanmin(lo),np.nanmax(lo),np.nanmin(hi),np.nanmax(hi)))
print('ensemble',len(d['ens_energy']),'nominal',len(d['nominal_energy']))
assert (hi[m]>=lo[m]).all(), 'upper below lower somewhere'
print('OK')
"
```
Expected: `upper>=lower` on all populated bins, plausible pdyn ranges (upper max ≈ 1.5–2.5 kPa), non-empty ensemble + nominal, `OK`.

- [ ] **Step 5: Commit the collector (small-scale corridor.npz regenerated in Task 3)**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/scripts/collect_corridor.py
git commit -m "paper(corridor): reachable-corridor collector (dispersed piecewise MC)"
```

---

### Task 2: Rewrite the corridor figure

**Files:**
- Rewrite: `articles/paper/scripts/fig_corridor.py`

**Interfaces:**
- Consumes: `articles/paper/data/corridor.npz` (Task 1 schema); `figlib` (`style`, `save`, `C`, `SIZE1`, `DATA`).
- Produces: `articles/paper/figures/fig_corridor.svg`.

- [ ] **Step 1: Replace `fig_corridor.py` entirely**

```python
"""fig_corridor -- the reachable aerocapture corridor (problem & objective, §3).

Shaded capture corridor between the p99.5 (upper, crash-side) and p0.5 (lower,
escape-side) dynamic-pressure boundaries traced by a dispersed randomized
piecewise-constant MC (collect_corridor.py), with the deployed Mamba ensemble
and its undispersed nominal flying inside. Data: articles/paper/data/corridor.npz.
"""

import math

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    fl.style()
    d = np.load(fl.DATA / "corridor.npz", allow_pickle=True)
    e, lo, hi = d["energy_bins"], d["lower_pdyn"], d["upper_pdyn"]
    green = fl.C["mamba"]

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    ax.fill_between(e, lo, hi, color=green, alpha=0.13, lw=0, zorder=0)
    ax.plot(e, hi, color=green, lw=1.3, zorder=2)
    ax.plot(e, lo, color=green, lw=1.3, zorder=2)

    ens_e, ens_p = d["ens_energy"], d["ens_pdyn"]
    alpha = max(0.04, min(0.22, 1.5 / math.sqrt(max(len(ens_e), 1))))
    for te, tp in zip(ens_e, ens_p, strict=True):
        ax.plot(te, tp, color=green, lw=0.5, alpha=alpha, zorder=3)
    ax.plot(d["nominal_energy"], d["nominal_pdyn"], color="#111", lw=1.6, zorder=4, solid_capstyle="round")

    ax.axvline(0.0, color="#555", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(float(e.min()), float(e.max()))
    ax.set_ylim(0, float(np.nanmax(hi)) * 1.08)
    ytop = ax.get_ylim()[1]
    ax.text(-0.3, ytop * 0.97, "bound\n(E < 0)", fontsize=8, color="#555", va="top", ha="right")
    ax.text(0.3, ytop * 0.97, "hyperbolic\n(E > 0)", fontsize=8, color="#555", va="top", ha="left")
    ax.set_xlabel("orbital energy (MJ/kg)")
    ax.set_ylabel("dynamic pressure (kPa)")
    ax.legend(handles=[
        Patch(facecolor=green, alpha=0.16, edgecolor=green, label="reachable capture corridor"),
        Line2D([], [], color=green, lw=1.0, alpha=0.6, label=f"deployed ensemble ({len(ens_e)})"),
        Line2D([], [], color="#111", lw=1.6, label="undispersed nominal"),
    ], loc="upper left")
    fig.tight_layout()
    fl.save(fig, "fig_corridor")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render from the small-scale corridor.npz and inspect**

Run: `cd /Users/govit/Git/Govit/Aerocapture/articles/paper/scripts && uv run python fig_corridor.py`
Expected: `wrote articles/paper/figures/fig_corridor.svg`. Rasterize and view (band between two smooth green boundaries, ensemble spaghetti + black nominal inside, E=0 divider):
```bash
SCR=/private/tmp/claude-501/-Users-govit-Git-Govit-Aerocapture/bde7d896-ec32-488f-9789-2cb52745ca76/scratchpad
cp /Users/govit/Git/Govit/Aerocapture/articles/paper/figures/fig_corridor.svg $SCR/fig_corridor.svg
printf '#set page(width: auto, height: auto, margin: 6pt)\n#image("fig_corridor.svg", width: 760pt)\n' > $SCR/wc.typ
typst compile --format png --ppi 150 --root $SCR $SCR/wc.typ $SCR/corridor_new.png && echo ok
```
Then Read `$SCR/corridor_new.png`. (Boundaries may be a little rough at 8k sims — Task 3's full run smooths them.)

- [ ] **Step 3: Commit the figure script**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/scripts/fig_corridor.py
git commit -m "paper(corridor): reachable-corridor figure (band + ensemble + nominal)"
```

---

### Task 3: Full run, caption, compile, verify

**Files:**
- Regenerate: `articles/paper/data/corridor.npz`, `articles/paper/figures/fig_corridor.svg`
- Modify: `articles/paper/paper.typ` (Figure-1 caption + the surrounding §2.1 sentence if it names the four-envelope construction)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: the deployed corridor artifacts + updated caption.

- [ ] **Step 1: Run the full 300k collection**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python articles/paper/scripts/collect_corridor.py --n-sims 300000 2>&1 | tail -4`
Expected: capture rate 0<r<100%, `corridor: <N> captures, ~200/200 energy bins populated`, `wrote ...`. If many tail bins are empty (`pop` well under 200), rerun with `--n-sims 1000000`.

- [ ] **Step 2: Regenerate the figure at full resolution and inspect**

Run: `cd /Users/govit/Git/Govit/Aerocapture/articles/paper/scripts && uv run python fig_corridor.py` then rasterize as in Task 2 Step 2 and Read the PNG.
Expected: smooth, monotone-ish band spanning the energy range; ensemble + nominal comfortably inside the band; no empty gaps.

- [ ] **Step 3: Update the Figure-1 caption in `paper.typ`**

Find the `#fig("fig_corridor.svg", [...])` block (§2.1) and replace its caption body with:
```
Reachable aerocapture capture corridor in the (orbital energy, dynamic pressure)
plane, traced by a dispersed Monte-Carlo of randomized piecewise-constant bank
profiles. The shaded band spans the corridor: the upper edge is the $p_(99.5)$
dynamic pressure of all capturing trajectories (the crash-side limit), the lower
edge the $p_(0.5)$ of trajectories capturing below a $5000$ km apoapsis (the
escape-side limit). The vehicle enters hyperbolic ($E > 0$, right) and bleeds
energy into a bound orbit ($E < 0$, left); the deployed Mamba ensemble and its
undispersed nominal (heavy line) fly well inside the corridor.
```
Also scan the §2.1 body prose (lines ~178–186) for any sentence describing the corridor as "two constant-bank profiles / overshoot / undershoot boundaries" and reconcile it with the new construction if it now contradicts the figure (keep the physics description; only fix wording that specifically claims the figure shows the old envelopes).

- [ ] **Step 4: Compile the paper and verify Figure 1 in-context**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
typst compile articles/paper/paper.typ articles/paper/paper.pdf 2>&1 | head -3 && echo COMPILE_OK
SCR=/private/tmp/claude-501/-Users-govit-Git-Govit-Aerocapture/bde7d896-ec32-488f-9789-2cb52745ca76/scratchpad
typst compile --format png --pages 3 --ppi 110 articles/paper/paper.typ "$SCR/paper_corridor.png" >/dev/null 2>&1 && echo p3ok
```
Then Read `$SCR/paper_corridor.png`: the new corridor renders on page 3, caption matches, STIX-serif consistent with the other figures.

- [ ] **Step 5: Commit the regenerated artifacts + caption**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/data/corridor.npz articles/paper/figures/fig_corridor.svg articles/paper/paper.typ
git commit -m "paper(corridor): deploy 300k reachable corridor + caption"
```

---

### Task 4: Whole-branch doc sync + final commit

- [ ] **Step 1: Invoke the smart-commit skill over the whole branch**

Use the `smart-commit` skill, telling it to take the whole git branch into account: sync any stale doc reference (e.g. a `paper_resume.md` line recording that `fig_corridor` is now the dispersed-piecewise reachable corridor built by the parameterized `collect_corridor.py`, and that `corridor.npz` no longer holds the 4-envelope schema) and make a final commit of anything outstanding for this feature. Do NOT stage the user's in-flight exp-13 files (`TODO.md`, `articles/paper/data/robustness_retrain.json`, `experiments/paper/13_robustness_retrain.sh`, `configs/training/paper/robustness_retrain/*.toml`).

---

## Self-Review

**Spec coverage:** Dispersed piecewise MC + random bank U[0,180], N/n_sims as CLI params → Task 1 collector + Global Constraints. Env dispersions ON via base config → BASE_TOML + Task 1. Streaming histogram p99.5/p0.5 + smoothing → `build_corridor`/`_percentile_from_hist`/`_smooth`. apoapsis<5000 lower gate → `hist_lo`. Overlay (ensemble + nominal) → `build_overlay`. Batched/memory-bounded → chunk loop + `del batch`. Figure (band + ensemble + nominal, drop 4-zone) → Task 2. Caption → Task 3 Step 3. Smoke/verify (override-effect, populated bins, ordering) → Task 1 Steps 2–4, Task 3 Steps 1–2. Reserved seed offsets → Global Constraints. Final smart-commit → Task 4. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step is complete; every run step has an exact command + expected output. `PDYN_MAX = 2.8` is a concrete value (Task 3 Step 1 flags a rerun if tails are empty, which would also be the signal to raise it).

**Type consistency:** `corridor.npz` keys written in Task 1 (`energy_bins, lower_pdyn, upper_pdyn, nominal_energy, nominal_pdyn, ens_energy, ens_pdyn`) exactly match the Task 2 figure reads. Column indices (`FR_*`, `TC_*`) match the Global Constraints and the verified source. `build_corridor(args)` / `build_overlay(n_ens)` signatures match their `main()` call sites. CLI flag names match argparse dests used in `build_corridor`.
