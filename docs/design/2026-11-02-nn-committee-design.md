# Committee of the headline seeds: one deployed artifact from the three Mamba 962 seeds

> **Date:** 2026-11-02. **Status:** closed, negative (user go-ahead on the pre-registered design 2026-11-02; the pilot
> decided it the same day, Section 6). **Issue:** #232. **Record:** `experiments/committee/README.md`.
> Vocabulary: `CONTEXT.md` (Committee, Champion, Sizing Tail, Regime).

## 1. Hypothesis

The v4 record separates the networks from the classical laws at sizing depth and not from one
another: the three headline Mamba 962 seeds span 15.8 m/s of CVaR99.9 (158.5 +- 15.8 on the
10 x 100,000 confirmatory pool), the deployment rule fixed seed 1 blind, and it is the worst far
tail of its family (173.9). Read as policy-level variance, that spread invites the textbook
reducer: fly the three seeds as one artifact and combine their commands. No training, one Rust
layer type, one confirmatory flight. An ensemble number is quotable only for an ensemble that is
genuinely deployed, with its command-combination rule and computational cost.

## 2. Design (pre-registered in #232)

- **Members.** Primary arm: the three headline Mamba 962 seeds
  (`training_output/ou_marginal/hl_mamba_p962{,_s2,_s3}`), one family, one recipe. Second arm:
  those three plus the three headline dense 515 seeds, six members across two families at parity.
  Every member shares `input_mask`, `normalization` and the `atan2_signed` decoder.
- **Rules.** Each member keeps its own recurrent state on the shared input; the 2-output heads
  combine before the one atan2 decode. `circular_mean` (primary): sum of the unit-normalised
  heads, one vote per member. `raw_sum`: sum of the raw heads. `medoid`: the raw head of the
  member closest to the circular mean (bit-identical to that member on that tick).
- **Win rule.** On the confirmatory pool: CVaR99.9 below the deployed seed's 173.9 by more than
  two replicate standard errors, 10^6 of 10^6 captured, heat-load exceedances not above the
  deployed seed's. At or below the family mean 158.5 the strong outcome.
- **Non-goals.** Training a committee; paper placement and the deployment rule, decided from the
  result.

## 3. Mechanism in the runtime

A `committee` layer type (`src/rust/src/data/neural/layers/committee.rs`, `LayerSpec::Committee
{ rule, members, member_sha256 }`): the members are whole v2 layer lists nested in the JSON
(`weights.layer_<i> = {"members": [<member weights block>, ...]}`), each with its own
`LayerState` chain (`LayerState::Committee`), the forward step shared with the plain model through
`forward_layer`. `NnState`, `nn_bank_angle`, the input path and every existing layer codec are
untouched; the `[[network.architecture]]` cross-check passes a committee when at least one member
matches the TOML (`toml_architecture_agrees`). Deploy-only: the torch mirror refuses the layer,
no chromosome layout exists. The builder (`aerocapture.training.committee`) assembles the JSON
from deployed `best_model.json` files, records their sha256s in the spec, copies member 0's
`best_params.json` (the cell flies the deployed seed's scaffolding) and writes a `committee.json`
sidecar the confirmatory driver reads as the cell's finished marker.

Gates: a one-member committee flies bit-identically to its member under `raw_sum` and `medoid`
(Rust unit test and `tests/test_committee.py` through the golden NN config); `circular_mean`
and `sign_leader` normalise the head, so a one-member committee matches to the last ulp of
atan2 and the trajectory diverges by a few 1e-2 m/s over a flight (the same-seed-x3 check in the
record), which is why the bit-identity gate names the two raw-head rules. The six guidance
goldens and `rl_env_parity` are untouched.

## 4. Pilot (2000 confirmatory-range scenarios, per_draw)

The pool is `make_confirmatory_pools(42, 1, 2000)` (`paired_vs_champion.py --replicates 1 --n 2000`).
Every committee row flew the base TOML's navigation filter and command shaping, not the deployed
seed's trained `best_params.json` (the builder copied none until review); the primary arm with the
deployed seed's scaffolding on the same pool gives 72.15% capture, mean 144.2, CVaR95 208.5, max 314.

| cell | capture | mean | CVaR95 | max | paired mean vs champion |
|---|---|---|---|---|---|
| champion `hl_mamba_p962` | 100.00% | 110.6 | 119.5 | 151 | |
| `hl_mamba3_circular_mean` (primary) | 72.70% | 143.0 | 203.8 | 311 | +32.3, better on 0.3% |
| `hl_mamba3_raw_sum` | 79.80% | 150.3 | 190.8 | 250 | |
| `hl_mamba3_medoid` | 81.90% | 155.4 | 271.0 | 296 | |
| `hl_mixed6_circular_mean` | 99.95% | 118.0 | 137.6 | 198 | |

Every arm fails the win rule at n = 2000 (the champion loses 0 of 2000 and 0 of 10^6); the
pre-registered 10^6 flight was flown for the primary arm only, as the record's row.

## 5. Mechanism: the seeds are different policies, not perturbations of one

On 40 shared scenarios the three Mamba seeds command the same bank magnitude (mean |bank| 133 /
127 / 114 deg) with independent lateral strategies: pairwise bank-sign agreement 0.49 / 0.57 /
0.41 (chance), sign flips per flight 25 / 145 / 15. Averaging a +130 deg and a -130 deg command
is a lift-down command no member issued, and the capture loss follows. A post-hoc rule that
takes the magnitude from all members and the sign from the deployed seed (`sign_leader`)
recovers capture (99.40% / 99.45% for the two arms) and not the policy (paired mean +41 / +26
m/s, CVaR95 266 / 201): the magnitude profile is coupled to each member's own lateral timing.

## 6. Verdict and what it closes

Negative. Policy averaging needs members in one basin; independently trained seeds of this
problem are not, because the lateral plan is a discrete choice the search makes per run. The
seed spread of Section 6.2 is therefore not reducible by combining seeds at deploy time; it is
reducible only by selection (a tail-aware gate) or by more seeds. The layer type stays (it is the
cheapest way to fly any future ensemble candidate); the arms are not paper material. Reopen only
with members that share a lateral plan by construction (a signed-bank scheme with the sign fixed
by the lateral logic, or seeds warm-started from one champion).
