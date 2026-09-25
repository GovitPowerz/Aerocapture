# Learned dynamics (world model) of the aerocapture plant

Issue #113. A learned, action-conditioned dynamics model of the aerocapture plant is trained on
simulator flights, then asked to do the three jobs a world model is for: predict a trajectory,
predict what an intervention changes, and plan. It is scored the way a world-model paper would
score it, and it plans through FNPAG's corrector next to a clairvoyant replay of the true plant.
The deliverable is where it breaks and why.

Nothing in the simulator or the deployed cells changes. Everything runs through the existing PyO3
seam, `aerocapture_rs.BatchedSimulation`, with the bank injected from Python; the plant constructs
it with `auto_reset=False` (#153), so a finished flight freezes at its terminal state instead of
auto-resetting on a fresh seed and being integrated until the whole batch ends. The win is not
the skipped physics (Rayon-parallel, under 0.1 us per slot-tick amortized) but the skipped
observation build, serial at about 1 us per slot: a frozen slot's rows are copied from its cached
terminal state, so a 1,000-slot step falls from 1.2 ms to 0.2 ms once the slots are done and a
1,000-flight lockstep batch from 1.2 s to 0.7 s. The #155 rerun on the frozen plant reproduces every
model and planner number in `world_model.json`; the change shows only in the wall-clock numbers,
and there only where the plant dominates: the data stage and the clairvoyant replay (its ms per
flight-replan and the planning figure's cost panel with it). The train times and the learned
planners' ms per replan moved within run noise.

## Result in one paragraph

The models fit one step well (teacher-forced error 0.012 to 0.013 standardized units, against
0.021 for "nothing changes") and free-run better than "nothing changes" for 100 to 161 ticks (GRU)
or 62 to 97 ticks (one-step MLP). Every downstream use that needs a longer horizon fails.
Free-running, the predicted altitude drifts away from what the predicted speed and flight-path
angle imply (h' = V sin(gamma) holds 35 to 61 times worse than on the flown trajectories after 50
ticks), and a pass flown from entry reads as a crash. All six models predict a crash for every
constant bank from 20 to 110 deg, where the plant ranges from a 10,800 km apoapsis to a sure crash.
FNPAG's corrector trusts that answer and pins the minimum bank, and every learned-model planner
captures with a 5,700 to 10,000 km median apoapsis error and 770 to 965 m/s of correction delta-v.
The clairvoyant replay flies the same corrector to 137 m/s. The sampled uncertainty is honest at
one tick (85 to 90% coverage of the 90% interval) and overconfident by 20 ticks (44 to 52%);
pooling three training seeds restores 79 to 81%. Models with the same validation likelihood
disagree on the direction of the bank's effect on the apoapsis at the bounce: all three GRUs and
one of three MLPs get it backwards.

## Plant, state and action

The plant is the paper's NN training environment (`configs/training/msr_aller_nn_atan2_train.toml`:
medium dispersions, `per_draw` noise, NN-tuned navigation) with a stub model that exposes all 35
candidate inputs (`configs/wm_medium.toml`). The out-of-distribution plant `configs/wm_high.toml`
raises to `high` the four domains the paper's robustness stress raises: atmosphere, density
perturbation, navigation, nav filter. The issue's second shift, legacy to `per_draw` noise, is not
run: it needs a second set of models trained on legacy-regime data, and it was traded at design
time for three training seeds per model. The planner's own flights are the third shift.

- **State, 42 channels.** The 35 candidate NN inputs as the onboard software sees them
  (navigation estimates, the config's calibrated normalization), plus the 7 aux channels of the RL
  env: energy, dynamic pressure, predicted correction delta-v x3, heat flux and heat load
  fractions. Two inputs never vary on this plant (`cos_bank_nominal`, `inclination_err_rate`), so
  the loss and every metric use the other 40. The density dispersion is not in the state: the
  plant is partially observed.
- **One signed bank per 1 s guidance tick.** The command replaces the deployed NN's output, so it
  is gated and command-shaped like one (bank acceleration limit), then flown through the pilot
  model (rate limit, biases). Lateral guidance, the exit-phase law and the thermal limiter are
  bypassed, as for the deployed `full_neural` NN. The shaper is part of the plant the model learns.
- **Deploy timing.** `step(a_k)` flies tick k and returns the navigation at tick k + 1 (#149), the
  input a deployed NN reads when choosing a_{k+1}; the telemetry inputs record the shaped command.
  Row k of a flight is that post-a_k observation, and the model predicts row k + 1 from rows <= k
  and a_{k+1}. `tests/test_world_model.py` pins this contract and the replay determinism that the
  oracle and the counterfactuals rest on: re-flying a seed on a shared action prefix reproduces
  the prefix bit for bit, whatever its batch neighbours.

## Data

The behavior policy is a random piecewise-constant signed bank (`wm_plant.bank_schedule`): a
per-flight base magnitude drawn from U(40, 100) deg, segments of log-uniform duration in
[5, 200] ticks around it (sigma 25 deg, random sign), plus a 3 deg Ornstein-Uhlenbeck jitter
(tau 10 s). The base range sits on the capture corridor: on this plant a constant bank of 20 to
50 deg leaves on a 3,900 to 10,800 km apoapsis, 65 deg hits the 500 km target on the median flight,
and 70 deg or more crashes most flights (the bank sweep in `world_model.json`). Open-loop random
banks rarely thread that corridor, so only 9% of training flights exit with an apoapsis between
200 and 1,500 km.

| pool | config | seeds | flights | steps | crash | capture | apoapsis 200-1500 km |
|---|---|---|---|---|---|---|---|
| train | `wm_medium` | `WORLD_MODEL_SEED_OFFSET` stream | 10,000 | 3.92 M | 50.8% | 48.9% | 9.5% |
| val | `wm_medium` | same stream | 1,000 | 0.39 M | 51.6% | 48.0% | 8.4% |
| test_id | `wm_medium` | same stream | 1,000 | 0.39 M | 52.7% | 47.2% | 9.3% |
| test_ood | `wm_high` | same stream | 1,000 | 0.38 M | 43.7% | 46.0% | 8.2% |
| planning | `wm_medium` | same stream, the next 1,000 | 1,000 | flown by each planner | | | |

Two closed-loop pools join the evaluation once the planners have flown: `closed_loop` (the
clairvoyant planner's 1,000 flights) and `self_planned` (the GRU s0 planner's own flights).

## Models

`wm_model.py`. Both are plain PyTorch, not the `torch_mirror` layers: the mirror has no
probabilistic head, and flying a learned dynamics model onboard is out of scope here. Both models
read the standardized state x_k and the next action as
(sin a_{k+1}, cos a_{k+1}), and output a diagonal Gaussian over the standardized increment
x_{k+1} - x_k. Training is teacher-forced Gaussian negative log-likelihood over whole flights
(Adam 1e-3 with cosine decay to 1e-4, gradient clip 1, 64 flights per batch, 40 epochs, the best
validation epoch kept), three training seeds each.

| model | parameters | best val NLL (s0 / s1 / s2) | best epoch | train time |
|---|---|---|---|---|
| GRU(128) + MLP head | 198,484 | -2.97 / -2.91 / -2.97 | 24 / 23 / 25 | 12 min |
| one-step MLP (ablation, no memory) | 98,900 | -3.11 / -3.12 / -3.11 | 38 / 39 / 39 | 2.1 min |

Every GRU's validation NLL bottoms out at epoch 23 to 25 and then climbs by 1.9 to 5.9 nats while
the training NLL keeps falling, so the best-validation checkpoint is the one kept. The memoryless
MLP reaches the better validation NLL.

## The six evaluations

All on the in-distribution test pool unless named. Standardized RMSE is over the 40 varying
channels, in units of the training pool's per-channel standard deviation, median over
(flight, start) pairs with starts every 50 ticks.

1. **Rollout error vs horizon** (`fig_rollout_error.svg`). Free-running on the flown banks vs
   teacher-forced vs persistence (x stays at x_t0), plus the kinematic residual
   |altitude step - V sin(fpa) dt|.
2. **Calibration** (`fig_calibration.svg`). 32 sampled free runs per start: reliability of
   central intervals and ensemble CRPS at 1 to 200 ticks, for each model and for the 96-member
   ensemble of a kind's three training seeds.
3. **Out-of-distribution** (`fig_ood.svg`). Rollout error, CRPS and 90% coverage at 50 ticks on
   four pools: in-distribution, high dispersions, the clairvoyant planner's flights, and the GRU
   s0 planner's own flights.
4. **Tail prediction** (`fig_tail.svg`). A free run on the flown banks from tick 100, 200, 300 or
   400, from the descent to late in the ascent (absolute ticks, because a start at a fraction of
   each flight would leak its length), predicts the
   terminal correction delta-v, a crash reading as infinite. Two ROC AUCs: the delta-v tail of the
   clairvoyant planner's flights ("not captured or delta-v at or above the pool's p95"), against
   persistence = the current navigation delta-v estimate; and crashes in the in-distribution test
   pool, against persistence = the lowest current osculating periapsis. Flights already over at
   the start are dropped.
5. **Counterfactual response** (`fig_counterfactual.svg`). On the clairvoyant planner's flights,
   the bank magnitude is pulsed by +10 or -10 deg for 20 ticks at 0.25 / 0.5 / 0.75 / 1.0 of the
   bounce tick, the rest of the flown banks unchanged. The true response comes from replaying the
   seed. Three readouts: the orbital energy and the osculating apoapsis 30 ticks after the pulse
   starts (apoapsis only where the orbit is bound on both branches), and the exit apoapsis. Model
   rollouts for the 30-tick readouts run a fixed horizon.
6. **Planning** (`fig_planning.svg`). FNPAG's corrector (`wm_planner.py`): every 10 ticks,
   bisect a constant bank magnitude so the predicted exit apoapsis hits 500.13 km (11
   predictions per replan, FNPAG's deployed bank limits and apoapsis tolerance), roll sign from a
   port of the lateral reversal law at FNPAG's deployed gains. The predictor is a learned model
   (six arms) or a replay of the true plant from t = 0 on the same seed (the oracle, which sees
   the flight's future noise). A model rollout ends at a predicted exit or impact like the plant,
   and also at a post-bounce dip back into the atmosphere, which it scores as a crash (the plant
   and fnpag.rs keep integrating). That rule is not what removes the capture region: without it,
   four of the six models still predict a crash at every bank of the sweep, and the other two
   (GRU s2, MLP s0) leave the atmosphere around tick 180 in a state with no usable apoapsis. The
   fourth panel is the planner's first query: the exit apoapsis of a constant bank flown from
   tick 0, model vs plant, 200 flights.

## Results

Numbers are medians over pairs, reported as the range over the three training seeds.

| evaluation | GRU | one-step MLP | reference |
|---|---|---|---|
| one-step error, teacher-forced | 0.013 | 0.012 | persistence 0.021 |
| free-running error at 10 / 50 / 200 ticks | 0.06 / 0.29-0.34 / 2.5-16.5 | 0.07-0.09 / 0.34-0.51 / 110-500 | persistence 0.19 / 0.70 / 1.48 |
| horizon where free-running loses to persistence | 100-161 ticks | 62-97 ticks | |
| kinematic residual at 1 / 50 / 200 ticks | 3-5x / 35-61x / 150-3,100x the flown trajectories' | | flown 0.002 km per tick |
| 90% coverage at 1 / 20 / 50 ticks | 0.85-0.90 / 0.45-0.52 / 0.44-0.48 | 0.89-0.90 / 0.44-0.51 / 0.43-0.48 | nominal 0.90 |
| 90% coverage at 10 / 20 / 50 ticks, 3-seed ensemble | 0.79 / 0.80 / 0.81 | 0.79 / 0.80 / 0.80 | nominal 0.90 |
| 90% coverage at 1 tick, high dispersions | 0.70-0.74 | 0.74-0.75 | nominal 0.90 |
| 90% coverage at 50 ticks, own planned flights | 0.37 (s0) | | 0.44 on the test pool |
| delta-v tail AUC from tick 100 / 200 / 300 / 400 | 0.16-0.50 / 0.14-0.76 / 0.26-0.77 / 0.50-0.98 | 0.50 / 0.12-0.50 / 0.26-0.47 / 0.20-0.90 | persistence 0.16 / 0.13 / 0.72 / 0.98 |
| crash AUC from tick 100 / 200 / 300 / 400 | 0.49-0.50 / 0.54-0.94 / 0.80-0.98 / 0.75-0.96 | 0.50-0.58 / 0.72-0.92 / 0.89-0.96 / 0.78-0.91 | persistence 0.82 / 0.88 / 0.99 / 0.99 |
| counterfactual sign agreement, energy, at 0.5 / 0.75 / 1.0 of the bounce tick | 0.51-0.99 / 1.0 / 1.0 | 0.91-1.0 / 1.0 / 1.0 | |
| counterfactual slope, energy, at 0.5 / 0.75 / 1.0 | -0.01-0.24 / 0.26-0.46 / 0.38-0.41 | 0.22-1.72 / 0.25-0.60 / 0.27-0.68 | 1 |
| counterfactual sign agreement, apoapsis, at 0.75 / 1.0 | 1.0 / 0.00-0.13 | 1.0 / 0.04-1.0 | |
| usable exit-apoapsis counterfactuals | 0 of 8,000 | 0 of 8,000 | |

Planning on the 1,000-seed planning pool (`per_draw` noise):

| planner | capture | delta-v p50 | p95 | CVaR95 (95% CI) | abs apoapsis error p50 / p95, km | periapsis p50, km | first 150 ticks at bank_min | ms per flight-replan |
|---|---|---|---|---|---|---|---|---|
| clairvoyant plant replay | 100% | 137 | 171 | 175 (174-177) | 13 / 31 | -23 | 0% | 3.6 |
| GRU s0 | 100% | 946 | 1256 | 1342 (1314-1365) | 9,844 / 32,979 | 36 | 99.3% | 0.9 |
| GRU s1 | 100% | 963 | 1269 | 1342 (1318-1363) | 9,991 / 33,900 | 31 | 99.3% | 1.4 |
| GRU s2 | 100% | 859 | 1253 | 1342 (1313-1365) | 7,803 / 31,650 | 42 | 99.0% | 0.8 |
| MLP s0 | 100% | 772 | 937 | 977 (963-990) | 5,745 / 9,165 | 33 | 86.5% | 0.4 |
| MLP s1 | 100% | 938 | 1256 | 1342 (1314-1365) | 9,736 / 31,944 | 38 | 99.3% | 0.4 |
| MLP s2 | 100% | 823 | 1253 | 1342 (1313-1365) | 7,414 / 31,650 | 43 | 98.1% | 0.3 |
| FNPAG deployed cell (context) | 99.5% | 122 | 140 | 150 (145-155) | 20 / 74 | 8 | | |

Paired on the same seeds, every learned planner costs 634 to 829 m/s more than the clairvoyant one
on average and wins on no flight (`vs_oracle` in the JSON).
Delta-v statistics are over captured flights (`paper_stats.run_stats`). Planning cost is wall time
per flight and replan, batched over the 1,000 flights on this machine: the replay re-flies each
flight from t = 0 with 14 Rayon threads, the learned models run on 10 torch threads and stop a
rollout once every flight in the batch has terminated, so a model that predicts early crashes
plans cheaply. For scale, FNPAG flies a whole flight, every replan included, in 89 ms on one
thread (`docs/performance.md`).

FNPAG's deployed cell is context, not a head-to-head: it runs its own navigation tuning, replans
every 2 s and hands off to the exit-phase law after the bounce. The clairvoyant planner hits the
apoapsis more tightly than FNPAG (median error 13 km vs 20 km) yet pays 15 m/s more at the median
and 25 m/s more at CVaR95. Eight of those median m/s are the periapsis-raise burn: it exits with a
median periapsis of -23 km against FNPAG's +8 km. The replanned constant bank has no exit-phase
law, the likeliest source; this experiment does not separate it from the 10 s replan period and
the navigation tuning.

## What counts as failure

- **Rollout.** A free-running prediction fails at the horizon where its median error exceeds
  persistence's.
- **Calibration.** A 90% interval fails when it covers less than 80% of outcomes.
- **Tail.** A predictor fails when its AUC does not beat persistence (the navigation's own
  current delta-v estimate).
- **Counterfactual.** A predictor fails when it gets the sign of the response wrong more often
  than right.
- **Planning.** A learned-model planner fails when its captured delta-v CVaR95 exceeds the
  clairvoyant planner's 95% interval.

## Where it fails

### 1. Free runs leave the kinematics behind, so the planner's question has no usable answer

On the flown trajectories, an altitude step and V sin(fpa) dt agree to 0.002 km per tick. A GRU
free run misses that identity by 3 to 5 times as much at one tick, 35 to 61 times at 50 ticks and
150 to 3,100 times at 200 ticks, and its altitude is 7 to 12 km off after 100 ticks. The planner's
first query is the exit apoapsis of a constant bank flown from the first tick, which needs a
rollout to the exit: hundreds of ticks, beyond the 100 to 161 (GRU) and 62 to 97 (MLP) ticks where
the models beat persistence. On 200 test flights the plant's median answer is 10,814 km at 20 deg,
499 km at 65 deg and a crash from 70 deg. All six models answer "crash" at every bank from 20 to
110 deg. None has a capture region.

Diagnosis. Each model outputs an independent Gaussian increment per channel. Teacher forcing hands
it a consistent state at every step, so the training loss never meets an altitude that disagrees
with the predicted speed and flight-path angle, and nothing in the model enforces h' = V sin(gamma)
once its own outputs feed back. The residual growth measures that coupling coming apart. One-step
quality does not rank the models either: the MLP has the lower teacher-forced error (0.012 vs
0.013) and the better validation NLL, and the worse free run, diverging numerically after about
100 ticks.

The same drift decides the tail evaluation. The navigation's own delta-v estimate (persistence)
ranks the clairvoyant planner's tail backwards around the bounce (0.16 at tick 100, 0.13 at 200)
and near-perfectly late in the ascent (0.98 at 400). At tick 100 five of six models predict a crash
for every flight (none crashed) and all six sit at or below chance. Two GRUs rank the tail forward
at tick 200 (0.61 and 0.76, where persistence is backwards), and GRU s2 edges past persistence at
300 (0.77 against 0.72) and matches it at 400; no other model reaches it late in the pass. On
crashes in the test pool, where persistence is the lowest current periapsis (0.82 / 0.88 / 0.99 /
0.99 from tick 100 / 200 / 300 / 400), three models beat it at tick 200 (GRU s0 0.94, MLP s0 and s2
0.92) and none at the other three starts.

### 2. The corrector turns a wrong model into a saturated command

FNPAG's bisection reads only the sign of the apoapsis residual. When the model answers "crash" at
both ends of the bank bracket, the residual is negative everywhere and the corrector commands the
minimum bank. Five of the six learned planners sit at the minimum bank for 98 to 99% of the first
150 ticks and MLP s0 for 87% (the clairvoyant planner: 0%). The vehicle skims the atmosphere and
leaves with a median apoapsis error of 5,700 to 10,000 km, which costs 770 to 965 m/s of correction
delta-v against 137 m/s for the same corrector on the true plant. Every learned planner still
captures 100% of flights, so the capture rate alone would have scored this as a success.

### 3. The sampled uncertainty covers the noise, not the model

The 90% interval of 32 sampled free runs covers 85 to 90% of outcomes at one tick and 45 to 52% at
20 ticks (GRU). Between 10 and 20 ticks the median error of the ensemble mean is 1.25 to 2.1 times
the median ensemble spread (GRU); a calibrated Gaussian ensemble keeps that ratio at 0.67. The 32
samples share one model's mean dynamics, so that model's bias is in every sample. Pooling the three
training seeds (96 members) lifts the 90% coverage at 10 / 20 / 50 ticks to 0.79 / 0.80 / 0.81
(GRU) and 0.79 / 0.80 / 0.80 (MLP), and cuts the CRPS by 5% (GRU) and 11% (MLP) at 10 ticks and
12% and 24% at 50. The missing term is epistemic: seeds disagree where the model is wrong. Three
members still under-cover.

Distribution shift shows up in the same place. The high-dispersion plant moves the median error
by little (0.013 to 0.016 at one tick, 0.29-0.34 to 0.31-0.35 at 50 ticks) but drops one-step
coverage from 0.85-0.90 to 0.70-0.75 (ensemble: 0.94-0.95 to 0.81-0.82). On the flights it planned
itself, GRU s0's 50-tick coverage falls to 0.37, against 0.44 on the test pool.

### 4. Equal likelihood, opposite answers to "what does this bank do"

A +10 or -10 deg pulse of 20 ticks at half the bounce tick changes the orbital energy 30 ticks
later by a median 5.2 kJ/kg on the plant. Five of six models get the direction right on 91 to 100%
of flights and GRU s1 on 51%, a coin flip. The size is worse: the regression slope of predicted on
true change runs from -0.01 (no response) to 1.7 at that time, and from 0.25 to 0.68 later in the
pass, where the effect is underestimated 1.5 to 4 times.

The apoapsis, the quantity the corrector steers, splits the models. Thirty ticks after a pulse at
the bounce, the plant's osculating apoapsis moves by a median 78 km. All three GRUs predict the
opposite direction (sign agreement 0.00, 0.00 and 0.13), and so does MLP s1 (0.04), whose
validation NLL is within 0.02 nats of the two MLPs that get it right on every flight. With the
pulse at 0.75 of the bounce tick every model that keeps the orbit bound gets the sign, but against
a true median change of 13,000 km the slope of predicted on true change is 0.0001 to 0.3 for four
models (MLP s0 keeps no flight bound; MLP s1 keeps 150 of 2,000, at a slope of 43). Earlier pulses
have no apoapsis to score: the orbit is still hyperbolic 30 ticks later. The exit-apoapsis version
has no answer at all: for every model, none of the 8,000 pulses ends in a bound predicted exit on
both branches.

Diagnosis. Validation likelihood does not tell these models apart, so it does not identify the
bank's 30-tick effect. A one-step loss scores the bank's effect one tick at a time; the 30-tick
effect is a product of 30 such steps through altitude and density, and nothing in the objective
checks the product.

### What it says

Every failure traces to one fact: the models learned the conditional distribution of the next
navigation vector given the recent ones, and the three downstream jobs need the transition law.
Dispersion shift barely moves the mean error. The horizon does, and the planner's own flights
cost calibration (0.37 coverage at 50 ticks for the model that flew them). The evidence points to
two changes, both cheap here: train on multi-step rollouts, so the objective sees the drift and the
action response, and ensemble, which already recovers most of the calibration. A state built
around the plant's seven physical states is the heavier third option.

## Compute

One Apple M4 Pro (14 cores, 48 GB), CPU only, 10 torch threads, 56 minutes summed over the stages
(stage `meta` fields and per-stage wall times in `world_model.json`; the #155 rerun was resumed
once from its cached pools and models after a machine sleep, so the stages did not run on one
continuous clock):

| stage | wall time |
|---|---|
| data: 13,000 flights, 5.1 M steps, 340,000 to 352,000 env-ticks/s | 15 s |
| train: 3 GRU at 12 min (18 s per epoch), 3 MLP at 2.1 min | 42 min |
| plan: clairvoyant replay 3.5 min, learned models 12 to 49 s each, FNPAG 8 s | 6.2 min |
| eval 7.4 min, plot 1 s | 7.4 min |

The estimate before starting was 4 to 5 hours. `BatchedSimulation` and small CPU models made it an
hour; freezing the finished slots (#153) then took the data stage from 24 s to 15 s and the
clairvoyant replay from 5.9 to 3.5 min.

## Reproduce

Build the extension first (`./build.sh`), then run one command from the repo root:

```bash
uv run python experiments/world_model/world_model.py
```

Stages run in order and each resumes from its cache under `training_output/world_model/`:
`data` (the four pools, seconds), `train` (six models), `plan` (seven planners on the planning pool
plus FNPAG's deployed cell), `eval`, `plot`. `--stages eval plot` recomputes the evaluations from
cached models and flights, `--stages plot` only redraws the figures from `world_model.json`,
`--force` recomputes cached artifacts, and `--arms` selects the planners of the `plan` stage.

## Files

- `world_model.py`: the driver, every stage, the figures.
- `wm_plant.py`: the seam as the model sees it: behavior policy, stub model, lockstep plant, open-loop fly.
- `wm_model.py`: features, the GRU and MLP dynamics models, training, rollouts.
- `wm_planner.py`: the lateral port, FNPAG's bisection, the readouts, the two predictor arms, the MPC loop.
- `wm_metrics.py`: ensemble CRPS, central-interval coverage, rank AUC (the closed-form Gaussian CRPS that checks it lives in `tests/test_world_model.py`).
- `configs/wm_medium.toml`, `configs/wm_high.toml`: the two plants.
- `world_model.json`: every number above, per model and seed.
