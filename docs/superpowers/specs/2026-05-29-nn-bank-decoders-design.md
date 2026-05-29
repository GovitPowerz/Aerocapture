# NN bank decoders: `scaled_pi` + `delta`, with `(sin,cos)` angle inputs

Date: 2026-05-29
Status: design approved, pre-implementation
Scope: PSO-first (Rust runtime + config + warm-start + PSO training). PPO/V2Policy mirror deferred.

## Motivation

The NN bank-angle decoder currently has two output parameterizations:

- `atan2_signed` (full_neural): `bank = atan2(out[0], out[1]) in (-pi, pi]`.
- `acos_tanh` (magnitude_only): `bank = acos(out[0]) in [0, pi]`, lateral guidance picks the sign.

Both signed paths inherit the `+pi / -pi` representation seam: two outputs that are
physically identical (a bank of `+pi` and `-pi` are the same attitude) sit at opposite
ends of the decoder range. `atan2` papers over this for the network's output, but the
seam still bites in two places we want to attack directly:

1. We want to test a **unidimensional** signed output (the old "direct" path, removed as
   dead code in `e1f263a`) with a tunable knob so we can push the seam away from the
   operating region rather than relying on the 2-output atan2 trick.
2. We want a **bounded-delta** decoder where the network commands an increment on the
   previous realized bank. Accumulation sidesteps the seam entirely (the integrator
   lives in an unbounded angle space; only the final command is wrapped).

We also fix the seam on the **input** side: bank-angle history fed to the network is
currently a single normalized angle (`bank / pi`), which jumps from `+1` to `-1` across
an identical attitude. Cyclic features should be `(sin, cos)` pairs.

## Taxonomy (settled)

The two config axes are kept but their relationship is made explicit:

- **`mode` (NeuralNetMode) is a routing decision** -- who picks the sign and which
  scaffolding runs. `full_neural`: NN emits a signed bank, bypasses exit/lateral/thermal.
  `magnitude_only`: NN emits a magnitude in `[0, pi]`, lateral picks the sign,
  thermal/exit/FTC scaffolding run.
- **`output_param` (OutputParam) is a decoder decision** -- how raw network outputs map
  to that scalar. Each decoder's output semantics (signed vs magnitude) determine a
  compatible mode.

The `+pi / -pi` discontinuity only exists when the NN owns the sign (`full_neural`); in
`magnitude_only` the NN emits `[0, pi]` and lateral chooses the sign, so there is no
wrap. Therefore both new decoders are **signed -> full_neural-only**, exactly parallel to
how `acos_tanh` is magnitude_only-only.

| decoder            | outputs | last activation     | range / formula                  | mode            | knob          |
|--------------------|---------|---------------------|----------------------------------|-----------------|---------------|
| `atan2_signed`     | 2       | any                 | `atan2(o0, o1) in (-pi, pi]`      | full_neural     | --            |
| `scaled_pi` (new)  | 1       | `tanh` (validated)  | `n * pi * o0`, then wrapped       | full_neural     | `scaled_pi_n` |
| `delta` (new)      | 1       | `tanh` (validated)  | `prev_realized + delta_max * o0`, then wrapped | full_neural | `delta_max` |
| `acos_tanh`        | 1       | `tanh`              | `acos(o0) in [0, pi]`             | magnitude_only  | --            |

The current *advisory* "matched setups" warning at config load becomes a **hard
validation**: each decoder declares its required mode; a mismatch is a `ValueError` /
`DataError` at load, not a warning.

## Cross-cutting principle

- **Bank-angle inputs -> `(sin, cos)` pairs** (seam-free, bounded).
- **Bank output -> produced unbounded/continuous, wrapped once at the guidance->shaper
  boundary.** The pilot's `shortest_angle_diff` is already wrap-aware, so a `+1.5pi`
  command is realized as `-0.5pi` via the short path. The integrator base and the
  network's output space stay unbounded; only the boundary wraps.

## Component design

### 1. Config surface

`[guidance.neural_network]` gains:

- `output_parameterization = "scaled_pi" | "delta"` (in addition to the existing two).
- `scaled_pi_n` (float, default `1.0`) -- only meaningful for `scaled_pi`.
- `delta_max` (float, radians, default `0.35`) -- only meaningful for `delta`.

Rust: `OutputParam` enum gains `ScaledPi` and `Delta` variants (`#[serde(rename_all =
"snake_case")]`). The scalar knobs are new fields on `NeuralNetModel` (persisted in v2
JSON, default-valued for v1 / older v2 files). Config parsing in `config.rs` reads the
two floats from the `[guidance.neural_network]` block and cross-checks them against the
loaded model (mirrors the existing `output_parameterization` cross-check at
`a9372f8`).

Validation, at config/model load (extends `validate_output_size` /
`validate_output_activation`):

- `scaled_pi` and `delta` require last-layer `output_size == 1` and `activation == tanh`.
- `scaled_pi`, `delta`, `atan2_signed` require `mode == full_neural`.
- `acos_tanh` requires `mode == magnitude_only`.
- Mismatch -> hard error naming the offending decoder/mode pair.

### 2. Candidate input vector: +4 effective (raw array 25 -> 31)

The agreed growth is **net +4 effective inputs** for new configs. Under the recommended
append-only layout (below) the raw candidate array widens 25 -> 31 (three `(sin,cos)`
pairs appended), while the two seamed singles at indices 20 and 22 are retired from new
masks -- so a `scaled_pi`/`delta` config that uses all bank history sees +4 over the
prior 16-input baseline, not +6.

Three bank angles each become a `(sin, cos)` pair:

- exit-bank teacher (was index 20, single normalized angle)
- previous **commanded** bank (was index 22, `bank / pi`)
- previous **realized** bank (new)

Layout is **append-only** to preserve indices 0-19 and the lateral-telemetry semantics
(21 incl_err_rate, 23 time_since_flip, 24 incl_err_integral). Proposed canonical layout:

- Indices 0-24: unchanged in meaning. Indices 20 and 22 retain their existing single
  normalized-angle values for backward compat with the default mask and existing models.
- Index 25: `sin(exit_bank_teacher)`
- Index 26: `cos(exit_bank_teacher)`
- Index 27: `sin(prev_commanded_bank)`
- Index 28: `cos(prev_commanded_bank)`
- Index 29: `sin(prev_realized_bank)`
- Index 30: `cos(prev_realized_bank)`

That is +6 raw slots. To honor the agreed **net +4** growth and the "replace, not
duplicate" intent, the seamed singles at 20 and 22 are **dropped from the default mask**
(their `(sin,cos)` pairs supersede them); they remain physically present in the full
candidate array only if a legacy mask references them. Net effective growth for new
configs is +4 (three pairs added, two singles retired from use). `NN_FULL_INPUT_SIZE`
becomes 31 to host the appended pairs; the default mask stays `[0..16]` (untouched, fully
backward compatible).

> Implementation note (resolve in the plan): decide between (a) literally appending and
> retiring 20/22 from new masks as above, or (b) renumbering for a clean 29-wide vector.
> (a) preserves golden bit-identity for default-mask models with zero renumbering risk and
> is the recommended path. The numbers above are the (a) layout.

`input_mask` range validation upper bound updated to the new `NN_FULL_INPUT_SIZE`.

**Migration note:** any existing v2 model carrying an explicit `input_mask` that
references indices >= 20 needs a one-line mask update. Acceptable: these are research
models that get retrained. The default-mask path (`[0..16]`) and all existing golden
models are unaffected.

### 3. Decoders + boundary wrap (`gnc/guidance/neural.rs::nn_bank_angle`)

```
match nn.output_param {
    Atan2Signed => out[0].atan2(out[1]),
    AcosTanh    => out[0].acos(),
    ScaledPi    => wrap_to_pi(nn.scaled_pi_n * PI * out[0]),
    Delta       => wrap_to_pi(prev_realized_bank + nn.delta_max * out[0]),
}
```

- `ScaledPi`: `out[0] in [-1, 1]` (tanh head), so `bank in [-n*pi, n*pi]` before wrap.
  Knob `n` controls how far the wrap seam sits from the operating region.
- `Delta`: per-step increment hard-bounded to `+/- delta_max` (the "bounded delta"
  safety). Base is the previous realized bank (unbounded); the sum is wrapped.
- `wrap_to_pi(angle) -> (-pi, pi]`: new pub helper in `gnc/control/angle_utils.rs`. For
  `atan2_signed` / `acos_tanh` the result is already in range, so wrap is a no-op (apply
  it uniformly or gate to the two new decoders; either is fine -- prefer uniform for
  clarity, it is idempotent on in-range inputs).

`nn_bank_angle` gains a `prev_realized_bank: f64` parameter, threaded from dispatch.rs
alongside the existing telemetry scalars.

### 4. State plumbing

- New field `GuidanceState::prev_realized_bank_for_nn: f64`, initialized in
  `GuidanceState::new` and snapshotted post-guidance in `tick.rs` (mirrors
  `prev_bank_for_nn`; capture the previous value before it is overwritten by this tick's
  pilot output).
- This single field feeds **both** the `delta` decoder's base **and** the prev-realized
  `(sin, cos)` input slots, so the network input and the integrator base are consistent.
- `dispatch.rs` snapshots `state.prev_realized_bank_for_nn` into a local before the mut
  borrow of `nn_state` (same pattern as `prev_bank`, `integral`, etc.) and passes it to
  `nn_bank_angle` and to `build_nn_input`.

### 5. Warm-start target encoding (`training/warm_start.py`, PSO scope)

Both new decoders use a tanh head, so the loss reads `means[..., 0]` directly (mirrors
the existing `acos_tanh` branch at `warm_start.py:479`):

- `scaled_pi`: `target = clamp(y / (n * pi), -1, 1)`
- `delta`: `target = clamp(shortest_angle_diff(prev_realized, y) / delta_max, -1, 1)`

where `y` is the existing supervised teacher signal (`pre_shaper_signed`) and
`prev_realized` is the teacher trajectory's realized-bank series. The supervised-collect
path (`aerocapture_rs.collect_supervised`) must expose per-step realized bank for the
`delta` target (it is already implicitly present once prev-realized is a candidate input;
the plan decides whether to read it from the input slots via `atan2(sin, cos)` or to add
an explicit field to the collected dict -- explicit field is cleaner).

Config plumbed into the Python `NetworkConfig` (`scaled_pi_n`, `delta_max`). The deployed
`best_model.json` carries `output_param` + the knob value(s) so the Rust runtime stays
self-describing.

No structural `V2Policy` change: `forward_seq_means` returns pre-decode `means`; only the
loss-target dispatch in `_chunked_bptt_train` gains two branches.

### 6. PSO training configs

Two leaf configs inheriting `configs/training/nn_common.toml`, `full_neural`:

- `configs/training/msr_aller_nn_scaledpi_train.toml` (`output_parameterization =
  "scaled_pi"`, `scaled_pi_n = ...`, 1-output tanh head).
- `configs/training/msr_aller_nn_delta_train.toml` (`output_parameterization = "delta"`,
  `delta_max = ...`, 1-output tanh head).

Both use an `input_mask` that includes the new `(sin, cos)` pairs (and the prev-realized
pair). Registered as `neural_network_scaledpi_pso` and `neural_network_delta_pso` in
`compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES`, with `train_all.sh` aliases.

## Testing

Rust unit (`gnc/guidance/neural.rs`, `gnc/control/angle_utils.rs`, `data/neural.rs`):

- `scaled_pi` scales by `n * pi` (e.g. `n=2`, `out[0]=0.5` -> `bank = pi`, pre-wrap).
- `scaled_pi` wraps: `n=2`, `out[0]=1` -> `2pi` -> wrapped to `0` (or the boundary value).
- `delta` integrates on `prev_realized` and bounds the step to `+/- delta_max`.
- `delta` wraps across the seam (base near `pi`, positive delta -> wraps to `-pi+eps`).
- `wrap_to_pi` proptest: range `(-pi, pi]`, idempotence, `cos/sin` invariance vs input.
- Config/model validation: mode-compat hard errors for every decoder/mode pair;
  tanh-required for `scaled_pi`/`delta`; output_size==1 required; knob defaults.
- `(sin, cos)` inputs finite; `atan2(sin, cos)` round-trips the source angle.

Rust golden regression:

- All existing guidance golden files stay **bit-identical** (new decoders are opt-in;
  `NN_FULL_INPUT_SIZE` growth must not perturb default-mask `[0..16]` models -- verify the
  appended slots are never read under the default mask).

Python:

- Warm-start target-encoding branches for `scaled_pi` and `delta` (unit: known
  `y`/`prev_realized` -> expected clamped target).
- Config parse + validation (unknown-knob rejection, mode/decoder cross-check).

## Out of scope (deferred)

- `V2Policy` / PPO decoder mirror and PPO training configs for the new decoders.
- Cross-language equivalence tests for `scaled_pi` / `delta`. The decoders live in
  `nn_bank_angle` (post-`forward`), not in `nn.forward`, so the existing cross-language
  gate on `nn_forward` is unaffected. PSO deploys through the Rust runtime directly.
- Allowing `scaled_pi` / `delta` in `magnitude_only` (the seam they solve does not exist
  there). YAGNI.

## Risks / call-outs

- **Input-vector growth touches a shared builder.** `build_nn_input` is shared with
  supervised collection; the append-only layout + default-mask invariance keeps golden
  models bit-identical, but the plan must add a test asserting default-mask output is
  unchanged.
- **`delta` warm-start needs the teacher's realized-bank series.** Cleanest is an explicit
  field from `collect_supervised`; falling back to decoding the input `(sin, cos)` slots
  is possible but couples target computation to the mask layout.
- **Knob defaults are guesses.** `scaled_pi_n = 1.0` and `delta_max = 0.35 rad` are
  starting points; they are the experimental knobs the user wants and will be tuned.
