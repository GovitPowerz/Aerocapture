# Time-Varying Density Perturbations

**Date:** 2026-04-01
**Status:** Approved
**Ref:** IMPROVEMENTS.md Section 1.1

## Problem

Current MC density dispersions are static per-run: a single fractional multiplier is drawn at init and applied uniformly to every density lookup throughout the trajectory. Real atmospheric variability includes transient features (gravity waves, dust storm onset, diurnal cycles) that evolve during a pass. The guidance/navigation system is never tested against mid-trajectory density changes.

## Solution

Add a first-order Gauss-Markov (Ornstein-Uhlenbeck) process that produces a time-varying fractional density perturbation. The perturbation is stepped once per GNC tick using the exact OU transition and stacks multiplicatively on top of the existing static bias.

## Stochastic Model

Ornstein-Uhlenbeck scalar process x(t):

```
dx = -(1/tau) * x * dt + sigma_noise * dW
```

Exact transition between GNC ticks (dt = GNC period):

```
x(t+dt) = x(t) * exp(-dt/tau) + sigma * sqrt(1 - exp(-2*dt/tau)) * N(0,1)
```

Parameters:
- `tau` (seconds): correlation time. Controls decorrelation speed. Range 30-120s gives 3-15 decorrelation events per typical 300-600s Mars aerocapture trajectory.
- `sigma` (fractional): steady-state RMS amplitude. e.g., 0.10 = 10% RMS fluctuation.
- `x(0) = 0` always (static bias captures the initial offset).

Application to density:

```
rho_effective = rho_table(alt) * (1 + static_bias) * (1 + x(t))
```

The perturbation is time-only -- same multiplier at all altitudes within a given timestep. This is the simplest model that tests guidance robustness to transient density features.

## TOML Configuration

New optional section, independent of existing dispersion levels:

```toml
[monte_carlo.density_perturbation]
level = "medium"  # Off / Low / Medium / High / Custom
# Custom overrides (only used when level = "custom"):
# tau = 60.0        # correlation time (seconds)
# sigma = 0.15      # steady-state RMS amplitude (fractional)
```

### Preset Ladder

| Level  | tau (s) | sigma | Character |
|--------|---------|-------|-----------|
| Off    | --      | 0.0   | No perturbation (default, backward compatible) |
| Low    | 120.0   | 0.05  | Slow, gentle drift (~5% RMS) |
| Medium | 60.0    | 0.10  | Moderate fluctuations (~10% RMS) |
| High   | 30.0    | 0.20  | Fast, aggressive (~20% RMS) |
| Custom | user    | user  | Full control via tau/sigma fields |

### Rationale

- Tau 30-120s: short enough to stress-test guidance within a single pass, long enough to not be fully filtered by the navigation density estimator (exponential LPF with lambda ~0.8).
- Sigma 5-20%: stacks on top of static bias. At High static (+-100%) + High GM (20% RMS), worst-case instantaneous error ~2.4x nominal -- aggressive but not unphysical for Mars.
- Section absent = Off. Fully backward compatible.
- Not GA-optimizable. These are environment parameters, not guidance parameters.

## Rust Implementation

### New config struct

In `dispersions.rs` (or a new `density_perturbation.rs` if cleaner):

```rust
pub struct DensityPerturbationConfig {
    pub tau: f64,    // correlation time (s)
    pub sigma: f64,  // steady-state RMS (fractional)
}
```

With a `from_level()` method mirroring the existing `AtmosphereSigmas` pattern, plus TOML deserialization for custom values.

### New state in RunState

In `init.rs`:

```rust
pub density_perturbation: f64,  // current GM process value x(t), init 0.0
```

### GM step function

Pure, testable function:

```rust
pub fn step_density_perturbation(
    x: f64, dt: f64, tau: f64, sigma: f64, normal_sample: f64
) -> f64 {
    if sigma <= 0.0 || tau <= 0.0 {
        return 0.0;
    }
    let decay = (-dt / tau).exp();
    x * decay + sigma * (1.0 - (-2.0 * dt / tau).exp()).sqrt() * normal_sample
}
```

Called once per integration tick (the outer loop period, typically 0.1s) in the simulation loop, before density lookups. The dt passed to the step function is the integration period.

### Centralized density function

Modify `physics::atmosphere::density()` to accept both biases:

```rust
pub fn density(
    atm: &AtmosphereModel, altitude: f64,
    density_bias: f64, density_perturbation: f64
) -> f64 {
    atm.density_at(altitude) * (1.0 + density_bias) * (1.0 + density_perturbation)
}
```

All call sites pass the extra arg. Physics is centralized in one place.

### Per-run RNG

Dedicated `StdRng` seeded from the run seed with an offset to avoid correlation:

```rust
let gm_rng = StdRng::seed_from_u64(run_seed.wrapping_add(0xDENS));
```

Lives in `RunState`, stepped once per GNC tick. Independent from dispersion draw RNG. Each Rayon thread has its own -- no contention.

## Output & Observability

- **Trajectory output**: Add `density_perturbation` as column 17 (index 16) in the per-timestep trajectory data. The existing `truth_density_kg_m3` column already captures net dispersed density; the raw GM value lets post-analysis separate static vs dynamic contributions.
- **DispersionDraw / final record**: No change. Static bias is already recorded. GM is time-varying and doesn't belong in per-run summary.
- **Charts**: No changes required. Existing density ratio vs time chart naturally shows GM fluctuations. Dedicated visualization can be added later.

## Testing Strategy

### Unit tests (Rust, inline #[cfg(test)])

- `step_density_perturbation` with sigma=0 returns 0 (disabled invariant)
- `step_density_perturbation` with tau=0 returns 0 (degenerate case)
- Statistical: run N steps, verify mean ~0 and std ~sigma within tolerance
- Determinism: same seed + same dt/tau/sigma sequence = same output
- Proptest: output is always finite for reasonable input ranges

### Integration tests (Rust)

- Config parsing: `[monte_carlo.density_perturbation]` with level presets and custom values
- Backward compatibility: no section = zero perturbation, bit-identical to current behavior
- E2E: short sim with GM enabled, trajectory completes, density_perturbation column non-zero

### Python tests

- PyO3: trajectory column count = 17, new column accessible
- Config round-trip: `load_config()` with density_perturbation section parses correctly

## Non-goals

- Altitude-correlated perturbations (gravity wave spatial structure). Can be layered later.
- Multi-scale models (slow + fast processes). Can be added by composing two OU processes.
- GA optimization of tau/sigma. These are environment parameters.
- Chart changes. Existing plots suffice for now.
