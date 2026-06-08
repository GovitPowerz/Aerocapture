# DV-inferred Reward for Dense PPO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable RL reward potential `Phi = -(w1·dv1 + w2·dv2 + w3·dv3) - constraint·(hf² + hl²)` built from the per-tick predicted correction delta-v, and a dense-PPO atan2 training config that uses it.

**Architecture:** The raw DV components (`predicted_dv_for_nn`, m/s) are surfaced through the PyO3 env's aux channel (grown `(N,2)→(N,5)`). A new `potential="dv"` mode in `StepRewardCalculator` reads them. The phase-aware potential and all existing configs stay bit-identical (non-destructive). PBRS (Ng-Harada-Russell) preserves the optimum; `Phi ≈ -V*` densifies the gradient toward the true objective.

**Tech Stack:** Rust (PyO3 / `aerocapture-py`), Python (numpy, dataclass config), pytest. Spec: `docs/superpowers/specs/2026-06-08-rl-ppo-dv-reward-design.md`.

---

## File Structure

- **Modify** `src/rust/aerocapture-py/src/env.rs` — extend aux `(N,2)→(N,5)` with raw `[dv1,dv2,dv3]`; add `predicted_dv_for_state` helper; new imports.
- **Modify** `tests/test_env_pyo3.py` — aux shape `(4,2)→(4,5)`; add a DV-column behavioral test.
- **Modify** `src/python/aerocapture/training/rl/rewards.py` — `potential` mode dispatch, `_potential_dv`, relaxed required-index validation.
- **Modify** `tests/rl/test_rewards.py` — DV-mode unit tests.
- **Modify** `src/python/aerocapture/training/rl/config.py` — `RewardConfig` gains `potential`, `dv1/2/3_weight`.
- **Modify** `src/python/aerocapture/training/rl/train.py` — `_build_shaper_and_norms` threads the new fields.
- **Create** `tests/rl/test_reward_config.py` — config parse + atan2-config-load tests.
- **Modify** `configs/training/msr_aller_nn_atan2_ppo_train.toml` — repurpose into a working RL config.
- **Create** `tests/test_atan2_rl_ppo_smoke.py` — end-to-end PPO smoke (@slow).
- **Modify** `src/python/aerocapture/training/compare_guidance.py` — register `neural_network_atan2_rl`.

---

## Task 1: Rust — raw DV into the aux channel `(N,5)`

**Files:**
- Modify: `tests/test_env_pyo3.py` (shape asserts + new test)
- Modify: `src/rust/aerocapture-py/src/env.rs:17-26` (imports), `:193` (outcomes type), `:208-210` (closure aux), `:268-277` (step aux array), `:326-336` (build_aux); add helper after `:370`

- [ ] **Step 1: Update the failing env tests**

In `tests/test_env_pyo3.py`, change the two aux shape assertions from `(4, 2)` to `(4, 5)`:

`test_batched_simulation_reset_shape` (currently line 25):
```python
    assert aux.shape == (4, 5)
```

`test_step_advances_and_returns_correct_shapes` (currently line 64):
```python
    assert aux.shape == (4, 5)
```

Append a new behavioral test at the end of the file:
```python
def test_aux_carries_dv_components() -> None:
    """Aux columns 2-4 are the raw predicted-DV correction budget (finite, live)."""
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4, seed_base=3_000_000)
    _, aux = env.reset()
    assert aux.shape == (4, 5)
    assert np.isfinite(aux).all()
    seen_nonzero = np.zeros(3, dtype=bool)
    for _ in range(50):
        _, _, _, _, aux = env.step(np.zeros(4, dtype=np.float32))
        assert aux.shape == (4, 5)
        assert np.isfinite(aux).all()
        seen_nonzero |= np.abs(aux[:, 2:5]).max(axis=0) > 0.0
    assert seen_nonzero.any(), "predicted-DV aux columns never became nonzero"
    env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env_pyo3.py::test_batched_simulation_reset_shape tests/test_env_pyo3.py::test_aux_carries_dv_components -v`
Expected: FAIL — current aux is `(4, 2)` (shape assert fails; new test fails on shape).

- [ ] **Step 3: Add imports to `env.rs`**

After the existing `use aerocapture::...` block (the `use aerocapture::simulation::runner::{...};` ending at line 26), add:
```rust
use aerocapture::orbit::{elements, maneuver};
```

- [ ] **Step 4: Add the `predicted_dv_for_state` helper**

After the `build_obs_for_env` function (ends at line ~370, just before `struct TerminalOutcome`), add:
```rust
/// Predicted correction delta-v [dv1, dv2, dv3] (raw m/s) on the current
/// osculating orbit for one env. Mirrors `build_obs_for_env`'s orbit
/// construction so the aux DV equals candidate inputs 32-34 that
/// `build_nn_input` produces (pre-normalization). Consumed by the DV-reward
/// potential (`StepRewardCalculator`, potential = "dv").
fn predicted_dv_for_state(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> [f64; 3] {
    let nav = state.last_nav_output();
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &config.planet,
    );
    maneuver::predicted_dv_for_nn(&orbit, &data.target_orbit, &data.parking_orbit, &config.planet)
}
```

- [ ] **Step 5: Grow the `outcomes` tuple aux to `[f64; 5]`**

At line 193, change the `outcomes` type annotation:
```rust
        let outcomes: Vec<(bool, Option<TerminalOutcome>, [f64; 5])> = py.detach(|| {
```

Inside the Rayon closure, replace the aux capture (currently lines 208-210):
```rust
                    // Capture aux (energy, pdyn, dv1, dv2, dv3) from nav output
                    // before potential reset. The 3 DV components are the raw m/s
                    // correction-budget signals the DV-reward potential consumes.
                    let nav = state.last_nav_output();
                    let dv = predicted_dv_for_state(state, sim_data, sim_input);
                    let aux = [
                        nav.energy_estimated,
                        nav.dynamic_pressure_estimated,
                        dv[0],
                        dv[1],
                        dv[2],
                    ];
```

(The `(true, Some(term), aux)` / `(false, None, aux)` arms and the `for (i, (done, _, _)) in outcomes.iter()` reset loop are unchanged — they ignore the aux element or use it positionally.)

- [ ] **Step 6: Grow the step() aux array to 5 columns**

Replace the step() aux-array block (currently lines 268-277):
```rust
        // Aux array: (n_envs, 5) with [energy, pdyn, dv1, dv2, dv3].
        // Values are from the pre-reset nav output (terminal steps get their final-tick values).
        let aux = PyArray2::<f32>::zeros(py, [self.n_envs, 5], false);
        {
            let mut aux_view = unsafe { aux.as_array_mut() };
            for (i, (_, _, a)) in outcomes.iter().enumerate() {
                for j in 0..5 {
                    aux_view[[i, j]] = a[j] as f32;
                }
            }
        }
```

- [ ] **Step 7: Grow `build_aux` to 5 columns**

Replace the entire `build_aux` method (currently lines 326-336):
```rust
    /// Auxiliary array (n_envs, 5): [energy_estimated, dynamic_pressure_estimated,
    /// predicted_dv1, predicted_dv2, predicted_dv3] per env. The 3 DV components are
    /// the raw m/s correction-budget estimate consumed by the DV-reward potential.
    fn build_aux<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let arr = PyArray2::<f32>::zeros(py, [self.n_envs, 5], false);
        let mut view = unsafe { arr.as_array_mut() };
        for (i, env) in self.envs.iter().enumerate() {
            let nav = env.last_nav_output();
            let dv = predicted_dv_for_state(env, &self.sim_data, &self.sim_input);
            view[[i, 0]] = nav.energy_estimated as f32;
            view[[i, 1]] = nav.dynamic_pressure_estimated as f32;
            view[[i, 2]] = dv[0] as f32;
            view[[i, 3]] = dv[1] as f32;
            view[[i, 4]] = dv[2] as f32;
        }
        arr
    }
```

- [ ] **Step 8: Format, lint, rebuild the bindings**

Run (from repo root):
```bash
cargo fmt --manifest-path src/rust/aerocapture-py/Cargo.toml
cargo clippy --manifest-path src/rust/aerocapture-py/Cargo.toml -- -D warnings
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```
Expected: fmt clean, clippy no warnings, maturin builds and installs `aerocapture_rs`.

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_env_pyo3.py -v`
Expected: PASS (all, including the new `test_aux_carries_dv_components`).

- [ ] **Step 10: Commit**

```bash
git add src/rust/aerocapture-py/src/env.rs tests/test_env_pyo3.py
git commit -m "feat(rl): surface raw predicted-DV in the env aux channel (N,5)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Python — DV reward potential mode

**Files:**
- Modify: `src/python/aerocapture/training/rl/rewards.py:28-95`
- Modify: `tests/rl/test_rewards.py`

- [ ] **Step 1: Write the failing DV-mode tests**

Append to `tests/rl/test_rewards.py`:
```python
def _dv_calc(**kw: float) -> StepRewardCalculator:
    return StepRewardCalculator(input_mask=list(range(23)), potential="dv", constraint_weight=0.2, gamma=0.99, **kw)


def _aux5(n: int = 1, dv1: float = 0.0, dv2: float = 0.0, dv3: float = 0.0) -> np.ndarray:
    aux = np.zeros((n, 5), dtype=np.float32)
    aux[:, 2] = dv1
    aux[:, 3] = dv2
    aux[:, 4] = dv3
    return aux


def test_dv_potential_value() -> None:
    calc = _dv_calc(dv1_weight=1.0, dv2_weight=1.0, dv3_weight=1.0)
    obs = _make_obs(n=1)  # hf_frac = hl_frac = 0
    phi = calc._potential(obs, _aux5(dv1=100.0, dv2=20.0, dv3=5.0))
    assert np.isclose(phi[0], -(100.0 + 20.0 + 5.0), atol=1e-6)


def test_dv_potential_weights_linear() -> None:
    calc = _dv_calc(dv1_weight=1.0, dv2_weight=2.0, dv3_weight=0.0)
    obs = _make_obs(n=1)
    phi = calc._potential(obs, _aux5(dv1=10.0, dv2=10.0, dv3=10.0))
    assert np.isclose(phi[0], -(10.0 + 20.0 + 0.0), atol=1e-6)


def test_dv_potential_keeps_thermal_term() -> None:
    calc = _dv_calc()
    obs = _make_obs(n=1, **{"6": 1.0, "7": 1.0})  # hf_frac = hl_frac = 1
    phi = calc._potential(obs, _aux5())  # dv = 0
    assert np.isclose(phi[0], -0.2 * (1.0 + 1.0), atol=1e-6)


def test_dv_reward_positive_when_dv_decreases() -> None:
    calc = _dv_calc()
    obs = _make_obs(n=1)
    r = calc.step_reward(obs, obs, _aux5(dv1=200.0), _aux5(dv1=100.0))
    # gamma*Phi(next) - Phi(cur) = 0.99*(-100) - (-200) = 101 > 0
    assert r[0] > 0


def test_dv_mode_relaxes_required_indices() -> None:
    # Only the thermal pair (6, 7) is required in dv mode; 13/15/19/0 not needed.
    StepRewardCalculator(input_mask=[6, 7], potential="dv")


def test_dv_mode_missing_thermal_raises() -> None:
    with pytest.raises(ValueError, match="missing required indices"):
        StepRewardCalculator(input_mask=[0, 1, 2], potential="dv")


def test_invalid_potential_raises() -> None:
    with pytest.raises(ValueError, match="potential must be"):
        StepRewardCalculator(input_mask=list(range(23)), potential="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rl/test_rewards.py -k "dv or invalid_potential" -v`
Expected: FAIL — `StepRewardCalculator` has no `potential` keyword (TypeError).

- [ ] **Step 3: Implement the DV mode in `rewards.py`**

Add the new dataclass fields (after `energy_scale`, before `cost_kwargs`):
```python
    energy_scale: float = 1.0e6
    potential: str = "phase_aware"
    dv1_weight: float = 1.0
    dv2_weight: float = 1.0
    dv3_weight: float = 1.0
    cost_kwargs: dict = field(default_factory=dict)
```

Replace `__post_init__`:
```python
    def __post_init__(self) -> None:
        if self.potential not in ("phase_aware", "dv"):
            raise ValueError(f"potential must be 'phase_aware' or 'dv', got {self.potential!r}")
        self._rev: dict[int, int] = {v: i for i, v in enumerate(self.input_mask)}
        if self.potential == "dv":
            # DV potential is phase-agnostic; only the thermal-proximity pair is read from obs.
            required = [_IDX_HEAT_FLUX_FRAC, _IDX_HEAT_LOAD_FRAC]
        else:
            required = [_IDX_ECC_EXCESS, _IDX_HEAT_FLUX_FRAC, _IDX_HEAT_LOAD_FRAC, _IDX_SMA_ERROR, _IDX_BOUNCE_FLAG, _IDX_PDYN_ERROR]
        missing = [r for r in required if r not in self._rev]
        if missing:
            raise ValueError(f"input_mask missing required indices: {missing}")
```

Rename the existing `_potential` method to `_potential_phase_aware` (change only the `def` line; body unchanged):
```python
    def _potential_phase_aware(
        self,
        obs: npt.NDArray[np.float32],
        aux: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
```

Add the dispatcher and the DV potential immediately after `_potential_phase_aware`:
```python
    def _potential(
        self,
        obs: npt.NDArray[np.float32],
        aux: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        if self.potential == "dv":
            return self._potential_dv(obs, aux)
        return self._potential_phase_aware(obs, aux)

    def _potential_dv(
        self,
        obs: npt.NDArray[np.float32],
        aux: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        """DV-correction potential: Phi = -(w·dv) - constraint·(hf² + hl²).

        dv1/dv2/dv3 are the raw m/s correction-budget components from aux[:, 2:5]
        (predicted_dv_for_nn). Not phase-gated -- the DV signal is smooth across
        the bounce. The thermal-proximity term is retained (DV is blind to heat
        limits, and the terminal penalty alone is a sparse teacher).
        """
        hf_frac = (obs[:, self._col(_IDX_HEAT_FLUX_FRAC)].astype(np.float64) + 1.0) / 2.0
        hl_frac = (obs[:, self._col(_IDX_HEAT_LOAD_FRAC)].astype(np.float64) + 1.0) / 2.0
        dv1 = aux[:, 2].astype(np.float64)
        dv2 = aux[:, 3].astype(np.float64)
        dv3 = aux[:, 4].astype(np.float64)
        dv_term = self.dv1_weight * dv1 + self.dv2_weight * dv2 + self.dv3_weight * dv3
        return -dv_term - self.constraint_weight * (hf_frac**2 + hl_frac**2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rl/test_rewards.py -v`
Expected: PASS (new DV tests + all existing phase-aware tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/rewards.py tests/rl/test_rewards.py
git commit -m "feat(rl): add DV-correction reward potential mode" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Config wiring — `RewardConfig` + `_build_shaper_and_norms`

**Files:**
- Modify: `src/python/aerocapture/training/rl/config.py:18-32`
- Modify: `src/python/aerocapture/training/rl/train.py:177-187`
- Create: `tests/rl/test_reward_config.py`

- [ ] **Step 1: Write the failing config-parse test**

Create `tests/rl/test_reward_config.py`:
```python
"""[rl.reward] parsing + atan2 RL config load."""

from __future__ import annotations

from pathlib import Path

import tomli_w
from aerocapture.training.rl.config import RLConfig


def test_reward_config_parses_dv_fields(tmp_path: Path) -> None:
    p = tmp_path / "rl.toml"
    p.write_bytes(tomli_w.dumps({"rl": {"algorithm": "ppo", "reward": {"potential": "dv", "dv2_weight": 2.0}}}).encode())
    cfg = RLConfig.from_toml(p)
    assert cfg.reward.potential == "dv"
    assert cfg.reward.dv2_weight == 2.0
    assert cfg.reward.dv1_weight == 1.0  # default
    assert cfg.reward.dv3_weight == 1.0  # default


def test_reward_config_default_potential_is_phase_aware(tmp_path: Path) -> None:
    p = tmp_path / "rl.toml"
    p.write_bytes(tomli_w.dumps({"rl": {"algorithm": "ppo"}}).encode())
    cfg = RLConfig.from_toml(p)
    assert cfg.reward.potential == "phase_aware"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_reward_config.py::test_reward_config_parses_dv_fields -v`
Expected: FAIL — `RewardConfig.__init__() got an unexpected keyword argument 'potential'`.

- [ ] **Step 3: Add fields to `RewardConfig`**

In `config.py`, add to the `RewardConfig` dataclass after `energy_scale` (line 27):
```python
    energy_scale: float = 1.0e6
    # DV-correction potential (potential = "dv"); ignored when potential = "phase_aware".
    potential: str = "phase_aware"
    dv1_weight: float = 1.0
    dv2_weight: float = 1.0
    dv3_weight: float = 1.0
```

- [ ] **Step 4: Thread the fields in `_build_shaper_and_norms`**

In `train.py`, in the `StepRewardCalculator(...)` construction (lines 177-187), add the new kwargs (after `energy_scale=...`):
```python
        energy_scale=cfg.reward.energy_scale,
        potential=cfg.reward.potential,
        dv1_weight=cfg.reward.dv1_weight,
        dv2_weight=cfg.reward.dv2_weight,
        dv3_weight=cfg.reward.dv3_weight,
        cost_kwargs=read_cost_kwargs(toml_path),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_reward_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/config.py src/python/aerocapture/training/rl/train.py tests/rl/test_reward_config.py
git commit -m "feat(rl): parse [rl.reward] dv knobs and thread into the shaper" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Training config — repurpose `msr_aller_nn_atan2_ppo_train.toml`

**Files:**
- Modify: `configs/training/msr_aller_nn_atan2_ppo_train.toml` (full rewrite)
- Modify: `tests/rl/test_reward_config.py` (add load test)

- [ ] **Step 1: Write the failing config-load test**

Append to `tests/rl/test_reward_config.py`:
```python
def test_atan2_rl_config_loads() -> None:
    from aerocapture.training.rl.train import _parse_network_config

    cfg = RLConfig.from_toml(Path("configs/training/msr_aller_nn_atan2_ppo_train.toml"))
    assert cfg.algorithm == "ppo"
    assert cfg.reward.potential == "dv"
    assert cfg.reward.dv1_weight == 1.0
    assert cfg.reward.dv2_weight == 1.0
    assert cfg.reward.dv3_weight == 1.0
    # n_envs / steps inherited from rl_common.toml
    assert cfg.n_envs == 64
    input_mask, architecture, input_dim = _parse_network_config(cfg)
    assert len(input_mask) == 17
    assert input_dim == 17
    assert {32, 33, 34}.issubset(set(input_mask))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_reward_config.py::test_atan2_rl_config_loads -v`
Expected: FAIL — current config has no `[rl]` section (so `cfg.reward.potential == "phase_aware"`, not `"dv"`; and `cfg.algorithm` defaults but `n_envs` would be the 64 default — the `potential` assert fails first).

- [ ] **Step 3: Rewrite the config**

Replace the entire contents of `configs/training/msr_aller_nn_atan2_ppo_train.toml`:
```toml
# MSR outbound -- dense PPO with DV-inferred reward.
# Phi = -(w1*dv1 + w2*dv2 + w3*dv3) - constraint*(hf^2 + hl^2), DV from
# predicted_dv_for_nn (candidate inputs 32-34) surfaced via the env aux channel.
# See docs/superpowers/specs/2026-06-08-rl-ppo-dv-reward-design.md.
base = ["../missions/mars.toml", "common.toml", "rl_common.toml"]

[guidance]
type = "neural_network"

[guidance.neural_network]
mode = "full_neural"
output_parameterization = "atan2_signed"

[network]
# 17 inputs: 13-base + hdot_nominal(18) + pdyn_error(19) + seam-free bank-history
# (sin,cos) pairs (27-30) + live correction-DV predicted_dv1/2/3 (32,33,34).
# predicted_dv* = per-tick predicted_dv_for_nn on the current osculating orbit:
# smooth across e=1, no sentinel. All inputs renormalized data-driven
# (calibrate_inputs.py): asinh on heavy-tailed, calibrated affine on bounded.
input_mask = [0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30, 32, 33, 34]

# Per-input normalization override (MUST be under [network] and BEFORE the
# [[network.architecture]] blocks -- a bare key attaches to the preceding table).
normalization = [
    { transform = "none", scale = 0.8655390346546353, center = 0.9204641308303162 }, # 0 eccentricity_excess
    { transform = "asinh", scale = 0.8175600848274288, center = 0.8867681772014051 }, # 1 inclination_error
    { transform = "asinh", scale = 462.6300273768903, center = -333.1697854608212 }, # 2 radial_velocity
    { transform = "none", scale = 5413222.0902614985, center = -508770.60939412704 }, # 3 orbital_energy
    { transform = "none", scale = 1157.3507227945558, center = 4549.679429911598 }, # 4 velocity
    { transform = "asinh", scale = 10.358129862795693, center = 12.17320223341297 }, # 5 accel_magnitude
    { transform = "asinh", scale = 0.33924629426269115, center = 0.40005844701414567 }, # 6 heat_flux_fraction
    { transform = "none", scale = 0.42411593596120784, center = 0.42858041580763234 }, # 7 heat_load_fraction
    { transform = "asinh", scale = 33.09832239288295, center = 82.65759631477488 }, # 8 altitude
    { transform = "asinh", scale = 0.0911362193061205, center = -0.04718278665783951 }, # 9 fpa
    { transform = "asinh", scale = 0.2151370816795333, center = 0.29274513397808116 }, # 10 latitude
    { transform = "asinh", scale = 9.832939972853932, center = 11.55598244317186 }, # 11 drag_accel
    { transform = "asinh", scale = 3.256400734674897, center = 3.827025266270693 }, # 12 lift_accel
    { transform = "asinh", scale = 5107742.117847206, center = -3590633.2113008387 }, # 13 sma_error
    { transform = "asinh", scale = 10217099.16686349, center = -6706141.091617538 }, # 14 apoapsis_alt
    { transform = "none", scale = 0.5, center = 0.5 }, # 15 bounce_flag
    { transform = "none", scale = 1e-06, center = 0.4262529995153208 }, # 16 cos_bank_nominal
    { transform = "asinh", scale = 594.4827404994671, center = 716.3429422450554 }, # 17 pdyn_nominal
    { transform = "asinh", scale = 412.89140647256187, center = -302.7939124482325 }, # 18 hdot_nominal
    { transform = "asinh", scale = 154.12675819494976, center = -169.7611160606572 }, # 19 pdyn_error
    { transform = "none", scale = 1.5707963267948966, center = 1.5707963267948966 }, # 20 exit_bank_teacher
    { transform = "none", scale = 0.1, center = 0.0 }, # 21 inclination_err_rate
    { transform = "none", scale = 3.141592653589793, center = 0.0 }, # 22 prev_bank_signed
    { transform = "tanh", scale = 30.0, center = 0.0 }, # 23 time_since_sign_flip
    { transform = "tanh", scale = 100.0, center = 0.0 }, # 24 inclination_err_integral
    { transform = "none", scale = 1.0, center = 0.0 }, # 25 exit_bank_teacher_sin
    { transform = "none", scale = 1.0, center = 0.0 }, # 26 exit_bank_teacher_cos
    { transform = "none", scale = 1.0, center = 0.0 }, # 27 prev_bank_signed_sin
    { transform = "none", scale = 1.0, center = 0.0 }, # 28 prev_bank_signed_cos
    { transform = "none", scale = 1.0, center = 0.0 }, # 29 prev_realized_sin
    { transform = "none", scale = 1.0, center = 0.0 }, # 30 prev_realized_cos
    { transform = "asinh", scale = 17774.23743300369, center = 20521.130577512882 }, # 31 periapsis_alt
    { transform = "asinh", scale = 500.0, center = 500.0 }, # 32 predicted_dv1
    { transform = "none", scale = 60.0, center = 60.0 }, # 33 predicted_dv2
    { transform = "none", scale = 100.0, center = 0.0 }, # 34 predicted_dv3
]

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 24
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 24
output_size = 12
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 12
output_size = 2
activation = "asinh"

[rl.reward]
# DV-inferred potential. Replaces the orbital-tracking proxy terms; keeps the
# thermal-proximity term and the terminal compute_cost anchor.
potential   = "dv"
dv1_weight  = 1.0
dv2_weight  = 1.0
dv3_weight  = 1.0
constraint_weight = 0.2

[data]
neural_network = "training_output/neural_network_atan2_rl/best_model.json"
results_suffix = ".train_nn_atan2_rl"

# Command shaper and navigation run for ALL guidance schemes (dispatch-layer and
# upstream of guidance respectively), so these are live under full_neural.
[guidance.command_shaping]
enabled = true
max_bank_acceleration = 12.674110367187794   # deg/s^2

[navigation]
density_filter_gain = 0.34888052188696395
density_gain_max_delta = 0.13961756994200195
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rl/test_reward_config.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add configs/training/msr_aller_nn_atan2_ppo_train.toml tests/rl/test_reward_config.py
git commit -m "feat(rl): dense-PPO atan2 config with DV-inferred reward" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: End-to-end PPO smoke test

**Files:**
- Create: `tests/test_atan2_rl_ppo_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_atan2_rl_ppo_smoke.py`:
```python
"""5-update dense-PPO smoke on the atan2 DV-reward config. Exercises the full
path: TOML parse, V2Policy + atan2 head, rollout with (N,5) DV aux, DV-reward
potential, BPTT update, validation, v2 JSON export, Rust nn_forward consumes it.

Runs in the python-pyo3 CI job (bindings required). Not a convergence test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_atan2_rl_ppo_smoke_5_updates(tmp_path: Path) -> None:
    import tomli_w
    from aerocapture.training.rl.config import RLConfig
    from aerocapture.training.rl.display import make_display
    from aerocapture.training.rl.logger import RLLogger
    from aerocapture.training.rl.train import _generate_seed_model, _run_ppo
    from aerocapture.training.toml_utils import load_toml_with_bases

    resolved = load_toml_with_bases(Path("configs/training/msr_aller_nn_atan2_ppo_train.toml"))

    # Shrink RL dimensions for CI. n_envs=4 * rollout_steps=64 * 5 updates = 1280 steps.
    rl_section: dict[str, Any] = resolved.setdefault("rl", {})
    rl_section["n_envs"] = 4
    rl_section["total_env_steps"] = 4 * 64 * 5
    rl_section["validation_n_sims"] = 4
    rl_section["validation_interval_updates"] = 5
    rl_section["checkpoint_interval_updates"] = 5

    ppo_section: dict[str, Any] = rl_section.setdefault("ppo", {})
    ppo_section["rollout_steps"] = 64
    ppo_section["bptt_length"] = 64  # dense: one chunk
    ppo_section["update_epochs"] = 2
    ppo_section["minibatches"] = 2

    data_section: dict[str, Any] = resolved.setdefault("data", {})
    seed_model_path = tmp_path / "seed_model.json"
    data_section["neural_network"] = str(seed_model_path)

    resolved.pop("base", None)
    smoke_toml = tmp_path / "smoke.toml"
    smoke_toml.write_bytes(tomli_w.dumps(resolved).encode())

    output_dir = tmp_path / "neural_network_atan2_rl_smoke"
    output_dir.mkdir()

    cfg = RLConfig.from_toml(smoke_toml)
    assert cfg.reward.potential == "dv"

    _generate_seed_model(cfg, seed_model_path)
    env_overrides = {"data.neural_network": str(seed_model_path)}

    logger = RLLogger(output_dir, config_hash="smoke")
    display = make_display(cfg.total_env_steps, enabled=False)
    interrupted = {"v": False}

    try:
        _run_ppo(cfg, smoke_toml, output_dir, logger, display, interrupted, None, env_overrides, None)
    finally:
        display.close()
        logger.close()

    best_model = output_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {output_dir}"

    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    layer_types = [entry["type"] for entry in raw["architecture"]]
    assert layer_types == ["dense", "dense", "dense"], f"unexpected arch: {layer_types}"

    # nn_forward applies input_mask internally, so it needs the FULL candidate
    # vector: expected_len = max(input_mask) + 1 = 34 + 1 = 35 (lib.rs:416-417).
    output = aerocapture_rs.nn_forward(str(best_model), [0.0] * 35)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/test_atan2_rl_ppo_smoke.py -v -m slow`
Expected: PASS (~60-120s). If `best_model.json` is not promoted within 5 updates (validation may reject), the test fails on `best_model.exists()` — in that case raise `validation_interval_updates` coverage by setting `rl_section["validation_interval_updates"] = 1` so a validation runs every update, guaranteeing at least one promotion attempt against the random seed model.

- [ ] **Step 3: Commit**

```bash
git add tests/test_atan2_rl_ppo_smoke.py
git commit -m "test(rl): end-to-end dense-PPO atan2 DV-reward smoke" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Register `neural_network_atan2_rl` in compare_guidance

**Files:**
- Modify: `src/python/aerocapture/training/compare_guidance.py:31-51`, `:56-76`, `:80-87`
- Create: `tests/test_atan2_rl_scheme_registered.py`

- [ ] **Step 1: Write the failing registration test**

Create `tests/test_atan2_rl_scheme_registered.py`:
```python
"""neural_network_atan2_rl is wired into compare_guidance."""

from __future__ import annotations


def test_atan2_rl_scheme_registered() -> None:
    from aerocapture.training import compare_guidance as cg

    assert "neural_network_atan2_rl" in cg.SCHEMES
    assert cg.SCHEME_TRAINING_CONFIGS["neural_network_atan2_rl"] == "configs/training/msr_aller_nn_atan2_ppo_train.toml"
    assert "neural_network_atan2_rl" in cg._NN_DEPLOY_SCHEMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_atan2_rl_scheme_registered.py -v`
Expected: FAIL — `"neural_network_atan2_rl"` not in `SCHEMES`.

- [ ] **Step 3: Add to `SCHEMES`**

In `compare_guidance.py`, insert after the `"neural_network_rl",` line in the `SCHEMES` list (line 38):
```python
    "neural_network_rl",
    "neural_network_atan2_rl",
```

- [ ] **Step 4: Add to `SCHEME_TRAINING_CONFIGS`**

Insert after the `"neural_network_rl": ...` entry (line 63):
```python
    "neural_network_rl": "configs/training/msr_aller_rl_train.toml",
    "neural_network_atan2_rl": "configs/training/msr_aller_nn_atan2_ppo_train.toml",
```

- [ ] **Step 5: Add to `_NN_DEPLOY_SCHEMES`**

Insert after the `"neural_network_rl",` entry (line 82):
```python
    "neural_network_rl",
    "neural_network_atan2_rl",
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_atan2_rl_scheme_registered.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/compare_guidance.py tests/test_atan2_rl_scheme_registered.py
git commit -m "feat(rl): register neural_network_atan2_rl scheme" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full verification + smart-commit

- [ ] **Step 1: Lint the Python**

Run: `./lint_code.sh`
Expected: ruff (imports, format, lint) + mypy all pass. Fix any issues in the files this plan touched.

- [ ] **Step 2: Run the relevant Python test suites**

Run:
```bash
uv run pytest tests/rl/test_rewards.py tests/rl/test_reward_config.py tests/test_env_pyo3.py tests/test_atan2_rl_scheme_registered.py -v
uv run pytest tests/test_atan2_rl_ppo_smoke.py -v -m slow
```
Expected: all PASS.

- [ ] **Step 3: Rust check**

Run: `./check_all.sh`
Expected: Rust test + fmt --check + clippy + release build all pass (confirms the `env.rs` change is clean workspace-wide and the guidance goldens are unaffected — this change touches only the PyO3 aux channel, not physics).

- [ ] **Step 4: smart-commit over the whole branch**

Invoke the `smart-commit` skill, instructing it to take the entire `feature/rl-ppo-dv-reward` branch into account (sync CLAUDE.md / README docs for the new `[rl.reward] potential = "dv"` knob, the `(N,5)` aux channel, and the `neural_network_atan2_rl` scheme, then commit anything outstanding).

---

## Self-Review Notes

- **Spec coverage:** aux `(N,5)` (Task 1), DV potential + thermal term + weighted sum (Task 2), selectable mode default `phase_aware` (Tasks 2-3), config repurpose with 17-input atan2 mask (Task 4), end-to-end smoke (Task 5), scheme registration (Task 6, spec "optional" — included), final smart-commit (Task 7). Terminal cost unchanged — no task needed (existing `train.py:547` path untouched).
- **DV-from-nav-estimate** and **smoothness across e=1** (spec risks) are implicitly exercised by `test_aux_carries_dv_components` running real trajectories through capture/bounce and asserting finiteness every step.
- **Type/name consistency:** `predicted_dv_for_state(state, data, config)` defined once (Task 1 Step 4) and called in the closure (Step 5) and `build_aux` (Step 7). `potential` field name consistent across `rewards.py`, `config.py`, the TOML `[rl.reward]`, and tests.
- **SAC:** shares `_build_shaper_and_norms` and the env aux, so DV mode works for SAC for free; not separately tested (out of scope — dense PPO is the focus).
