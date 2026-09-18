"""Differentiable PyTorch mirror of the Rust NN runtime.

One torch module per cell type (`layers/`), the `LayerSpec` discriminated union
that validates every v2 `[[network.architecture]]` and sizes every NN chromosome
(`schemas.py`), the step-wise stateful `V2Policy` (`policy.py`), and the v2 JSON
writer `export_v2_policy_to_json` (`export.py`; the reader is
`aerocapture.training.model_io`). Held to numerical agreement with the native
runtime by the `tests/test_*_equivalence.py` gates. Consumed by the shipped
population-training path (encoding, config, warm-start); depends on nothing in
the PPO/SAC trainer package `rl/`, which depends on it.
"""
