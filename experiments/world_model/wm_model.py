"""Action-conditioned probabilistic dynamics models of the plant (#113): a GRU and a one-step MLP ablation.

State x_k (42) = the 35 candidate inputs of step k + the 7 aux channels (energy MJ/kg, pdyn kPa,
asinh(dv/100 m/s) x3, heat flux / load fractions). The action fed with x_k is a_{k+1} as
(sin, cos) (the timing contract in wm_plant). Both models output a diagonal Gaussian over the
standardized increment x_{k+1} - x_k and train on its negative log-likelihood, teacher forced.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_X = 42
N_A = 2
HIDDEN = 128
HEAD = 256
LOG_STD_MIN, LOG_STD_MAX = -7.0, 2.0
BATCH_FLIGHTS = 64
LR = 1e-3  # Adam, cosine-annealed to LR / 10 over the run


def features(obs: np.ndarray, aux: np.ndarray) -> np.ndarray:
    a = aux.astype(np.float64)
    extra = np.concatenate([a[..., :1] / 1e6, a[..., 1:2] / 1e3, np.arcsinh(a[..., 2:5] / 100.0), a[..., 5:7]], axis=-1)
    return np.concatenate([obs, extra], axis=-1).astype(np.float32)


def action_features(bank: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(bank), np.cos(bank)], axis=-1).astype(np.float32)


@dataclass
class Sequences:
    """A pool of flights as model tensors: x (T, 42) per flight and a_next (T - 1, 2), a_next[k] = a_{k+1}."""

    x: list[np.ndarray]
    a_next: list[np.ndarray]

    @classmethod
    def from_arrays(cls, obs: list[np.ndarray], aux: list[np.ndarray], actions: list[np.ndarray]) -> Sequences:
        return cls([features(o, a) for o, a in zip(obs, aux, strict=True)], [action_features(b[1:]) for b in actions])


@dataclass
class Normalizer:
    mu_x: np.ndarray
    sd_x: np.ndarray
    mu_d: np.ndarray
    sd_d: np.ndarray

    @classmethod
    def fit(cls, seqs: Sequences) -> Normalizer:
        x = np.concatenate(seqs.x)
        d = np.concatenate([np.diff(s, axis=0) for s in seqs.x])
        return cls(x.mean(0), np.maximum(x.std(0), 1e-6), d.mean(0), np.maximum(d.std(0), 1e-6))

    @property
    def active(self) -> np.ndarray:
        """Channels that vary in the training data (the loss and every metric skip constant ones)."""
        return np.asarray(self.sd_x > 1e-5)


class Dynamics(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        n_in = N_X + N_A
        self.gru = nn.GRU(n_in, HIDDEN, batch_first=True) if kind == "gru" else None
        head_in = n_in + (HIDDEN if self.gru is not None else 0)
        self.head = nn.Sequential(nn.Linear(head_in, HEAD), nn.SiLU(), nn.Linear(HEAD, HEAD), nn.SiLU(), nn.Linear(HEAD, 2 * N_X))

    def forward(self, x: torch.Tensor, a: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(B, T, 42) standardized states + (B, T, 2) actions -> mean and log-std of the standardized increment."""
        inp = torch.cat([x, a], dim=-1)
        if self.gru is not None:
            z, h = self.gru(inp, h)
            inp = torch.cat([z, inp], dim=-1)
        mean, raw = self.head(inp).chunk(2, dim=-1)
        return mean, LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * torch.sigmoid(raw), h


def _batch(seqs: Sequences, idx: np.ndarray, norm: Normalizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded (x_k, a_{k+1}) inputs, standardized increment targets and the step mask for flights idx."""
    t_max = max(len(seqs.a_next[i]) for i in idx)
    x = np.zeros((len(idx), t_max, N_X), np.float32)
    a = np.zeros((len(idx), t_max, N_A), np.float32)
    d = np.zeros((len(idx), t_max, N_X), np.float32)
    m = np.zeros((len(idx), t_max), np.float32)
    for j, i in enumerate(idx):
        s, n = seqs.x[i], len(seqs.a_next[i])
        x[j, :n] = (s[:-1] - norm.mu_x) / norm.sd_x
        a[j, :n] = seqs.a_next[i]
        d[j, :n] = (np.diff(s, axis=0) - norm.mu_d) / norm.sd_d
        m[j, :n] = 1.0
    return torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(d), torch.from_numpy(m)


def _nll(model: Dynamics, x: torch.Tensor, a: torch.Tensor, d: torch.Tensor, m: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    mean, log_std, _ = model(x, a)
    per_step = (0.5 * ((d - mean) * torch.exp(-log_std)) ** 2 + log_std)[..., active].mean(-1)
    loss: torch.Tensor = (per_step * m).sum() / m.sum()
    return loss


def _length_batches(seqs: Sequences, batch: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Flights sorted by length (with a random tie-break), cut into batches, batch order shuffled: little padding."""
    lengths = np.array([len(s) for s in seqs.a_next])
    order = np.lexsort((rng.random(len(lengths)), lengths))
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    rng.shuffle(batches)  # type: ignore[arg-type]
    return batches


def fit(kind: str, train: Sequences, val: Sequences, seed: int, epochs: int) -> tuple[Dynamics, Normalizer, list[dict[str, float]]]:
    """Teacher-forced Gaussian-NLL training; returns the epoch with the best validation NLL."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Normalizer.fit(train)
    model = Dynamics(kind)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = epochs * len(_length_batches(train, BATCH_FLIGHTS, rng))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=LR / 10)
    active = torch.from_numpy(norm.active)
    val_batches = [_batch(val, idx, norm) for idx in _length_batches(val, 256, np.random.default_rng(0))]
    best, best_state, history = np.inf, model.state_dict(), []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for idx in _length_batches(train, BATCH_FLIGHTS, rng):
            loss = _nll(model, *_batch(train, idx, norm), active)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            weights = [float(b[3].sum()) for b in val_batches]
            val_nll = float(np.average([_nll(model, *b, active).item() for b in val_batches], weights=weights))
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": val_nll, "wall_s": time.perf_counter() - t0})
        print(f"  {kind} s{seed} epoch {epoch:3d}  train {history[-1]['train_nll']:.4f}  val {val_nll:.4f}  ({history[-1]['wall_s']:.0f} s)")
        if val_nll < best:
            best, best_state = val_nll, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, norm, history


def save(path: Path, model: Dynamics, norm: Normalizer, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.kind, "state": model.state_dict(), "norm": norm.__dict__, "history": history}, path)


def load(path: Path) -> tuple[Dynamics, Normalizer, list[dict[str, float]]]:
    ckpt = torch.load(path, weights_only=False)
    model = Dynamics(ckpt["kind"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, Normalizer(**ckpt["norm"]), ckpt["history"]


class Predictor:
    """numpy-facing rollouts of a trained model (raw 42-feature space in and out)."""

    def __init__(self, model: Dynamics, norm: Normalizer) -> None:
        self.model, self.norm = model, norm

    @property
    def recurrent(self) -> bool:
        return self.model.gru is not None

    def _std(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((x - self.norm.mu_x) / self.norm.sd_x).astype(np.float32))

    @torch.no_grad()
    def teacher_forced(self, x: np.ndarray, a_next: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """One flight's one-step predictions: mean and std of x_{k+1} for every k, plus the GRU state after each step.

        x (T, 42), a_next (T - 1, 2). Returns mean (T - 1, 42), std (T - 1, 42), h (T - 1, HIDDEN) or None.
        h[k] is the state that has consumed (x_0, a_1) .. (x_k, a_{k+1}).
        """
        mean, log_std, _ = self.model(self._std(x[:-1])[None], torch.from_numpy(a_next)[None])
        mu = x[:-1] + self.norm.mu_d + self.norm.sd_d * mean[0].numpy()
        sd = self.norm.sd_d * np.exp(log_std[0].numpy())
        h = None
        if self.model.gru is not None:
            z, _ = self.model.gru(torch.cat([self._std(x[:-1])[None], torch.from_numpy(a_next)[None]], dim=-1))
            h = z[0].numpy()
        return mu, sd, h

    @torch.no_grad()
    def step(self, h: np.ndarray | None, x: np.ndarray, bank: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray | None, np.ndarray]:
        """Advance B states one tick under bank a_{k+1}: (h_k, x_k) -> (h_{k+1}, x_{k+1}); mean unless rng is given.

        The MLP ignores h and returns None for it.
        """
        ht = None if h is None or self.model.gru is None else torch.from_numpy(h.astype(np.float32))[None]
        mean, log_std, h_new = self.model(self._std(x)[:, None], torch.from_numpy(action_features(bank))[:, None], ht)
        d = mean[:, 0].numpy()
        if rng is not None:
            d = d + np.exp(log_std[:, 0].numpy()) * rng.standard_normal(d.shape).astype(np.float32)
        x_next = x + self.norm.mu_d + self.norm.sd_d * d
        return (None if h_new is None else h_new[0].numpy()), x_next.astype(np.float32)

    def free_run(
        self,
        h: np.ndarray | None,
        x: np.ndarray,
        banks: np.ndarray,
        rng: np.random.Generator | None = None,
        stop: Callable[[np.ndarray], np.ndarray] | None = None,
        keep: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Roll B states forward on banks (B, H) = a_{k+1..k+H}; returns x_{k+1..k+H} (B, H, 42).

        With `stop` (x -> finished mask), the rollout ends once every row has finished at least once.
        With `keep` (1-based horizons), only those steps are returned: (B, len(keep), 42).
        """
        slots = None if keep is None else {hz - 1: i for i, hz in enumerate(keep)}
        n = banks.shape[1] if keep is None else max(keep)
        out = np.empty((x.shape[0], n if keep is None else len(keep), N_X), np.float32)
        finished = np.zeros(x.shape[0], bool)
        for j in range(n):
            h, x = self.step(h, x, banks[:, j], rng)
            if slots is None:
                out[:, j] = x
            elif j in slots:
                out[:, slots[j]] = x
            if stop is not None:
                finished |= stop(x)
                if finished.all():
                    return out[:, : j + 1]
        return out
