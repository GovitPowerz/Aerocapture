# Development

Most commits in this repository carry a `Co-Authored-By: Claude` trailer: the code is written largely with coding agents (Claude Code, briefed by [CLAUDE.md](CLAUDE.md) and the contracts in [docs/agents/](docs/agents/)). This page says which decisions are the author's, what evidence a change needs before it lands, and how the split is enforced.

## Division of labour

Agents implement. They take a GitHub issue with stated acceptance criteria and turn it into a branch and a pull request: features, refactors behind bit-identity gates, test suites, first drafts of docs, and commissioned reviews of the codebase.

The author decides. Scientific hypotheses, experiment design and the seed pools reserved before a run, the acceptance criteria written into each issue, the architecture decisions recorded as [ADRs](docs/adr/), which numbers the paper and README quote, and every merge to `main` stay human-owned. An agent may draft any of these; none becomes true until the author has checked it against the tree or re-run it.

## Evidence rule

A pull request is accepted when its description names the gates it re-ran and they pass. The standing gates are listed in [CLAUDE.md](CLAUDE.md) (Build & Development Commands): the six guidance goldens under `tests/reference_data/rust_golden/`, `tests/test_pyo3.py::test_pyo3_matches_subprocess`, the `run_grid` bit-identity gate (`tests/test_run_grid.py`, [ADR-0004](docs/adr/0004-run-grid-bit-identity-chokepoint.md)) and its thread-count companion (`tests/test_thread_invariance.py`), and the per-layer Rust/PyTorch equivalence gates (`tests/test_nn_equivalence.py`). Numbers must not move; a change that touches another bit-identity gate names it.

A numerical or scientific claim needs an independent check on top: a paired comparison on reserved pools, seed-repeat error bars, or the frozen 10 x 100,000-scenario confirmatory pools. Claims an agent writes into a PR, a design doc or the paper are re-run or checked against the tree before merge; a commissioned review's findings are fact-checked one by one before any becomes an issue.

## Enforcement

Merge authority and `main` are human-only. The agent's permission layer, [.claude/settings.json](.claude/settings.json), allows issue work, feature-branch pushes and PR creation, and denies the merge commands, force pushes and pushes to `main`. It binds the agent's tooling, not GitHub:

- `Bash(gh pr merge:*)`
- `Bash(gh api * --method PUT:*)`
- `Bash(gh api * -X PUT:*)`
- `Bash(gh api --method PUT:*)`
- `Bash(gh api -X PUT:*)`
- `Bash(gh api *merge*)`
- `Bash(gh repo:*)`
- `Bash(git push --force:*)`
- `Bash(git push -f:*)`
- `Bash(git push --force-with-lease:*)`
- `Bash(git push * --force*)`
- `Bash(git push * -f*)`
- `Bash(git push origin main:*)`
- `Bash(git push origin main)`
- `Bash(git push -u origin main:*)`
- `Bash(git push --set-upstream origin main:*)`
- `Bash(git push origin HEAD:main:*)`
- `Bash(git push * main:*)`
- `Bash(git push * main)`

`tests/test_development_md.py` fails if this list and the settings file drift apart. Branch, issue, PR and label conventions are the machine-facing contracts in [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md) and [docs/agents/triage-labels.md](docs/agents/triage-labels.md); every PR runs the CI described in [CLAUDE.md](CLAUDE.md) (Conventions, CI).

## Worked examples

- [PR #94](https://github.com/GovitPowerz/Aerocapture/pull/94): a bounded refactor from an issue brief. The invariant is stated up front (each NN layer declares its tensors once), 13 byte-identity fixtures are frozen on the pre-refactor tree, and the cross-language gates are re-run before merge.
- [ADR-0003](docs/adr/0003-per-draw-noise-seeding.md) and Appendix E of the [paper](articles/paper/paper.pdf): a result-changing flaw in the project's own evaluation (one shared noise path) found after release, quantified on paired pools, repaired, and every headline cell retrained.
- [Issue #101](https://github.com/GovitPowerz/Aerocapture/issues/101) and [PR #130](https://github.com/GovitPowerz/Aerocapture/pull/130): a GRU-PPO cell quoted as "dense PPO" in the paper, caught in review and requoted from protocol-matched cells.
- [experiments/trainer_seam_gate/](experiments/trainer_seam_gate/README.md): a trainer refactor accepted only after three real trainings (GA fixed seeds, GA adaptive seeds on a Mamba policy, islands) matched the pre-refactor tree record for record. Building the gate exposed that pymoo's RNG had never been seeded.
- [Commit d23053a](https://github.com/GovitPowerz/Aerocapture/commit/d23053a): the pymoo `<0.6.2` pin. A silent SIGABRT with no traceback, traced to moocore's `igd()` on problems wider than 32 dimensions, pinned with a one-line repro that says when to lift it.
- The two commissioned architecture reviews of September 2026 (tracked as GitHub issues): each finding checked against the tree before it became an issue.
