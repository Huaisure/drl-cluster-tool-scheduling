# Repository Guidelines

## Project Structure & Module Organization

`cluster_rl/` contains the Gymnasium environment, heterogeneous-graph adapter, HGT/Transformer policy, and PPO trainer. `cluster_toolkit/` provides the problem schema, event-driven engine, instance generator, and schedule validators; each toolkit component keeps focused tests under its own `tests/` directory. Repository-wide RL and integration tests live in `tests/`. Use `examples/` for fixed scenarios, `datasets/{train,validation,test}/` for materialized inputs, `scripts/` for reproducible workflows, and `runs/` for generated checkpoints and metrics. Architecture details are in `cluster_rl/TRAINING.md` and `cluster_rl/hetero_graph/README.md`.

## Build, Test, and Development Commands

This is a Python project without a separate build step. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
./scripts/generate_datasets.sh
TOTAL_STEPS=10000 NUM_ENVS=2 ./scripts/train_rl.sh
```

The dataset script regenerates all three splits and overwrites their contents; reduce `TRAIN_COUNT`, `VALIDATION_COUNT`, and `TEST_COUNT` for experiments. For a small direct smoke run, follow the command in `cluster_rl/TRAINING.md`.

## Coding Style & Naming Conventions

Use Python 3 type hints, four-space indentation, and PEP 8 naming: `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep imports grouped standard-library, third-party, then local. Prefer small, explicit domain operations over duplicated intermediate state. No formatter or linter is configured, so match nearby code and keep lines readable. Treat the feature order in `cluster_rl/hetero_graph/feature_schema.py` as a model input protocol; update builders and tests together.

## Testing Guidelines

Tests use `pytest`; files and functions follow `test_*.py` and `test_*` naming. Add unit tests beside toolkit components and cross-component behavior under `tests/`. Use deterministic seeds and `tmp_path` for generated files. Run the full suite before submitting; while iterating, target a module, for example `python -m pytest tests/test_network.py -q`. No numeric coverage threshold is enforced, but new behavior and regressions should be exercised.

## Commit & Pull Request Guidelines

History favors short, imperative subjects, with occasional Conventional Commit prefixes such as `feat:` and `fix:`. Prefer a scoped form such as `fix: prevent FIFO rollout deadlock`. Pull requests should explain the scheduling or model behavior changed, list validation commands, and call out dataset/schema or checkpoint compatibility. Link the relevant issue; include plots or metric excerpts when training behavior changes. Keep generated `runs/` artifacts and local environment files out of commits unless they are intentional review fixtures.
