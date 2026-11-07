# Contributing

Use a separate branch and include focused tests for data adapters or changed scientific behaviour. Keep private paths, credentials, clinical images and patient spreadsheets outside version control.

```bash
python -m pip install -e ".[dev]"
python -m black src scripts examples tests --line-length 100
python -m pytest -q
python -m build
```

Describe whether a change affects preprocessing, training-only normalization, source weights, model selection, calibration, evaluation units or explanation targets. Preserve deterministic ID alignment and group/role audits. A changed configuration or implementation must not silently reuse an incompatible cache/checkpoint. End-to-end backbone fine-tuning would be a new feature and a different protocol from the frozen-encoder manuscript experiments.

Tests use synthetic images and untrained toy encoders, run without model downloads and cover arbitrary classes/metadata, four-role splits, group integrity, training-only statistics, few-shot pairing, calibration, selected-model evaluation, perturbation equivalence and explanation additivity.
