# Commands, logs and debugging

Run commands from the repository root after `pip install -e .`. The executable and `python -m otoscopy_ai` are equivalent. Global options (`--config`, `--set`) precede the stage command.

## Stages

```bash
otoscopy-ai --config configs/my_data.yaml train --section backbones --seed 42
otoscopy-ai --config configs/my_data.yaml train --section layers --seed 42
otoscopy-ai --config configs/my_data.yaml train --section fewshot --seed 42
otoscopy-ai --config configs/my_data.yaml evaluate --all-internal
otoscopy-ai --config configs/my_data.yaml select-model
otoscopy-ai --config configs/my_data.yaml evaluate --selected
otoscopy-ai --config configs/my_data.yaml calibrate --seed 42 --name fusion_dinov3_vitb16
otoscopy-ai --config configs/my_data.yaml robustness --all-seeds --name fusion_dinov3_vitb16
otoscopy-ai --config configs/my_data.yaml explain --seed 42 --name fusion_dinov3_vitb16 --ids YOUR_CASE_ID
otoscopy-ai --config configs/my_data.yaml summarize
otoscopy-ai --config configs/my_data.yaml compare --a fusion_dinov3_vitb16 --b backbone_convnext_base
otoscopy-ai --config configs/my_data.yaml plot
```

Without `--seed`, training uses every configured outer seed. Explanation/robustness without `--seed` or `--all-seeds` uses the selected candidate. Robustness is internal by default; readers extending external perturbation analysis should keep their original external-list membership and locked parameters. No trained model is selected from evaluation performance.

## Artifacts

```text
outputs/your_run/
  records.csv, classes.json, dataset_audit.json
  splits/seed_*/manifest.csv, class_counts.csv, overlap_audit.json
  cache/ENCODER/{pooled,cls,tokens}.npy, metadata.json
  runs/seed_*/heads/HEAD/
    history.csv, summary.json, checkpoint.pt
    selection_predictions.csv
    calibration.json, calibration_predictions.csv      # when calibrated
    predictions_internal_evaluation.csv, metrics_internal_evaluation.json
    predictions_COHORT.csv, metrics_COHORT.json         # selected-model external analysis
    risk_coverage_COHORT.csv
  runs/seed_*/fewshot_samples/K*_r*.csv
  runs/seed_*/robustness/HEAD/
  runs/seed_*/explanations/HEAD/
  internal_runs.csv, internal_summary.csv, fewshot_outer_means.csv
  selected_model.json, primary_candidates.csv
  figures/*.pdf, *.png
```

Dimensions are inferred from feature caches. Head checkpoints contain class order, normalization, exact training/selection IDs, head specification, selected epoch, cache fingerprint and original fusion/preprocessing settings. Training histories record loss, selection macro-F1/accuracy and scheduled learning rate. The epoch ceiling and actual epochs differ because of early stopping; selected epochs can precede the minimum number of executed epochs.

## Optional grid search

```bash
python scripts/tune.py --config configs/my_data.yaml --space configs/search.yaml --encoder dinov3_vitb16 --kind tokens --seed 42
```

Trials share the same deterministic frozen feature cache but use separate output/checkpoint directories. The table `tuning/selection_scores.csv` ranks **selection macro-F1 only**. No trial invokes evaluation or calibration. The supplied grid is an illustrative extension, not the manuscript's original tuning history. A search that changes input preprocessing or cached fusion token layers needs a new cache; this utility rejects such changes.

Copy `tuning/best_parameters.json` values into a final configuration with a fresh output directory, fit the intended final model, then explicitly calibrate/evaluate it. Trial histories/checkpoints remain an audit of the search; they are not automatically presented as final test results.

## Common fixes

- **No images / column missing:** check the adapter, source-level column mapping, glob/regex and configuration-relative roots. First run `prepare`, then `inspect`.
- **Missing class in a split:** increase class/group counts, adjust ratios or supply valid predefined splits. Dropping a rare class silently would change the task and is not done.
- **Content/group overlap:** inspect the split audit. Correct patient/content grouping, or explicitly use `report` for a documented overlap-retaining sensitivity analysis.
- **CUDA unavailable:** install a matched CUDA build or select `device: cpu`/`auto`.
- **GPU memory:** lower image/training batch sizes, use `feature_storage: stream`, use one encoder at a time. Frozen token caches can consume many GB even though head parameters are modest.
- **Checkpoint/cache signature changed:** use a fresh output, or deliberately rebuild the affected cache. Changed source data require `prepare` again; changed code/preprocessing/weights are fingerprinted.
- **DINOv3 import error:** use its official supported environment. This project loads the backbone module rather than segmentation adapters; a different already-imported DINO checkout requires a fresh process.
- **Original-image robustness mismatch:** inspect preprocessing, cache quantization and AMP/batch precision before accepting perturbation results.
- **Explanation too expensive:** reduce cases/regions; exact region enumeration costs 2^R, and full region×layer decomposition is much more expensive. The default preserves both successful and retention-only fallback cases.
- **Arial unavailable:** install Arial or change `plot.font`. PDF text remains vector, but font substitution should be reported for journal figures.

Use a new output directory for scientific changes. Cached features are sample-wise, but standardization, head fitting and calibration are trained quantities with distinct data roles.
