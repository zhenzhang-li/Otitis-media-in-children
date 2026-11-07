# Otoscopy LayerFusion research code

[中文使用说明](README.zh-CN.md)

A configurable implementation of the study's **ten frozen backbones, cumulative layer readouts, global/local LayerFusion, limited-label adaptation, probability calibration, robustness and region/layer explanations**. The code supports readers' own image collections; it does not require the study's directory names, clinical spreadsheets, four labels or image counts.

The internal record interface needs an image resource and a reference label. CSV/TSV, Excel, JSON/JSONL, directory scanning and custom Python adapters map your existing format to that interface. The default task is supervised, closed-set, RGB **image classification**; another modality/storage format needs an appropriate reader/decoder. This is a research implementation, not a clinical diagnostic product.

## Installation

Python 3.10+; install the PyTorch/torchvision build appropriate for your CPU/CUDA environment first, following [PyTorch](https://pytorch.org/get-started/locally/). Then, from the repository root:

```bash
python -m pip install -e ".[dev]"
otoscopy-ai --help
```

The release was checked with Python 3.13 and PyTorch 2.12/CUDA 12.6; see [validation](docs/VALIDATION.md). Dependencies are in `pyproject.toml`. No model or dataset is downloaded implicitly. Obtain the study's pretrained encoders separately; [weights and providers](docs/MODELS.md) explains the exact configuration. DINOv3 uses its [official code and separate license](https://github.com/facebookresearch/dinov3).

## Offline demonstration

The demo uses generated coloured patterns and two small **untrained toy encoders**, not medical photographs or substitute scientific results.

```bash
python examples/generate_synthetic.py
otoscopy-ai --config configs/smoke.yaml run
otoscopy-ai --config configs/smoke.yaml robustness --all-seeds
otoscopy-ai --config configs/smoke.yaml explain
otoscopy-ai --config configs/smoke.yaml summarize
otoscopy-ai --config configs/smoke.yaml plot
python -m pytest -q
```

Outputs appear under `outputs/smoke/`. The smoke configuration uses two outer splits, two inner seeds, short training and low-resolution plots to run quickly. It is deliberately distinct from the paper configuration.

## Run with your data

1. Copy `configs/study.yaml` to a new YAML in `configs/`.
2. Change `data.sources`: your metadata file(s), image roots and column mappings, or use a directory/custom adapter. Configure grouping and any label aliases explicitly. See [data adapters](docs/DATA.md).
3. Provide local model checkpoints and a local official DINOv3 checkout. Reduce the model list for a first run.
4. Set a new `output` directory, device, split mode/seeds and training options.

```bash
otoscopy-ai --config configs/my_data.yaml prepare
otoscopy-ai --config configs/my_data.yaml inspect
otoscopy-ai --config configs/my_data.yaml split
otoscopy-ai --config configs/my_data.yaml cache
otoscopy-ai --config configs/my_data.yaml train --section backbones
otoscopy-ai --config configs/my_data.yaml train --section layers
otoscopy-ai --config configs/my_data.yaml train --section fewshot
otoscopy-ai --config configs/my_data.yaml evaluate --all-internal
otoscopy-ai --config configs/my_data.yaml evaluate --selected
otoscopy-ai --config configs/my_data.yaml robustness --all-seeds
otoscopy-ai --config configs/my_data.yaml explain
otoscopy-ai --config configs/my_data.yaml summarize
otoscopy-ai --config configs/my_data.yaml plot
```

`run` performs preparation, splits, caching, **all training before evaluation**, internal evaluation, selection of one primary fusion checkpoint, its calibration/external evaluation, summaries and plots. Robustness and explanations are explicit separate commands because their compute cost is configurable. `evaluate --selected` picks the highest observed internal epoch-selection macro-F1; it never ranks candidates using external or internal evaluation scores. The candidate selection lists differ, so this rule does not prove unbiased optimality.

The paper configuration produces 10 backbone heads + 74 layer configurations + 150 limited-label heads per split; five splits give 1,170 fits. Sections 3.2/3.3/3.5 use repeated internal outcomes. The latest manuscript's section 3.4 uses one selection-score-best fusion checkpoint; the public pipeline evaluates external cohorts for that selected model, not a five-model ensemble. This selection/reporting decision in the manuscript was made after earlier evaluations; using the pipeline does not make historical data a new blind validation cohort. See [study mapping](docs/STUDY_PROTOCOL.md).

## Training and debugging

All heads use selection macro-F1, configurable early stopping, mixed precision, AdamW and cosine scheduling. The study default **freezes encoders**; “limited-label adaptation” retrains the linear/fusion head, not the encoder. End-to-end encoder fine-tuning is not implemented in this release.

Every head exports `history.csv`, selected-epoch metadata, selection predictions and a portable PyTorch checkpoint. Training-only feature means/standard deviations, class order, training/selection IDs and cache/config/source fingerprints are recorded. An incompatible checkpoint/cache triggers an error; a changed experiment should use a new output directory. `training.feature_storage: stream` lowers GPU memory use.

```bash
# Overrides precede the stage command. Use a new output for changed training settings.
otoscopy-ai --config configs/my_data.yaml --set output=../outputs/lr_trial --set training.lr=0.0003 train --section backbones --seed 42

# Optional selection-only grid search; no evaluation/calibration/external labels rank trials.
python scripts/tune.py --config configs/my_data.yaml --space configs/search.yaml --encoder dinov3_vitb16 --kind tokens
```

The optional grid-search utility is an extension for readers. It was not the search procedure used for the manuscript's fixed hyperparameter reruns. See [outputs and troubleshooting](docs/USAGE.md).

## Analysis and figures

`summarize` computes outer-split means and pointwise t intervals. Limited-label inner runs are averaged **within each outer split first**; they are not pooled as independent outer splits. Single-checkpoint evaluation uses image-level proportion/macro-metric intervals, with exact methods recorded in [statistics](docs/STATISTICS.md). Group/patient dependence and model-selection uncertainty are not included in image-level intervals.

Robustness applies each perturbation separately to the **source image**, then the same resize/crop/normalization. An original-image equivalence check catches preprocessing/precision drift. Explanations perform region retention/replacement, grey/blur reference checks, exact layer Shapley decomposition and optional full region-by-layer allocation. Failure/fallback cases are retained. Feature clusters are not validated clinical lesions; reference ranges are not statistical CIs.

Figure width, PNG DPI, Arial/font size, colours, line/marker sizes and figure heights are controlled in the YAML `plot` section. The study defaults are **190 mm, Arial, 600 dpi**, with PDF vector text/lines; example image panels are raster. Install Arial or explicitly change the configured font on your system.

## Repository layout

| Location | Purpose |
|---|---|
| `src/otoscopy_ai/data.py` | Configurable adapters, labels, grouping and four data roles |
| `models.py`, `fusion.py` | Ten encoders, extensible providers and fusion/readout heads |
| `engine.py`, `workflows.py` | Feature cache, training, shared few-shot samples and selection |
| `robustness.py`, `explain.py` | Perturbation and discrete region/layer analyses |
| `metrics.py`, `plotting.py` | Multiclass endpoints, intervals and figures |
| `configs/` | Study, offline demo and optional tuning settings |
| `examples/` | Synthetic generator and custom source/image-loader examples |
| `tests/`, `.github/workflows/` | Synthetic tests and Linux/Windows CI configuration |
| `scripts/export_release.py` | Source-only archive and private-path/artifact checks |

## GitHub and paper link

This folder is prepared for upload; it has **not** been published to a GitHub account. Follow [publishing](docs/PUBLISHING.md), fill your repository URL and citation metadata, then put the actual public URL in the manuscript. A [code-availability text template](docs/CODE_AVAILABILITY.md) is supplied; placeholder links are not published repositories.

No internal clinical data, patient metadata, fitted study checkpoints or pretrained third-party weights are bundled. The new project utilities use the MIT license; upstream code/weights retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
