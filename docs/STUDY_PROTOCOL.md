# Mapping to the manuscript

| Manuscript component | Public command / setting |
|---|---|
| 3.2 ten frozen backbone linear heads | `train --section backbones`; `evaluate --all-internal` |
| 3.3 cumulative forward/reverse layers | `train --section layers`; all depths inferred from CLS caches |
| Three-layer global/local fusion | same stage, `fusion.layers: [4,8,12]` |
| ViT-S native versus zero-padded head | `experiments.pad_to: {dinov3_vits16: 768}` |
| 3.4 one primary fusion checkpoint | `evaluate --selected`; selection-set macro-F1 only |
| 3.5 limited-label adaptation | `train --section fewshot`; shared sampled IDs across models |
| 3.5 degradation | `robustness --all-seeds` |
| 3.6 region/layer dependence | `explain`; optional `--ids` and full decomposition |
| Repeated split / nested mean CI | `summarize` |
| Paired model differences | `compare --a HEAD_A --b HEAD_B --metric accuracy` |

The paper's raw-directory experiment used 7,281 image records, image-stratified 60%/15%/15%/10% allocation and outer seeds 20261104/42/123/2026/3407. No such image counts or labels are enforced in reusable code. Reader studies and specialist agreement were not repeated here; the public model code does not fabricate or regenerate human responses.

`study.yaml` supplies the reported model/training settings but defaults to rejecting overlap for readers' new datasets. To reproduce an intentionally overlap-retaining raw-directory analysis, explicitly set `split.overlap_policy: report` and disclose the resulting audit; otherwise use appropriate independent grouping. This is a documented data-policy choice, not an automatically applied exclusion rule.

Per outer split: 10 backbone heads; ViT-B 12 forward + 12 reverse pooled CLS configurations; ViT-S native/padded each 12 forward + 12 reverse; and two fusion heads = 74 layer configurations. Five limited-label model variants × six budgets × five inner seeds = 150 heads. Total 234 per split and 1,170 across five splits. Identical full-depth forward/reverse features are not independent evidence. Cumulative CLS heads differ from LayerFusion in local inputs, nonlinear head and auxiliary losses; results cannot isolate sparse layer selection alone.

Hyperparameters in `study.yaml` preserve AdamW, 200-epoch ceiling, at least 30 executed epochs, selection macro-F1, 30 stale epochs and minimum early-stopping improvement 0.0001. The checkpoint maximises selection macro-F1; its epoch can precede 30. Linear features use training-only population SD, clamped at 0.0001. All encoders remain frozen. The five inner seeds for few-shot sampling/head initialisation are held fixed across outer splits; full-budget head seed is fixed separately.

The selected-model workflow ranks only observed epoch-selection macro-F1 across five candidate fusion heads, never evaluation scores, and uses the chosen model's separate calibration set for temperature/entropy threshold. The latest manuscript's seed 20261104/epoch22 was selected during a later reporting revision after previous evaluations; reusable source cannot turn historical cohorts into new blinded validation. Different candidate selection lists and model selection itself limit inference. The public package does not ship the internal cohort or claim its synthetic outputs reproduce manuscript percentages.

For a new dataset, specify patient/content groups if available, verify the reference standard, and report changed ratios, preprocessing, pretrained checkpoint variants, seeds, missing labels and overlaps. Do not choose favourable external seeds or image subsets after looking at outcomes.
