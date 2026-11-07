# Statistical units and uncertainty

All stored metric values are proportions in [0,1] unless explicitly described otherwise; plots convert to percent. Macro-F1 is the unweighted mean of class-wise F1, not the harmonic mean of macro-precision and macro-recall. Sensitivity and recall coincide at the class level. The optional `evaluation.clinical_pair` defines two labels for directional errors; neither the pair nor any otology label is hard-coded.

## Repeated internal splits

For n outer-split outcomes x, `summarize` reports arithmetic mean, sample SD and the pointwise interval:

```text
mean ± t(0.975, n−1) × SD / sqrt(n)
```

For n=5 the degrees of freedom are 4. With one outer outcome, CI/SD are left unavailable rather than inventing a repeated-split interval. The `n_outer_splits` column is always exported. Limited-label runs first average the inner seeds within one outer split, then use those outer means. Twenty-five inner runs across five outer splits are not twenty-five independent split estimates. Paired comparisons subtract models within matched outer seeds before applying the same formula; comparison outputs use proportions, not percentage points, until multiplied by 100.

These nominal, unadjusted intervals describe the mean under the configured split/fit procedure conditional on the same existing image pool and fixed encoder/seed schedule. Repeated overlapping splits are not independent clinical cohorts or k-fold cross-validation. Patient/content clustering, full development/selection uncertainty and historical cohort selection are not included. No claim of independent clinical significance or non-inferiority follows from a pointwise interval. This public utility does not automatically produce multiplicity-adjusted P values.

## One selected checkpoint

`evaluate --selected` selects by internal epoch-selection macro-F1, fits positive temperature/entropy threshold only on that candidate's calibration role and evaluates the fixed model. It reports image-level Wilson intervals for accuracy/class sensitivity/directional proportions, and configured image-bootstrap percentile intervals for macro-precision/recall/F1/specificity. The number of bootstrap replicates is a configuration value. AUC in this generic release is a point estimate; it is unavailable when the cohort lacks either positive or negative images for a class. NLL/Brier are point estimates. The manuscript's separately generated additional AUC/probability-score intervals are not claimed to be automatically reproduced by this generic release.

Temperature minimises calibration NLL over bounded log(T), remains positive and cannot change argmax. Entropy uses natural logarithms. The empirical lower quantile fixes the review threshold; ties mean achieved coverage need not equal the nominal target exactly. Retained accuracy is conditional on retained images and must be read with overall/class coverage and review burden. An offline review flag is not clinician correction or a prescribing outcome.

The single-model CI conditions on the chosen weights; it cannot replace an outer-split CI. Selection across different candidate selection lists may be optimistic, and the CI does not include that selection. A later reporting decision on previously examined cohorts is not a new blinded validation. No favourable external seed is chosen by this pipeline.

## Explanations

The target is the original predicted-vs-runner-up logit margin, held fixed throughout perturbation. Regions use spatially constrained Ward clustering on concatenated normalized layer patch features. Exhaustive retention/replacement evaluates all 2^R region unions against grey and same-image blur references. The smallest dual-feasible union is chosen; if none exists, the smallest retention-feasible union is kept and explicitly marked fallback. Exact layer Shapley values split the margin change between full and replaced representations, recomputing all fusion operations. Full regional Shapley plus conditional layer allocation is optional and checked for additivity.

Layer bars show mean/range over **two construction references**, not a sampling confidence interval. The method does not validate clinical lesions, establish causal pathophysiology or reproduce the archived clinician heatmaps. Synthetic perturbations do not establish real-world acquisition thresholds.
