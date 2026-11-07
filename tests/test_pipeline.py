import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from otoscopy_ai.config import output_dir, read_json
from otoscopy_ai.data import prepare, split, manifest
from otoscopy_ai.engine import cache_features, cache_for, fit, calibrate, evaluate_head
from otoscopy_ai.workflows import train_fewshot, summarize, select_checkpoint
from otoscopy_ai.metrics import mean_ci
from otoscopy_ai.robustness import run as robust
from otoscopy_ai.explain import run as explain, shapley


def test_frozen_head_pipeline_and_training_only_statistics(experiment):
    cfg, frame, path = experiment
    prepare(cfg)
    split(cfg)
    cache_features(cfg, "toy_b")
    spec = {"name": "linear", "encoder": "toy_b", "kind": "pooled", "section": "backbones"}
    fit(cfg, 19, spec)
    rows = manifest(cfg, 19)
    root = output_dir(cfg) / "runs/seed_19/heads/linear"
    ck = torch.load(root / "checkpoint.pt", weights_only=True)
    arrays, _ = cache_for(cfg, "toy_b")
    expected = (
        np.asarray(arrays["pooled"][rows.loc[rows.role.eq("train"), "index"]])
        .astype(np.float32)
        .mean(0)
    )
    assert ck["nclasses"] == 3
    assert np.allclose(ck["mu"].numpy()[0], expected, atol=1e-6)
    assert set(ck["train_ids"]) == set(rows.loc[rows.role.eq("train"), "id"])
    assert not set(ck["train_ids"]) & set(rows.loc[rows.role.ne("train"), "id"])
    with pytest.raises(ValueError, match="exclusively"):
        fit(cfg, 19, {**spec, "name": "forbidden"}, rows[rows.role.eq("calibration")])
    cal = calibrate(cfg, 19, "linear")
    assert cal["temperature"] > 0 and cal["argmax_unchanged"]
    assert set(cal["calibration_ids"]) == set(rows.loc[rows.role.eq("calibration"), "id"])
    results = evaluate_head(cfg, 19, "linear", external=True)
    assert {r["cohort"] for r in results} == {"internal_evaluation", "other_hospital"}
    cfg["training"]["lr"] *= 2
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        fit(cfg, 19, spec)


def test_fewshot_shared_samples_and_nested_aggregation(experiment):
    cfg, frame, path = experiment
    prepare(cfg)
    split(cfg)
    train_fewshot(cfg, 19)
    for k in [1, 2]:
        for repeat in [1, 2]:
            ck = []
            for alias in ["linear", "fusion"]:
                name = f"few_{alias}_K{k}_r{repeat}"
                ck.append(
                    torch.load(
                        output_dir(cfg) / f"runs/seed_19/heads/{name}/checkpoint.pt",
                        weights_only=True,
                    )
                )
                evaluate_head(cfg, 19, name)
            assert ck[0]["train_ids"] == ck[1]["train_ids"]
            assert len(ck[0]["train_ids"]) == 3 * k
    summary = summarize(cfg)
    assert all(r["n_outer_splits"] == 1 for r in summary)
    assert all(r["ci_low"] is None for r in summary)
    # A second synthetic outer mean must change n_outer, not count inner repeats as independent splits.
    for p in (output_dir(cfg) / "runs/seed_19/heads").glob("*/metrics_internal_evaluation.json"):
        v = read_json(p)
        v["seed"] = 29
        v["accuracy"] += 0.01
        v["macro_f1"] += 0.01
        dest = output_dir(cfg) / "runs/seed_29/heads" / p.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / p.name).write_text(json.dumps(v))
    summary = summarize(cfg)
    assert all(r["n_outer_splits"] == 2 for r in summary)
    outer = pd.read_csv(output_dir(cfg) / "fewshot_outer_means.csv")
    r = next(
        x for x in summary if x["model"] == "linear" and x["K"] == 1 and x["metric"] == "accuracy"
    )
    expected = mean_ci(outer[outer.alias.eq("linear") & outer.K.eq(1)].accuracy)
    assert np.isclose(r["ci_low"], expected["ci_low"])


def test_fusion_robustness_and_explanation(experiment):
    cfg, frame, path = experiment
    prepare(cfg)
    split(cfg)
    spec = {"name": "fusion_toy_b", "encoder": "toy_b", "kind": "tokens", "section": "layers"}
    fit(cfg, 19, spec)
    calibrate(cfg, 19, spec["name"])
    evaluate_head(cfg, 19, spec["name"])
    result = robust(cfg, 19, spec["name"])
    assert len(result) == 5
    assert result[0]["condition"] == "original"
    cases = explain(cfg, 19, spec["name"])
    assert len(cases) == 1
    assert len(cases[0]["layer_contribution_mean"]) == 3
    assert cases[0]["clinical_lesion_validation"] is False
    folder = output_dir(cfg) / "runs/seed_19/explanations/fusion_toy_b"
    array = np.load(next(folder.glob("*.npz")))
    assert np.allclose(
        array["region_layer_allocation"].sum(-1), array["regional_shapley"], atol=1e-4
    )


def test_selection_uses_selection_scores_only(experiment):
    cfg, _, _ = experiment
    for seed, score in [(19, 0.8), (29, 0.7)]:
        p = output_dir(cfg) / f"runs/seed_{seed}/heads/fusion_toy_b"
        p.mkdir(parents=True, exist_ok=True)
        (p / "summary.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "name": "fusion_toy_b",
                    "selected_epoch": 4,
                    "best_selection_macro_f1": score,
                    "signature": str(seed),
                }
            )
        )
        (p / "metrics_internal_evaluation.json").write_text(
            json.dumps({"accuracy": 0 if seed == 19 else 1})
        )
    assert select_checkpoint(cfg)["seed"] == 19


def test_exact_shapley_additivity_with_signed_interactions():
    values = np.array([0, 2, -1, 4, 3, 8, 1, 10], float)
    s = shapley(values)
    assert np.isclose(s.sum(), values[-1] - values[0])
    assert len(s) == 3
