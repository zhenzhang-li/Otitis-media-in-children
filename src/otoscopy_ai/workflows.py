"""Experiment specifications, shared few-shot sampling, selection and summaries."""

from pathlib import Path
import numpy as np
import pandas as pd
from .config import output_dir, save_json, read_json
from .data import manifest, read_records
from .engine import cache_for, fit, evaluate_head, calibrate
from .metrics import mean_ci


def specifications(cfg, section):
    if section == "backbones":
        return [
            {"name": "backbone_" + k, "encoder": k, "kind": "pooled", "section": "backbones"}
            for k in cfg["experiments"]["backbones"]
        ]
    specs = []
    for key in cfg["experiments"]["layer_encoders"]:
        arrays, meta = cache_for(cfg, key)
        if "cls" not in arrays or "tokens" not in arrays:
            raise ValueError(f"{key} must provide cls and tokens for layer experiments.")
        depth = arrays["cls"].shape[1]
        dim = arrays["cls"].shape[-1]
        variants = [("native", None)]
        if key in cfg["experiments"].get("pad_to", {}):
            variants.append(("padded", int(cfg["experiments"]["pad_to"][key])))
        for variant, pad in variants:
            for direction in ["forward", "reverse"]:
                for count in range(1, depth + 1):
                    layers = (
                        list(range(1, count + 1))
                        if direction == "forward"
                        else list(range(depth - count + 1, depth + 1))
                    )
                    specs.append(
                        {
                            "name": f"{key}_{variant}_{direction}_{count:02d}",
                            "encoder": key,
                            "kind": "cls",
                            "layers": layers,
                            "pad_to": pad,
                            "direction": direction,
                            "n_layers": count,
                            "section": "layers",
                        }
                    )
        specs.append(
            {"name": "fusion_" + key, "encoder": key, "kind": "tokens", "section": "layers"}
        )
    return specs


def train_section(cfg, section, seeds=None):
    results = []
    for seed in seeds or cfg["split"]["seeds"]:
        if section == "fewshot":
            results.extend(train_fewshot(cfg, int(seed)))
        else:
            for spec in specifications(cfg, section):
                results.append(fit(cfg, int(seed), spec))
    if results:
        pd.json_normalize(results).to_csv(output_dir(cfg) / f"training_{section}.csv", index=False)
    return results


def train_fewshot(cfg, seed):
    rows = manifest(cfg, seed)
    pool = rows[rows.role.eq("train")]
    _, classes = read_records(cfg)
    settings = cfg["fewshot"]
    results = []
    folder = output_dir(cfg) / f"runs/seed_{seed}/fewshot_samples"
    folder.mkdir(parents=True, exist_ok=True)
    for k in settings["budgets"]:
        for repeat, inner_seed in enumerate(settings["seeds"], 1):
            gen = np.random.default_rng(inner_seed)
            ids = []
            for c in range(len(classes)):
                available = pool.index[pool.label_idx.eq(c)].to_numpy()
                if len(available) < k:
                    raise ValueError(
                        f"K={k} exceeds available training samples in class {classes[c]}."
                    )
                ids.extend(gen.choice(available, size=k, replace=False))
            samples = pool.loc[ids].sort_values("index")
            samples.to_csv(folder / f"K{k}_r{repeat}.csv", index=False)
            for model in settings["models"]:
                spec = {
                    "name": f'few_{model["alias"]}_K{k}_r{repeat}',
                    "encoder": model["encoder"],
                    "kind": model["kind"],
                    "section": "fewshot",
                    "K": int(k),
                    "repeat": repeat,
                    "inner_seed": int(inner_seed),
                    "initialization_seed": int(inner_seed),
                    "alias": model["alias"],
                }
                results.append(fit(cfg, seed, spec, samples))
    return results


def internal_evaluation(cfg, seeds=None):
    result = []
    for seed in seeds or cfg["split"]["seeds"]:
        for path in sorted((output_dir(cfg) / f"runs/seed_{seed}/heads").glob("*/checkpoint.pt")):
            result.extend(evaluate_head(cfg, int(seed), path.parent.name, external=False))
    return result


def select_checkpoint(cfg):
    name = cfg["evaluation"]["primary_head"]
    candidates = []
    for seed in cfg["split"]["seeds"]:
        path = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}/summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Fit all candidate seeds before selection: {path}")
        candidates.append(read_json(path))
    winner = max(candidates, key=lambda x: (x["best_selection_macro_f1"], -x["seed"]))
    result = {
        "seed": winner["seed"],
        "name": name,
        "selected_epoch": winner["selected_epoch"],
        "selection_macro_f1": winner["best_selection_macro_f1"],
        "signature": winner["signature"],
        "criterion": "Highest observed internal epoch-selection macro-F1; no internal evaluation or external metric used. Candidate selection lists differ; this does not establish unbiased optimality.",
        "candidate_count": len(candidates),
    }
    pd.json_normalize(candidates).to_csv(output_dir(cfg) / "primary_candidates.csv", index=False)
    save_json(output_dir(cfg) / "selected_model.json", result)
    return result


def selected_evaluation(cfg):
    selected = select_checkpoint(cfg)
    calibrate(cfg, selected["seed"], selected["name"])
    return evaluate_head(cfg, selected["seed"], selected["name"], external=True)


def summarize(cfg):
    records = []
    for path in sorted(
        (output_dir(cfg) / "runs").glob("seed_*/heads/*/metrics_internal_evaluation.json")
    ):
        r = read_json(path)
        s = r["spec"]
        extra = {f"sensitivity__{cl}": v["sensitivity"] for cl, v in r.get("per_class", {}).items()}
        if r.get("clinical_pair"):
            pair = r["clinical_pair"]
            a, b = pair["labels"]
            extra.update(
                {
                    f"error__{a}__to__{b}": pair["forward_error"],
                    f"error__{b}__to__{a}": pair["reverse_error"],
                }
            )
        records.append(
            {
                "seed": r["seed"],
                "name": r["name"],
                "section": s["section"],
                "alias": s.get("alias", r["name"]),
                "K": s.get("K", 0),
                "repeat": s.get("repeat", 0),
                **{
                    k: r[k]
                    for k in [
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "macro_specificity",
                    ]
                },
                **extra,
            }
        )
    if not records:
        raise ValueError("No internal evaluation metrics; run evaluate --all-internal.")
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir(cfg) / "internal_runs.csv", index=False)
    allrows = []
    few = frame[frame.section.eq("fewshot")]
    if len(few):
        # Average within a split before estimating outer-split uncertainty.
        outer = (
            few.groupby(["seed", "alias", "K"], sort=False)[["accuracy", "macro_f1"]]
            .mean()
            .reset_index()
        )
        outer.to_csv(output_dir(cfg) / "fewshot_outer_means.csv", index=False)
        for (alias, k), g in outer.groupby(["alias", "K"]):
            for metric in ["accuracy", "macro_f1"]:
                allrows.append(
                    {
                        "analysis": "fewshot",
                        "model": alias,
                        "K": int(k),
                        "metric": metric,
                        **mean_ci(g[metric]),
                    }
                )
    other = frame[~frame.section.eq("fewshot")]
    for name, g in other.groupby("name"):
        if g.seed.duplicated().any():
            raise ValueError("One outcome per outer seed is required.")
        for metric in [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_specificity",
        ] + [c for c in frame if c.startswith(("sensitivity__", "error__"))]:
            allrows.append(
                {
                    "analysis": "internal",
                    "model": name,
                    "K": 0,
                    "metric": metric,
                    **mean_ci(g[metric]),
                }
            )
    pd.DataFrame(allrows).to_csv(output_dir(cfg) / "internal_summary.csv", index=False)
    robustness = []
    for path in (output_dir(cfg) / "runs").glob("seed_*/robustness/*/metrics.json"):
        for r in read_json(path):
            robustness.append(r)
    if robustness:
        rb = pd.DataFrame(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k
                    in [
                        "seed",
                        "condition",
                        "value",
                        "accuracy",
                        "macro_f1",
                        "macro_recall",
                        "macro_precision",
                    ]
                }
                for r in robustness
            ]
        )
        rb.to_csv(output_dir(cfg) / "robustness_runs.csv", index=False)
        rr = []
        for (condition, value), g in rb.groupby(["condition", "value"], dropna=False):
            for metric in ["accuracy", "macro_f1", "macro_recall", "macro_precision"]:
                rr.append(
                    {"condition": condition, "value": value, "metric": metric, **mean_ci(g[metric])}
                )
        pd.DataFrame(rr).to_csv(output_dir(cfg) / "robustness_summary.csv", index=False)
    return allrows


def paired_summary(cfg, a, b, metric="accuracy"):
    frame = pd.read_csv(output_dir(cfg) / "internal_runs.csv")
    aa = frame[frame.name.eq(a)].set_index("seed")
    bb = frame[frame.name.eq(b)].set_index("seed")
    if set(aa.index) != set(bb.index) or aa.index.duplicated().any() or bb.index.duplicated().any():
        raise ValueError(
            "Paired comparisons require exactly the same outer seeds, one outcome per model/seed."
        )
    differences = aa[metric].sort_index() - bb[metric].sort_index()
    result = {
        "model_a": a,
        "model_b": b,
        "metric": metric,
        "unit": "paired outer-split differences",
        **mean_ci(differences),
    }
    save_json(output_dir(cfg) / "paired_comparison.json", result)
    return result
