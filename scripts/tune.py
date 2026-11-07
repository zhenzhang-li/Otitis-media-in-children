"""Optional selection-only grid search. This is an extension, not a paper rerun."""

from copy import deepcopy
import argparse, shutil, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import ParameterGrid
import yaml
from otoscopy_ai.config import load_config, output_dir, path_for, save_json, public_config
from otoscopy_ai.data import prepare, split
from otoscopy_ai.engine import cache_features, cache_root, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--kind", choices=["pooled", "tokens"], default="tokens")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    base = load_config(args.config)
    root = output_dir(base)
    prepare(base)
    split(base)
    cache_features(base, args.encoder)
    space = yaml.safe_load(Path(args.space).read_text("utf-8"))["grid"]
    results = []
    import torch

    torch.set_num_threads(int(base.get("cpu_threads", 4)))
    for trial, params in enumerate(ParameterGrid(space)):
        cfg = deepcopy(base)
        for name, value in params.items():
            parts = name.split(".")
            node = cfg
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        if (
            cfg["preprocessing"] != base["preprocessing"]
            or cfg["fusion"]["layers"] != base["fusion"]["layers"]
        ):
            raise ValueError(
                "This search shares frozen features; use a separate output/cache for different preprocessing or fusion token layers."
            )
        trial_root = root / f"tuning/trial_{trial:03d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        cfg["output"] = str(trial_root)
        cfg.setdefault("cache", {})["directory"] = str(cache_root(base))
        for name in ["records.csv", "classes.json"]:
            shutil.copy2(root / name, trial_root / name)
        shutil.copytree(root / "splits", trial_root / "splits", dirs_exist_ok=True)
        scores = []
        for seed in [args.seed] if args.seed is not None else base["split"]["seeds"]:
            r = fit(
                cfg,
                int(seed),
                {
                    "name": "tuned_head",
                    "encoder": args.encoder,
                    "kind": args.kind,
                    "section": "tuning",
                },
            )
            scores.append(r["best_selection_macro_f1"])
        row = {
            "trial": trial,
            "mean_selection_macro_f1": sum(scores) / len(scores),
            "outer_seeds": len(scores),
            **params,
        }
        results.append(row)
        save_json(trial_root / "parameters.json", row)
    table = pd.DataFrame(results).sort_values("mean_selection_macro_f1", ascending=False)
    table.to_csv(root / "tuning/selection_scores.csv", index=False)
    save_json(root / "tuning/best_parameters.json", table.iloc[0].to_dict())
    print(table.to_string(index=False))
    print("No calibration, internal evaluation or external labels were used to rank trials.")


if __name__ == "__main__":
    main()
