"""One command interface; run `otoscopy-ai --help` for all stages."""

import argparse, json
from .config import load_config, save_json, output_dir, public_config


def main():
    parser = argparse.ArgumentParser(
        description="Portable frozen-backbone and LayerFusion study workflows"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override any YAML setting; use a new output directory for a new experiment.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    for cmd in ["prepare", "split", "cache", "inspect", "select-model", "summarize", "plot", "run"]:
        subs.add_parser(cmd)
    tr = subs.add_parser("train")
    tr.add_argument("--section", choices=["backbones", "layers", "fewshot", "all"], default="all")
    tr.add_argument("--seed", type=int)
    ev = subs.add_parser("evaluate")
    ev.add_argument("--all-internal", action="store_true")
    ev.add_argument("--selected", action="store_true")
    ev.add_argument("--seed", type=int)
    ev.add_argument("--name")
    cal = subs.add_parser("calibrate")
    cal.add_argument("--seed", required=True, type=int)
    cal.add_argument("--name", required=True)
    for cmd in ["robustness", "explain"]:
        sub = subs.add_parser(cmd)
        sub.add_argument("--seed", type=int)
        sub.add_argument("--name")
        sub.add_argument("--all-seeds", action="store_true")
        if cmd == "explain":
            sub.add_argument("--ids", nargs="*")
    pair = subs.add_parser("compare")
    pair.add_argument("--a", required=True)
    pair.add_argument("--b", required=True)
    pair.add_argument("--metric", default="accuracy")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out = output_dir(cfg)
    import torch

    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    from .data import prepare, split, read_records
    from .engine import cache_features, calibrate, evaluate_head
    from .workflows import (
        train_section,
        internal_evaluation,
        select_checkpoint,
        selected_evaluation,
        summarize,
        paired_summary,
    )

    if args.command in ["prepare", "run"]:
        prepare(cfg)
    if args.command in ["split", "run"]:
        split(cfg)
    if args.command in ["cache", "run"]:
        keys = set(
            cfg["experiments"]["backbones"]
            + cfg["experiments"]["layer_encoders"]
            + [x["encoder"] for x in cfg["fewshot"]["models"]]
        )
        for key in sorted(keys):
            cache_features(cfg, key)
    if args.command == "inspect":
        rows, classes = read_records(cfg)
        print(
            json.dumps(
                {
                    "records": len(rows),
                    "classes": classes,
                    "cohorts": rows.cohort.value_counts().to_dict(),
                    "models": list(cfg["models"]),
                    "split_seeds": cfg["split"]["seeds"],
                    "fractions": cfg["split"]["fractions"],
                    "configuration": public_config(cfg),
                },
                indent=2,
            )
        )
        return
    if args.command in ["train", "run"]:
        section = getattr(args, "section", "all")
        seeds = [args.seed] if getattr(args, "seed", None) is not None else None
        for stage in ["backbones", "layers", "fewshot"] if section == "all" else [section]:
            train_section(cfg, stage, seeds)
    if args.command == "calibrate":
        calibrate(cfg, args.seed, args.name)
    if args.command == "select-model":
        print(json.dumps(select_checkpoint(cfg), indent=2))
    if args.command == "evaluate":
        if args.all_internal:
            internal_evaluation(cfg)
        elif args.selected:
            selected_evaluation(cfg)
        elif args.seed is not None and args.name:
            evaluate_head(cfg, args.seed, args.name)
        else:
            parser.error("evaluate needs --all-internal, --selected, or both --seed and --name.")
    if args.command == "run":
        internal_evaluation(cfg)
        selected_evaluation(cfg)
    if args.command in ["robustness", "explain"]:
        from . import robustness, explain

        if args.all_seeds:
            targets = [
                (int(s), args.name or cfg["evaluation"]["primary_head"])
                for s in cfg["split"]["seeds"]
            ]
        elif args.seed is not None:
            targets = [(args.seed, args.name or cfg["evaluation"]["primary_head"])]
        else:
            selected = select_checkpoint(cfg)
            targets = [(selected["seed"], selected["name"])]
        for seed, name in targets:
            if args.command == "robustness":
                robustness.run(cfg, seed, name)
            else:
                explain.run(cfg, seed, name, args.ids)
    if args.command in ["summarize", "run"]:
        summarize(cfg)
    if args.command == "compare":
        print(json.dumps(paired_summary(cfg, args.a, args.b, args.metric), indent=2))
    if args.command in ["plot", "run"]:
        from .plotting import plot_all

        plot_all(cfg)
    save_json(out / "resolved_config.json", public_config(cfg))


if __name__ == "__main__":
    main()
