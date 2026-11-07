"""Publication-style plots controlled by one configuration section."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import output_dir, read_json


def style(cfg):
    p = cfg.get("plot", {})
    plt.rcParams.update(
        {
            "font.family": p.get("font", "Arial"),
            "font.size": p.get("font_size", 8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": p.get("axis_width", 0.7),
            "lines.linewidth": p.get("line_width", 1.2),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return p


def save(fig, cfg, name):
    p = cfg.get("plot", {})
    out = output_dir(cfg) / "figures"
    out.mkdir(exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(out / (name + suffix), dpi=p.get("dpi", 600), facecolor="white")
    plt.close(fig)


def plot_all(cfg):
    p = style(cfg)
    out = output_dir(cfg)
    width = p.get("width_mm", 190) / 25.4
    colors = p.get("colors", ["#328985", "#8395A3", "#B14E63", "#4777A8", "#AB8440"])
    size = p.get("marker_size", 4)
    caps = p.get("cap_size", 2)
    if (out / "internal_summary.csv").exists():
        s = pd.read_csv(out / "internal_summary.csv")
        bones = s[s.model.str.startswith("backbone_") & s.metric.eq("accuracy")].sort_values("mean")
        if len(bones):
            fig, ax = plt.subplots(figsize=(width, max(3, len(bones) * 0.3)))
            y = np.arange(len(bones))
            val = bones["mean"].to_numpy() * 100
            lo = bones.ci_low.to_numpy() * 100
            hi = bones.ci_high.to_numpy() * 100
            ax.errorbar(
                val,
                y,
                xerr=np.nan_to_num([val - lo, hi - val]),
                fmt="o",
                color=colors[0],
                capsize=caps,
                ms=size,
            )
            ax.set(
                yticks=y,
                yticklabels=bones.model.str.removeprefix("backbone_"),
                xlabel="Internal accuracy (%): outer-split mean and 95% t CI",
            )
            ax.grid(axis="x", alpha=0.15)
            fig.tight_layout()
            save(fig, cfg, "backbones")
        few = s[s.analysis.eq("fewshot")]
        if len(few):
            fig, axes = plt.subplots(1, 2, figsize=(width, p.get("learning_height_mm", 95) / 25.4))
            for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
                for i, (name, g) in enumerate(few[few.metric.eq(metric)].groupby("model")):
                    g = g.sort_values("K")
                    v = g["mean"].to_numpy() * 100
                    lo = g.ci_low.to_numpy() * 100
                    hi = g.ci_high.to_numpy() * 100
                    ax.errorbar(
                        g.K,
                        v,
                        yerr=np.nan_to_num([v - lo, hi - v]),
                        marker="o",
                        ms=size,
                        color=colors[i % len(colors)],
                        capsize=caps,
                        label=name,
                    )
                ax.set(
                    xlabel="Training images per class",
                    ylabel=metric.replace("_", " ").title() + " (%)",
                )
                ax.grid(alpha=0.15)
            axes[-1].legend(frameon=False, fontsize=p.get("legend_size", 6))
            fig.tight_layout()
            save(fig, cfg, "fewshot")
        layers = s[
            (s.analysis.eq("internal"))
            & s.metric.eq("accuracy")
            & ~s.model.str.startswith("backbone_")
        ]
        if len(layers):
            fig, ax = plt.subplots(figsize=(width, max(3, len(layers) * 0.14)))
            g = layers.sort_values("mean")
            v = g["mean"].to_numpy() * 100
            lo = g.ci_low.to_numpy() * 100
            hi = g.ci_high.to_numpy() * 100
            ax.errorbar(
                v,
                np.arange(len(g)),
                xerr=np.nan_to_num([v - lo, hi - v]),
                fmt="o",
                ms=3,
                color=colors[2],
                capsize=1,
            )
            ax.set(
                yticks=np.arange(len(g)),
                yticklabels=g.model,
                xlabel="Internal accuracy (%): outer-split mean and 95% t CI",
            )
            ax.tick_params(labelsize=6)
            fig.tight_layout()
            save(fig, cfg, "layers")
    for hist in (out / "runs").glob("seed_*/heads/*/history.csv"):
        h = pd.read_csv(hist)
        summary = read_json(hist.parent / "summary.json")
        fig, axes = plt.subplots(1, 2, figsize=(width, p.get("training_height_mm", 75) / 25.4))
        axes[0].plot(h.epoch, h.training_loss, color=colors[0])
        axes[0].set(xlabel="Executed epoch", ylabel="Training loss")
        axes[1].plot(h.epoch, h.selection_macro_f1 * 100, color=colors[2])
        axes[1].axvline(summary["selected_epoch"], ls="--", color=colors[1])
        axes[1].set(xlabel="Executed epoch", ylabel="Selection macro-F1 (%)")
        fig.suptitle(hist.parent.name, fontsize=9)
        fig.tight_layout()
        save(fig, cfg, f'training_{summary["seed"]}_{hist.parent.name}')
    for casefile in (out / "runs").glob("seed_*/explanations/*/cases.json"):
        for r in read_json(casefile):
            stem = hashlib.sha256(r["id"].encode()).hexdigest()[:16]
            data = np.load(casefile.parent / f"{stem}.npz")
            im = data["input_rgb"] / 255.0
            mask = data["selected_mask"]
            fig, axes = plt.subplots(1, 4, figsize=(width, 55 / 25.4))
            axes[0].imshow(im)
            axes[0].contour(mask, levels=[0.5], colors=[colors[0]], linewidths=0.8)
            axes[0].set_title("Selected regions")
            axes[1].imshow(im * mask[:, :, None] + 0.5 * (1 - mask[:, :, None]))
            axes[1].set_title("Retain")
            axes[2].imshow(im * (1 - mask[:, :, None]) + 0.5 * mask[:, :, None])
            axes[2].set_title("Replace")
            for ax in axes[:3]:
                ax.axis("off")
            v = np.array(r["layer_contribution_mean"])
            lo = np.array(r["layer_contribution_reference_min"])
            hi = np.array(r["layer_contribution_reference_max"])
            axes[3].barh(np.arange(len(v)), v, color=colors[0])
            axes[3].errorbar(
                v, np.arange(len(v)), xerr=[v - lo, hi - v], fmt="none", color=colors[1], capsize=2
            )
            axes[3].set(
                yticks=np.arange(len(v)),
                yticklabels=[f"L{x}" for x in cfg["fusion"]["layers"]],
                xlabel="Logit-margin change",
            )
            axes[3].set_title("Reference range, not CI")
            fig.tight_layout()
            save(fig, cfg, "explanation_" + stem)
    if (out / "robustness_summary.csv").exists():
        rb = pd.read_csv(out / "robustness_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(width, 90 / 25.4))
        for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
            for i, (kind, g) in enumerate(
                rb[rb.metric.eq(metric) & ~rb.condition.eq("original")].groupby("condition")
            ):
                g = g.sort_values("value")
                v = g["mean"].to_numpy() * 100
                ax.errorbar(
                    g.value,
                    v,
                    yerr=np.nan_to_num(
                        [v - g.ci_low.to_numpy() * 100, g.ci_high.to_numpy() * 100 - v]
                    ),
                    color=colors[i % len(colors)],
                    marker="o",
                    capsize=caps,
                    label=kind,
                )
            ax.set(
                xlabel="Perturbation value (source-image units)",
                ylabel=metric.replace("_", " ") + " (%)",
            )
            ax.legend(frameon=False, fontsize=6)
        fig.tight_layout()
        save(fig, cfg, "robustness")
    if (out / "selected_model.json").exists():
        chosen = read_json(out / "selected_model.json")
        root = out / f'runs/seed_{chosen["seed"]}/heads/{chosen["name"]}'
        metrics = []
        for path in root.glob("metrics_*.json"):
            metrics.append(read_json(path))
        if metrics:
            fig, ax = plt.subplots(figsize=(width, 80 / 25.4))
            for i, r in enumerate(metrics):
                val = r["accuracy"] * 100
                lo, hi = np.array(r["accuracy_ci"]) * 100
                ax.errorbar(
                    val, i, xerr=[[val - lo], [hi - val]], fmt="o", color=colors[0], capsize=caps
                )
            ax.set(
                yticks=np.arange(len(metrics)),
                yticklabels=[r["cohort"] for r in metrics],
                xlabel="Selected fixed model: accuracy (%) and image-level Wilson 95% CI",
            )
            fig.tight_layout()
            save(fig, cfg, "selected_model_evaluation")
