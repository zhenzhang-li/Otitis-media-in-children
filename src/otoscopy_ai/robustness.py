"""Separate source-image perturbations with the same encoder, head and calibration."""

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from .config import output_dir, save_json, read_json
from .data import manifest, Images
from .models import build_encoder
from .engine import restore, amp_context, prediction_frame, to_features
from .metrics import evaluate


def perturb(image, kind, value):
    if kind == "original":
        return image
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(float(value)))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value))
    if kind == "saturation":
        return ImageEnhance.Color(image).enhance(float(value))
    if kind == "rotation":
        return image.rotate(
            float(value), resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0)
        )
    raise ValueError(f"Unknown perturbation {kind}.")


def run(cfg, seed, name, cohort=None):
    head, ck, arrays, device, mu, std = restore(cfg, seed, name)
    encoder = build_encoder(cfg, ck["spec"]["encoder"]).to(device).eval()
    rows = manifest(cfg, seed)
    rows = rows[rows.cohort.eq(cohort)] if cohort else rows[rows.role.eq("evaluation")]
    root = output_dir(cfg) / f"runs/seed_{seed}/robustness/{name}"
    root.mkdir(parents=True, exist_ok=True)
    calpath = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}/calibration.json"
    cal = read_json(calpath) if calpath.exists() else {"temperature": 1.0}
    results = []
    conditions = [("original", 1.0)] + [
        (kind, float(value))
        for kind, values in cfg["robustness"]["conditions"].items()
        for value in values
    ]
    for kind, value in conditions:
        transform = lambda im: perturb(im, kind, value)
        loader = torch.utils.data.DataLoader(
            Images(cfg, rows, transform),
            batch_size=cfg["preprocessing"].get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )
        logits = []
        with torch.no_grad():
            for x, _ in loader:
                with amp_context(cfg, device):
                    features = encoder(x.to(device))
                spec = ck["spec"]
                v = (
                    features[spec["kind"]]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(cfg.get("cache", {}).get("dtype", "float16"))
                )
                if spec["kind"] == "cls":
                    v = v[:, [j - 1 for j in spec["layers"]]].astype(np.float32).mean(1)
                    if spec.get("pad_to"):
                        v = np.pad(v, ((0, 0), (0, spec["pad_to"] - v.shape[-1])))
                batch = to_features(cfg, v, spec, device, mu, std)
                with amp_context(cfg, device):
                    logits.append(head(batch).float().cpu().numpy())
        lg = np.concatenate(logits)
        pred = prediction_frame(
            rows, lg, ck["classes"], cal["temperature"], cal.get("entropy_threshold")
        )
        pred.to_csv(root / f"predictions_{kind}_{value:g}.csv", index=False)
        r = evaluate(
            rows.label_idx,
            lg,
            ck["classes"],
            cal["temperature"],
            0,
            seed,
            cfg.get("evaluation", {}).get("clinical_pair"),
        )
        r.update(
            seed=seed, name=name, condition=kind, value=value, checkpoint_signature=ck["signature"]
        )
        results.append(r)
    save_json(root / "metrics.json", results)
    # Original images must agree with cached inference; this catches transform/precision drift.
    old = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}/predictions_internal_evaluation.csv"
    if old.exists() and cohort is None:
        reference = pd.read_csv(old)
        base = pd.read_csv(root / "predictions_original_1.csv")
        joined = reference[["id", "prediction_idx"]].merge(
            base[["id", "prediction_idx"]],
            on="id",
            suffixes=("_cached", "_fresh"),
            validate="one_to_one",
        )
        changes = int(joined.prediction_idx_cached.ne(joined.prediction_idx_fresh).sum())
        save_json(
            root / "baseline_equivalence.json",
            {"n": len(joined), "prediction_changes": changes, "same_checkpoint": True},
        )
        if changes:
            raise RuntimeError(
                f"Original-image predictions differ for {changes} records; inspect batch/AMP/preprocessing before interpreting perturbations."
            )
    return results
