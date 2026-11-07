"""Discrete region retention/replacement and exact layer Shapley games.

Explanations target the original predicted-vs-runner-up logit margin. Regions are
feature clusters, not validated lesions. No reference diagnosis chooses the target.
"""

import itertools, math
import numpy as np
import torch
from PIL import ImageFilter
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from .config import output_dir, save_json, read_json
from .data import manifest, load_image, crop_transform, normalize_transform
from .engine import restore, amp_context, to_features
from .models import build_encoder


def shapley(values):
    values = np.asarray(values, float)
    n = int(round(np.log2(len(values))))
    if len(values) != 2**n:
        raise ValueError("Exact Shapley requires all 2^n coalition values.")
    result = np.zeros((n, *values.shape[1:]), float)
    for j in range(n):
        for subset in range(2**n):
            if subset & (1 << j):
                continue
            k = subset.bit_count()
            result[j] += (
                math.factorial(k)
                * math.factorial(n - k - 1)
                / math.factorial(n)
                * (values[subset | (1 << j)] - values[subset])
            )
    return result


@torch.no_grad()
def run(cfg, seed, name, ids=None):
    head, ck, _, device, _, _ = restore(cfg, seed, name)
    if ck["spec"]["kind"] != "tokens":
        raise ValueError("Region/layer analysis requires a LayerFusion token head.")
    enc = build_encoder(cfg, ck["spec"]["encoder"]).to(device).eval()
    rows = manifest(cfg, seed)
    settings = cfg["explain"]
    regions = int(settings.get("regions", 6))
    max_regions = int(settings.get("max_regions", 8))
    if regions < 2 or regions > max_regions:
        raise ValueError("Region count exceeds the explicitly configured exhaustive-search budget.")
    cases = (
        rows[rows.id.isin(ids)]
        if ids
        else rows[rows.role.eq("evaluation")]
        .sort_values("id")
        .head(int(settings.get("max_cases", 6)))
    )
    if ids and set(ids) != set(cases.id):
        raise ValueError("One or more requested case IDs are absent.")
    root = output_dir(cfg) / f"runs/seed_{seed}/explanations/{name}"
    root.mkdir(parents=True, exist_ok=True)
    crop = crop_transform(cfg)
    norm = normalize_transform(cfg)
    batch_size = int(settings.get("batch_size", 16))
    records = []
    dtype = cfg.get("cache", {}).get("dtype", "float16")
    nlayers = len(cfg["fusion"]["layers"])
    if nlayers > int(settings.get("max_layers", 6)):
        raise ValueError(
            "Exact layer Shapley exceeds explain.max_layers; explicitly increase the budget or use fewer fusion layers."
        )
    calpath = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}/calibration.json"
    temperature = read_json(calpath)["temperature"] if calpath.exists() else 1.0

    def encode(images):
        outputs = []
        for start in range(0, len(images), batch_size):
            with amp_context(cfg, device):
                v = enc(images[start : start + batch_size].to(device))["tokens"]
            outputs.append(
                to_features(cfg, v.float().cpu().numpy().astype(dtype), ck["spec"], device)
            )
        return torch.cat(outputs)

    def logits(tokens):
        values = []
        for start in range(0, len(tokens), batch_size):
            with amp_context(cfg, device):
                v = head(tokens[start : start + batch_size])
            values.append(v.float().cpu().numpy())
        return np.concatenate(values)

    for _, row in cases.iterrows():
        image = crop(load_image(row))
        x = norm(image)[None].to(device)
        full = encode(x)
        z = logits(full)[0]
        pred = int(z.argmax())
        alt = int(np.argsort(z)[-2])
        full_margin = float(z[pred] - z[alt])
        patches = full.shape[2] - 1
        side = int(round(np.sqrt(patches)))
        if side * side != patches:
            raise ValueError(
                "The supplied dense encoder must expose a square patch grid; implement a grid adapter for other token layouts."
            )
        features = (
            torch.nn.functional.normalize(full[0, :, 1:].float(), dim=-1)
            .permute(1, 0, 2)
            .reshape(patches, -1)
            .cpu()
            .numpy()
        )
        labels = (
            AgglomerativeClustering(
                n_clusters=regions, linkage="ward", connectivity=grid_to_graph(side, side)
            )
            .fit_predict(features)
            .reshape(side, side)
        )
        bits = np.array([[(s >> j) & 1 for j in range(regions)] for s in range(2**regions)], bool)
        small = bits[:, labels]
        masks = torch.tensor(small[:, None].astype(np.float32), device=device)
        masks = torch.nn.functional.interpolate(masks, size=x.shape[-2:], mode="nearest")
        area = masks.mean((1, 2, 3)).cpu().numpy()
        p = cfg["preprocessing"]
        mean = torch.tensor(p.get("mean", [0.485, 0.456, 0.406]), device=device)[
            None, :, None, None
        ]
        std = torch.tensor(p.get("std", [0.229, 0.224, 0.225]), device=device)[None, :, None, None]
        grey = (torch.full_like(x, 0.5) - mean) / std
        blur = norm(image.filter(ImageFilter.GaussianBlur(float(settings.get("blur_radius", 9)))))[
            None
        ].to(device)
        encoded = []
        allz = []
        for reference in [grey, blur]:
            tok = encode(x * masks + reference * (1 - masks))
            encoded.append(tok)
            allz.append(logits(tok))
        zz = np.stack(allz)
        margin = zz[:, :, pred] - zz[:, :, alt]
        endpoint = margin[:, -1]
        keep = (margin >= float(settings.get("retain_fraction", 0.9)) * endpoint[:, None]).all(
            0
        ) & (zz.argmax(-1) == pred).all(0)
        removed = np.arange(2**regions - 1, -1, -1)
        delete = (
            margin[:, removed] <= float(settings.get("remove_fraction", 0.5)) * endpoint[:, None]
        ).all(0)
        eligible = np.flatnonzero(keep & delete)
        dual = bool(len(eligible))
        if not dual:
            eligible = np.flatnonzero(keep)
        if not len(eligible):
            raise RuntimeError(
                "Even the full input failed retention; inspect precision/preprocessing equivalence."
            )
        chosen = min(eligible, key=lambda s: (area[s], -margin[:, s].min(), s))
        complement = (2**regions - 1) ^ chosen
        layer_values = []
        for tok, lg in zip(encoded, allz):
            games = torch.stack(
                [
                    torch.stack(
                        [tok[-1, j] if (s >> j) & 1 else tok[complement, j] for j in range(nlayers)]
                    )
                    for s in range(2**nlayers)
                ]
            )
            v = logits(games)
            v[0] = lg[complement]
            v[-1] = lg[-1]
            layer_values.append(shapley(v[:, pred] - v[:, alt]))
        regional = np.stack([shapley(v[:, pred] - v[:, alt]) for v in allz])
        conditional = None
        if settings.get("full_region_layer_decomposition", False):
            alloc = np.zeros((2, regions, nlayers), float)
            for ri, tok in enumerate(encoded):
                for region in range(regions):
                    for context in range(2**regions):
                        if context & (1 << region):
                            continue
                        after = context | (1 << region)
                        games = torch.stack(
                            [
                                torch.stack(
                                    [
                                        tok[after, j] if (s >> j) & 1 else tok[context, j]
                                        for j in range(nlayers)
                                    ]
                                )
                                for s in range(2**nlayers)
                            ]
                        )
                        v = logits(games)
                        v[0] = allz[ri][context]
                        v[-1] = allz[ri][after]
                        k = context.bit_count()
                        weight = (
                            math.factorial(k)
                            * math.factorial(regions - k - 1)
                            / math.factorial(regions)
                        )
                        alloc[ri, region] += weight * shapley(v[:, pred] - v[:, alt])
            if not np.allclose(alloc.sum(-1), regional, atol=1e-4):
                raise RuntimeError("Region/layer allocation failed Shapley additivity.")
            conditional = alloc
        prob = torch.softmax(torch.tensor(zz) / temperature, -1).numpy()
        lp = np.stack(layer_values)
        record = {
            "id": str(row.id),
            "reference": str(row.label),
            "prediction": ck["classes"][pred],
            "alternative": ck["classes"][alt],
            "selected_subset": int(chosen),
            "selected_area": float(area[chosen]),
            "dual_feasible": dual,
            "retention_only_fallback": not dual,
            "prediction_tie_margin": full_margin,
            "retain_probability": prob[:, chosen, pred].tolist(),
            "replace_probability": prob[:, complement, pred].tolist(),
            "layer_contribution_mean": lp.mean(0).tolist(),
            "layer_contribution_reference_min": lp.min(0).tolist(),
            "layer_contribution_reference_max": lp.max(0).tolist(),
            "construction_references": ["RGB 0.5", "same-image Gaussian blur"],
            "range_is_not_statistical_ci": True,
            "clinical_lesion_validation": False,
            "checkpoint_signature": ck["signature"],
        }
        name_safe = __import__("hashlib").sha256(str(row.id).encode()).hexdigest()[:16]
        values = {
            "region_labels": labels,
            "selected_mask": masks[chosen, 0].cpu().numpy(),
            "input_rgb": np.asarray(image),
            "coalition_logits": zz,
            "regional_shapley": regional,
            "layer_shapley": lp,
        }
        if conditional is not None:
            values["region_layer_allocation"] = conditional
        np.savez_compressed(root / f"{name_safe}.npz", **values)
        save_json(root / f"{name_safe}.json", record)
        records.append(record)
    save_json(root / "cases.json", records)
    return records
