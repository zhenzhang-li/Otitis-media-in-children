"""Frozen-feature caching, shared head training, calibration and fixed inference."""

from contextlib import nullcontext
from pathlib import Path
import json, time, re
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from .config import (
    output_dir,
    save_json,
    read_json,
    seed_everything,
    device_for,
    digest,
    public_config,
    path_for,
    file_sha,
)
from .data import read_records, manifest, Images
from .models import build_encoder, encoder_identity
from .fusion import LayerFusion, loss_with_aux
from .metrics import evaluate, from_confusion


def amp_context(cfg, device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(cfg["training"].get("amp", True) and device.type == "cuda"),
    )


def cache_root(cfg):
    return (
        path_for(cfg, cfg["cache"]["directory"])
        if cfg.get("cache", {}).get("directory")
        else output_dir(cfg) / "cache"
    )


def cache_features(cfg, key):
    rows, classes = read_records(cfg)
    identity = encoder_identity(cfg, key)
    signature = digest(
        {
            "rows": rows[["id", "path", "content_hash"]].to_dict("records"),
            "encoder": identity,
            "dtype": cfg.get("cache", {}).get("dtype", "float16"),
        }
    )
    out = cache_root(cfg) / key
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "metadata.json"
    if meta.exists():
        old = read_json(meta)
        if old["signature"] != signature:
            raise RuntimeError(
                f"{key}: cache inputs/config changed. Choose a fresh output/cache directory or remove this model cache explicitly."
            )
        return old
    device = device_for(cfg)
    seed_everything(int(cfg["training"].get("seed", 0)))
    enc = build_encoder(cfg, key).to(device).eval()
    loader = torch.utils.data.DataLoader(
        Images(cfg, rows),
        batch_size=cfg["preprocessing"].get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("workers", 0),
        pin_memory=device.type == "cuda",
    )
    buffers = {}
    start = time.time()
    with torch.no_grad():
        for x, indices in loader:
            with amp_context(cfg, device):
                values = enc(x.to(device))
            for kind, value in values.items():
                value = (
                    value.float().cpu().numpy().astype(cfg.get("cache", {}).get("dtype", "float16"))
                )
                if kind not in buffers:
                    buffers[kind] = np.lib.format.open_memmap(
                        out / f"{kind}.npy",
                        mode="w+",
                        dtype=value.dtype,
                        shape=(len(rows), *value.shape[1:]),
                    )
                buffers[kind][indices.numpy()] = value
    for a in buffers.values():
        a.flush()
    result = {
        "model": key,
        "signature": signature,
        "n": len(rows),
        "ids": rows.id.tolist(),
        "shapes": {k: list(v.shape) for k, v in buffers.items()},
        "encoder": identity,
        "preprocessing": cfg["preprocessing"],
        "dtype": cfg.get("cache", {}).get("dtype", "float16"),
        "seconds": time.time() - start,
        "samplewise_cache_no_fitted_statistics": True,
        "frozen_encoder": True,
    }
    save_json(meta, result)
    del enc, buffers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cache_for(cfg, key):
    meta = cache_features(cfg, key)
    root = cache_root(cfg) / key
    return {kind: np.load(root / f"{kind}.npy", mmap_mode="r") for kind in meta["shapes"]}, meta


def feature_numpy(arrays, spec, indices):
    value = np.array(arrays[spec["kind"]][np.asarray(indices, int)], copy=True)
    if spec["kind"] == "cls":
        value = value[:, [i - 1 for i in spec["layers"]]].astype(np.float32).mean(1)
        dim = spec.get("pad_to")
        if dim:
            if dim < value.shape[-1]:
                raise ValueError("pad_to cannot reduce the feature dimension.")
            value = np.pad(value, ((0, 0), (0, dim - value.shape[-1])))
    return value


def to_features(cfg, value, spec, device, mu=None, std=None):
    x = torch.from_numpy(np.asarray(value)).to(device)
    if spec["kind"] != "tokens":
        x = x.float()
        x = (x - mu) / std if mu is not None else x
    elif device.type != "cuda" or not cfg["training"].get("amp", True):
        x = x.float()
    return x


def construct_head(cfg, spec, dim, nclasses):
    if spec["kind"] == "tokens":
        return LayerFusion(
            dim,
            nclasses,
            len(cfg["fusion"]["layers"]),
            cfg["fusion"].get("heads", 8),
            cfg["fusion"].get("dropout", 0.1),
        )
    return nn.Linear(dim, nclasses)


@torch.no_grad()
def predict(cfg, model, arrays, spec, rows, device, mu=None, std=None):
    model.eval()
    outputs = []
    for start in range(0, len(rows), cfg["training"]["batch_size"]):
        ids = rows["index"].iloc[start : start + cfg["training"]["batch_size"]].to_numpy(int)
        x = to_features(cfg, feature_numpy(arrays, spec, ids), spec, device, mu, std)
        with amp_context(cfg, device):
            logits = model(x)
        outputs.append(logits.float().cpu().numpy())
    if not outputs:
        raise ValueError("Prediction list is empty.")
    return np.concatenate(outputs)


def fit(cfg, seed, spec, training=None):
    rows = manifest(cfg, seed)
    _, classes = read_records(cfg)
    tr = rows[rows.role.eq("train")] if training is None else training
    sel = rows[rows.role.eq("selection")]
    allowed = set(rows.loc[rows.role.eq("train"), "id"])
    if not set(tr.id).issubset(allowed):
        raise ValueError("Training samples must belong exclusively to the training role.")
    arrays, meta = cache_for(cfg, spec["encoder"])
    settings = cfg["training"]
    head_seed = int(spec.get("initialization_seed", settings.get("seed", 0)))
    seed_everything(head_seed)
    device = device_for(cfg)
    signature = digest(
        {
            "cache": meta["signature"],
            "train_ids": tr.id.tolist(),
            "selection_ids": sel.id.tolist(),
            "settings": settings,
            "fusion": cfg["fusion"],
            "spec": spec,
            "classes": classes,
            "seed": seed,
            "training_source_sha256": file_sha(Path(__file__)),
            "head_source_sha256": file_sha(Path(__file__).with_name("fusion.py")),
        }
    )
    root = output_dir(cfg) / f"runs/seed_{seed}/heads" / spec["name"]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "checkpoint.pt").exists():
        old = read_json(root / "summary.json")
        if old["signature"] != signature:
            raise RuntimeError(
                f"{root.name}: incompatible checkpoint/config. Use a fresh output experiment directory."
            )
        return old
    train_values = feature_numpy(arrays, spec, tr["index"].to_numpy())
    dim = train_values.shape[-1]
    mu = std = None
    if spec["kind"] != "tokens":
        fv = torch.from_numpy(train_values).float()
        mu = fv.mean(0, keepdim=True).to(device)
        std = fv.std(0, keepdim=True, unbiased=False).clamp_min(1e-4).to(device)
    model = construct_head(cfg, spec, dim, len(classes)).to(device)
    y = torch.tensor(tr.label_idx.to_numpy(int), device=device)
    sy = sel.label_idx.to_numpy(int)
    preload = settings.get("feature_storage", "device") == "device"
    xx = to_features(cfg, train_values, spec, device, mu, std) if preload else None
    del train_values
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
        fused=bool(device.type == "cuda" and settings.get("fused_optimizer", True)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["max_epochs"])
    )
    enabled = bool(settings.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=enabled)
    best = -np.inf
    early = -np.inf
    best_epoch = 0
    stale = 0
    history = []
    state = None
    start = time.time()
    nclasses = len(classes)
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        order = torch.randperm(len(tr), device=device)
        loss_total = 0.0
        for offset in range(0, len(tr), settings["batch_size"]):
            ids = order[offset : offset + settings["batch_size"]]
            if preload:
                x = xx[ids]
            else:
                ids_numpy = ids.cpu().numpy()
                ix = tr["index"].iloc[ids_numpy].to_numpy(int)
                x = to_features(cfg, feature_numpy(arrays, spec, ix), spec, device, mu, std)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(cfg, device):
                loss = (
                    loss_with_aux(model, x, y[ids], cfg["fusion"].get("aux_losses", {}))
                    if spec["kind"] == "tokens"
                    else nn.functional.cross_entropy(model(x), y[ids])
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f'Nonfinite loss for {spec["name"]}; inspect feature normalization/lr.'
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(ids)
        scheduler.step()
        lg = predict(cfg, model, arrays, spec, sel, device, mu, std)
        cm = np.bincount(sy * nclasses + lg.argmax(1), minlength=nclasses * nclasses).reshape(
            nclasses, nclasses
        )
        score = float(from_confusion(cm)["macro_f1"])
        acc = float(np.trace(cm) / cm.sum())
        if score > best + 1e-9:
            best = score
            best_epoch = epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if score > early + float(settings.get("min_delta", 1e-4)):
            early = score
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_total / len(tr),
                "selection_macro_f1": score,
                "selection_accuracy": acc,
                "lr": epoch_lr,
                "next_lr": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch == 1 or epoch % int(settings.get("log_every", 20)) == 0:
            print(
                f'{seed} {spec["name"]}: epoch {epoch}, selection F1={score:.5f}, best={best:.5f}@{best_epoch}',
                flush=True,
            )
        if epoch >= int(settings.get("min_epochs", 30)) and stale >= int(
            settings.get("patience", 30)
        ):
            break
    checkpoint = {
        "state_dict": state,
        "spec": spec,
        "dim": int(dim),
        "nclasses": nclasses,
        "classes": classes,
        "mu": mu.cpu() if mu is not None else None,
        "std": std.cpu() if std is not None else None,
        "seed": int(seed),
        "head_seed": head_seed,
        "best_epoch": best_epoch,
        "signature": signature,
        "cache_signature": meta["signature"],
        "train_ids": tr.id.tolist(),
        "selection_ids": sel.id.tolist(),
        "fusion_config": cfg["fusion"],
        "preprocessing": cfg["preprocessing"],
    }
    torch.save(checkpoint, root / "checkpoint.pt")
    pd.DataFrame(history).to_csv(root / "history.csv", index=False)
    model.load_state_dict(state)
    lg = predict(cfg, model, arrays, spec, sel, device, mu, std)
    prediction_frame(sel, lg, classes).to_csv(root / "selection_predictions.csv", index=False)
    summary = {
        "name": spec["name"],
        "seed": int(seed),
        "spec": spec,
        "signature": signature,
        "cache_signature": meta["signature"],
        "training_n": len(tr),
        "selection_n": len(sel),
        "epochs_run": epoch,
        "selected_epoch": best_epoch,
        "best_selection_macro_f1": best,
        "max_epochs": int(settings["max_epochs"]),
        "head_initialization_seed": head_seed,
        "seconds": time.time() - start,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    save_json(root / "summary.json", summary)
    del model, xx, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def restore(cfg, seed, name):
    path = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}/checkpoint.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    _, classes = read_records(cfg)
    if (
        classes != ck["classes"]
        or cfg["fusion"] != ck["fusion_config"]
        or cfg["preprocessing"] != ck["preprocessing"]
    ):
        raise RuntimeError(
            "Checkpoint class/fusion/preprocessing configuration differs; use its saved configuration."
        )
    arrays, meta = cache_for(cfg, ck["spec"]["encoder"])
    if meta["signature"] != ck["cache_signature"]:
        raise RuntimeError("Feature cache differs from checkpoint provenance.")
    device = device_for(cfg)
    model = construct_head(cfg, ck["spec"], ck["dim"], ck["nclasses"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    mu = ck["mu"].to(device) if ck["mu"] is not None else None
    std = ck["std"].to(device) if ck["std"] is not None else None
    return model, ck, arrays, device, mu, std


def prediction_frame(rows, logits, classes, temperature=1.0, threshold=None):
    df = rows[["id", "group", "cohort", "role", "label", "label_idx"]].copy()
    prob = softmax(logits.astype(float) / temperature, 1)
    df["prediction_idx"] = prob.argmax(1)
    df["prediction"] = [classes[i] for i in prob.argmax(1)]
    entropy = -(prob * np.log(np.maximum(prob, 1e-15))).sum(1)
    df["entropy"] = entropy
    for i in range(len(classes)):
        df[f"logit_{i}"] = logits[:, i]
        df[f"probability_{i}"] = prob[:, i]
    if threshold is not None:
        df["review_flag"] = entropy > threshold
    return df


def calibrate(cfg, seed, name):
    model, ck, arrays, device, mu, std = restore(cfg, seed, name)
    rows = manifest(cfg, seed)
    cal = rows[rows.role.eq("calibration")]
    lg = predict(cfg, model, arrays, ck["spec"], cal, device, mu, std).astype(float)
    y = cal.label_idx.to_numpy(int)

    def nll(logt):
        p = softmax(lg / np.exp(logt), 1)
        return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean())

    temp = float(
        np.exp(
            minimize_scalar(
                nll,
                bounds=tuple(cfg.get("calibration", {}).get("log_temperature_bounds", [-2.3, 3.0])),
                method="bounded",
            ).x
        )
    )
    p = softmax(lg / temp, 1)
    ent = -(p * np.log(np.maximum(p, 1e-15))).sum(1)
    target = cfg.get("calibration", {}).get("target_coverage", 0.8)
    threshold = float(np.quantile(ent, target, method="lower"))
    r = {
        "seed": seed,
        "name": name,
        "signature": ck["signature"],
        "temperature": temp,
        "entropy_threshold": threshold,
        "target_coverage": target,
        "n": len(cal),
        "calibration_ids": cal.id.tolist(),
        "retained_n": int((ent <= threshold).sum()),
        "argmax_unchanged": bool(np.array_equal(lg.argmax(1), p.argmax(1))),
    }
    root = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}"
    save_json(root / "calibration.json", r)
    prediction_frame(cal, lg, ck["classes"], temp, threshold).to_csv(
        root / "calibration_predictions.csv", index=False
    )
    return r


def evaluate_head(cfg, seed, name, external=False):
    model, ck, arrays, device, mu, std = restore(cfg, seed, name)
    rows = manifest(cfg, seed)
    root = output_dir(cfg) / f"runs/seed_{seed}/heads/{name}"
    calpath = root / "calibration.json"
    cal = read_json(calpath) if calpath.exists() else None
    if external and cal is None:
        raise RuntimeError("Calibrate the fixed checkpoint internally before external evaluation.")
    if cal and cal["signature"] != ck["signature"]:
        raise RuntimeError("Calibration and checkpoint signatures differ.")
    sets = [("internal_evaluation", rows[rows.role.eq("evaluation")])]
    if external:
        sets += [(str(co), g) for co, g in rows[rows.role.eq("external")].groupby("cohort")]
    if external and len(sets) > 2:
        sets.append(("pooled_external", rows[rows.role.eq("external")]))
    results = []
    for co, records in sets:
        cohort_file = (
            co if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", co) else "cohort_" + digest(co)[:16]
        )
        lg = predict(cfg, model, arrays, ck["spec"], records, device, mu, std)
        temperature = cal["temperature"] if cal else 1.0
        threshold = cal["entropy_threshold"] if cal else None
        pred = prediction_frame(records, lg, ck["classes"], temperature, threshold)
        result = evaluate(
            records.label_idx,
            lg,
            ck["classes"],
            temperature,
            cfg.get("evaluation", {}).get("bootstrap_replicates", 20000),
            int(seed),
            cfg.get("evaluation", {}).get("clinical_pair"),
        )
        result.update(
            seed=seed,
            name=name,
            cohort=co,
            spec=ck["spec"],
            temperature=temperature,
            checkpoint_signature=ck["signature"],
            unit="one fitted checkpoint; image-level uncertainty",
        )
        if threshold is not None:
            keep = ~pred.review_flag
            nk = int(keep.sum())
            good = pred.prediction_idx.eq(pred.label_idx)
            from .metrics import wilson

            result.update(
                review_n=int((~keep).sum()),
                coverage=nk / len(pred),
                retained_accuracy=float(good[keep].mean()) if nk else None,
                retained_accuracy_ci=wilson(int(good[keep].sum()), nk),
            )
            order = pred.sort_values(["entropy", "id"])
            k = np.arange(1, len(order) + 1)
            risk = 1 - np.cumsum(order.prediction_idx.eq(order.label_idx)) / k
            curve = pd.DataFrame({"retained_n": k, "coverage": k / len(order), "risk": risk})
            for cl in ck["classes"]:
                total = int(order.label.eq(cl).sum())
                curve[f"coverage_{cl}"] = np.cumsum(order.label.eq(cl)) / total if total else np.nan
            curve.to_csv(root / f"risk_coverage_{cohort_file}.csv", index=False)
        pred.to_csv(root / f"predictions_{cohort_file}.csv", index=False)
        save_json(root / f"metrics_{cohort_file}.json", result)
        results.append(result)
    return results
