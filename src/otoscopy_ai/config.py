"""Configuration, reproducibility and output provenance utilities."""

from pathlib import Path
import hashlib, importlib, json, os, random
import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=default), encoding="utf-8"
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path, overrides=()):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for override in overrides:
        key, value = override.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    cfg["_config_dir"] = str(path.parent)
    cfg["_config_file"] = str(path)
    return cfg


def path_for(cfg, value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p.resolve() if p.is_absolute() else (Path(cfg["_config_dir"]) / p).resolve()


def output_dir(cfg):
    p = path_for(cfg, cfg.get("output", "../outputs/run"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def plugin(name):
    module, attr = name.split(":", 1)
    return getattr(importlib.import_module(module), attr)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_for(cfg):
    requested = cfg.get("device", "auto")
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; select device: cpu or install a matching PyTorch CUDA build."
        )
    return torch.device(requested)


def public_config(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}
