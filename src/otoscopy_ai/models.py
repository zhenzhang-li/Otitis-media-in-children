"""Ten study encoders plus explicit extension points. No implicit weight download."""

from pathlib import Path
import importlib, sys
import torch
from torch import nn
import torchvision.models as tv
from .config import path_for, plugin, file_sha, digest

STUDY_MODELS = [
    "resnet50",
    "swin_b",
    "vit_s16",
    "vit_b16",
    "convnext_tiny",
    "convnext_base",
    "dinov3_vits16",
    "dinov3_vitb16",
    "dinov3_convnext_tiny",
    "dinov3_convnext_base",
]


class ToyEncoder(nn.Module):
    """Deterministic untrained encoder for offline integration tests, not a study model."""

    def __init__(self, dim=16, depth=4, patch=8):
        super().__init__()
        self.patch = patch
        self.depth = depth
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def outputs(self, x, layers):
        p = self.proj(x).flatten(2).transpose(1, 2)
        cls = []
        tokens = []
        for block in self.blocks:
            p = p + torch.tanh(block(p))
            normalized = self.norm(p)
            c = normalized.mean(1)
            cls.append(c)
            tokens.append(torch.cat([c[:, None], normalized], 1))
        return {
            "pooled": cls[-1],
            "cls": torch.stack(cls, 1),
            "tokens": torch.stack([tokens[i - 1] for i in layers], 1),
        }


class Encoder(nn.Module):
    def __init__(self, model, provider, layers):
        super().__init__()
        self.model = model
        self.provider = provider
        self.layers = layers

    def forward(self, x):
        if self.provider == "toy":
            return self.model.outputs(x, self.layers)
        if self.provider == "dinov3_vit":
            depth = len(self.model.blocks)
            outputs = self.model.get_intermediate_layers(
                x, n=list(range(depth)), return_class_token=True, norm=True
            )
            cls = torch.stack([a[1] for a in outputs], 1)
            tokens = torch.stack(
                [
                    torch.cat([outputs[i - 1][1][:, None], outputs[i - 1][0]], 1)
                    for i in self.layers
                ],
                1,
            )
            return {"pooled": cls[:, -1], "cls": cls, "tokens": tokens}
        if self.provider in ["torchvision", "timm"]:
            value = self.model(x)
        elif self.provider == "custom":
            return self.model(x)
        else:
            value = self.model.forward_features(x)
            if isinstance(value, dict):
                value = value.get("x_norm_clstoken", value.get("pooler_output"))
            if value is None:
                raise ValueError(
                    "Encoder dict has no pooled features; use a custom adapter returning pooled/cls/tokens."
                )
            if value.ndim == 4:
                value = value.mean((-2, -1))
            elif value.ndim == 3:
                value = value[:, 0]
        if value.ndim != 2:
            raise ValueError(f"Expected pooled features [B,D], obtained {value.shape}.")
        return {"pooled": value}


def build_encoder(cfg, key):
    spec = cfg["models"][key]
    provider = spec["provider"]
    weights = spec.get("weights")
    allow = bool(spec.get("allow_download", False))
    layers = cfg["fusion"].get("layers", [4, 8, 12])
    if provider == "toy":
        with torch.random.fork_rng():
            torch.manual_seed(int(spec.get("seed", 0)))
            model = ToyEncoder(**spec.get("kwargs", {}))
        kind = "toy"
    elif provider == "custom":
        model = plugin(spec["factory"])(spec)
        kind = "custom"
    elif provider == "dinov3":
        repo = path_for(cfg, spec["repository"])
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                "Clone the official DINOv3 repository and set models.<key>.repository; DINOv3 code/weights are not bundled."
            )
        if not weights:
            raise ValueError("A local, licensed DINOv3 checkpoint is required in weights.")
        wp = path_for(cfg, weights)
        if not wp.is_file():
            raise FileNotFoundError(wp)
        # Import only official backbones: hubconf also imports optional segmentation
        # dependencies that this classification package does not need.
        loaded = sys.modules.get("dinov3")
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(repo):
            raise RuntimeError(
                "A different DINOv3 checkout is already imported; use a fresh Python process."
            )
        sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module("dinov3.hub.backbones")
            model = getattr(module, spec.get("name", key))(weights=str(wp))
        finally:
            sys.path.remove(str(repo))
        kind = "dinov3_vit" if "vit" in spec.get("name", key) else "dinov3_convnext"
    elif provider == "torchvision":
        name = spec.get("name", key)
        model = tv.get_model(
            name, weights=None if weights or not allow else spec.get("weight_enum", "DEFAULT")
        )
        if weights:
            _load_local(model, path_for(cfg, weights))
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif name.startswith("convnext"):
            model.classifier[-1] = nn.Identity()
        elif hasattr(model, "heads"):
            model.heads = nn.Identity()
        elif hasattr(model, "head"):
            model.head = nn.Identity()
        else:
            raise ValueError("Unknown torchvision head; add a custom encoder adapter.")
        kind = "torchvision"
    elif provider == "timm":
        import timm

        model = timm.create_model(
            spec["name"], pretrained=allow and not weights, **spec.get("kwargs", {})
        )
        if weights:
            _load_local(model, path_for(cfg, weights))
        model.reset_classifier(0)
        kind = "timm"
    else:
        raise ValueError(f"Unknown encoder provider: {provider}")
    if provider not in ["toy", "custom", "dinov3"] and not weights and not allow:
        raise ValueError(
            f"{key}: weights missing. Supply local weights or explicitly allow_download: true. Random encoders cannot be used silently as pretrained baselines."
        )
    enc = Encoder(model, kind, layers).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def _load_local(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)


def encoder_identity(cfg, key):
    spec = cfg["models"][key]
    identity = {
        "model": spec,
        "preprocessing": cfg["preprocessing"],
        "layers": cfg["fusion"]["layers"],
        "adapter_source_sha256": file_sha(Path(__file__)),
        "data_source_sha256": file_sha(Path(__file__).with_name("data.py")),
    }
    if spec.get("weights"):
        identity["weights_sha256"] = file_sha(path_for(cfg, spec["weights"]))
    if spec.get("repository"):
        repo = path_for(cfg, spec["repository"])
        head = repo / ".git/HEAD"
        identity["repository_head"] = (
            head.read_text("utf-8").strip() if head.is_file() else "unversioned local clone"
        )
        # Hash the architecture sources even when .git is unavailable or on a branch.
        identity["architecture_sources"] = {
            p.relative_to(repo).as_posix(): file_sha(p)
            for p in sorted((repo / "dinov3").rglob("*.py"))
        }
    if spec.get("factory"):
        identity["factory"] = spec["factory"]
    return identity
