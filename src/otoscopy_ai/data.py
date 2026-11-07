"""Dataset adapters and four-role splits. Only the internal record interface is fixed."""

from pathlib import Path
import re, warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from .config import path_for, output_dir, plugin, file_sha, digest, save_json

ROLES = ["train", "selection", "calibration", "evaluation"]


def read_table(path, sheet=0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet, dtype=str)
    if suffix in [".jsonl", ".ndjson"]:
        return pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    if suffix == ".json":
        return pd.read_json(path, dtype=False, convert_dates=False)
    return pd.read_csv(
        path, sep="\t" if suffix == ".tsv" else ",", keep_default_na=False, dtype=str
    )


def source_records(cfg, spec):
    adapter = spec.get("adapter", "table")
    if adapter == "custom":
        frame = pd.DataFrame(plugin(spec["factory"])({**spec, "_config_dir": cfg["_config_dir"]}))
    elif adapter == "folders":
        root = path_for(cfg, spec["root"])
        records = []
        extensions = set(
            x.lower()
            for x in spec.get(
                "extensions", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
            )
        )
        for p in sorted(root.glob(spec.get("glob", "**/*"))):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            if spec.get("label_regex"):
                match = re.search(spec["label_regex"], p.relative_to(root).as_posix())
                if not match:
                    continue
                label = match.group("label")
            else:
                label = p.parents[int(spec.get("label_parent", 0))].name
            records.append({"path": str(p), "label": label})
        frame = pd.DataFrame(records)
    elif adapter == "table":
        frame = read_table(path_for(cfg, spec["table"]), spec.get("sheet", 0))
    else:
        raise ValueError(f"Unknown adapter {adapter}; use table/folders/custom.")
    # Mapping direction is canonical name -> reader's column name.
    columns = spec.get("columns", {})
    frame = frame.rename(columns={v: k for k, v in columns.items()})
    for key, value in spec.get("defaults", {}).items():
        if key not in frame:
            frame[key] = value
    for key in ["path", "label"]:
        if key not in frame:
            raise ValueError(
                f"Data adapter must return {key}; configure columns or defaults for your source."
            )
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {key} values in source records.")
    frame = frame.copy()
    frame["label"] = frame.label.astype(str).map(
        lambda x: str(cfg.get("data", {}).get("label_map", {}).get(x, x))
    )
    frame["cohort"] = frame.get("cohort", spec.get("cohort", "internal"))
    frame["role"] = frame.get("role", "")
    frame["cohort"] = frame.cohort.astype(str)
    frame["role"] = frame.role.fillna("").astype(str).replace(spec.get("role_map", {}))
    if spec.get("loader"):
        frame["loader"] = spec["loader"]
        frame["path"] = frame.path.astype(str)
    else:
        base = path_for(cfg, spec.get("root", str(Path(cfg["_config_dir"]))))
        frame["path"] = frame.path.astype(str).map(
            lambda x: (
                str((base / Path(x)).resolve())
                if not Path(x).is_absolute()
                else str(Path(x).resolve())
            )
        )
        frame["loader"] = ""
    if "id" not in frame:
        frame["id"] = [
            digest([p, c, l])[:24] for p, c, l in zip(frame.path, frame.cohort, frame.label)
        ]
    frame["id"] = str(spec.get("id_prefix", "")) + frame.id.astype(str)
    frame["group_provided"] = "group" in frame
    if "group" not in frame:
        frame["group"] = frame.id
    if frame.group.isna().any() or frame.group.astype(str).str.strip().eq("").any():
        raise ValueError("Missing group values; provide a complete group column or omit it.")
    frame["group"] = str(spec.get("group_prefix", "")) + frame.group.astype(str)
    return frame


def prepare(cfg):
    dc = cfg["data"]
    sources = dc.get("sources", [dc])
    df = pd.concat([source_records(cfg, s) for s in sources], ignore_index=True)
    if df.empty:
        raise ValueError("No images matched the configured adapters.")
    if df.id.duplicated().any():
        raise ValueError(
            "Image IDs must be unique. Provide an explicit id column for repeated file records."
        )
    if df[["path", "label", "group", "id"]].isna().any().any() or df.label.eq("").any():
        raise ValueError("Paths, labels, groups and IDs must be present.")
    labels = list(map(str, dc.get("classes", sorted(df.label.unique()))))
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("At least two unique classes are required.")
    unknown = set(df.label) - set(labels)
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; check class list and label_map.")
    df["label_idx"] = df.label.map({v: i for i, v in enumerate(labels)})
    df["index"] = np.arange(len(df))
    fingerprints = []
    for _, row in df.iterrows():
        if row.loader:
            fingerprints.append(str(row.get("content_hash", digest([row.path, row.id]))))
            continue
        p = Path(row.path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found for id={row.id}: {p}")
        fingerprints.append(
            file_sha(p)
            if dc.get("hash_images", True)
            else digest([str(p), p.stat().st_size, p.stat().st_mtime_ns])
        )
    df["content_hash"] = fingerprints
    out = output_dir(cfg)
    df.to_csv(out / "records.csv", index=False)
    save_json(out / "classes.json", labels)
    audit = {
        "records": len(df),
        "classes": labels,
        "cohorts": df.cohort.value_counts().to_dict(),
        "duplicate_content_records": int(df.content_hash.duplicated(keep=False).sum()),
        "group_column_records": int(df.group_provided.sum()),
        "image_fingerprint_mode": (
            "sha256"
            if dc.get("hash_images", True)
            else "path,size,mtime; weaker cache invalidation"
        ),
    }
    save_json(out / "dataset_audit.json", audit)
    return df, labels


def read_records(cfg):
    out = output_dir(cfg)
    path = out / "records.csv"
    if not path.exists():
        return prepare(cfg)
    df = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "group": str, "label": str})
    labels = __import__("json").loads((out / "classes.json").read_text("utf-8"))
    return df, labels


def image_split(df, fractions, seed):
    fractions = np.asarray(fractions, float)
    exact = fractions * len(df)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts), kind="stable")[: len(df) - counts.sum()]:
        counts[i] += 1
    tr, rest = train_test_split(
        df, test_size=int(counts[1:].sum()), stratify=df.label_idx, random_state=seed
    )
    sel, rest = train_test_split(
        rest, train_size=int(counts[1]), stratify=rest.label_idx, random_state=seed + 1
    )
    cal, ev = train_test_split(
        rest, train_size=int(counts[2]), stratify=rest.label_idx, random_state=seed + 2
    )
    return [tr, sel, cal, ev]


def group_split(df, fractions, seed):
    """Greedy class-balanced whole-group allocation; ratios are approximate."""
    groups = list(df.groupby("group", sort=True))
    nclasses = int(df.label_idx.max() + 1)
    target = np.asarray(fractions)[:, None] * np.bincount(df.label_idx, minlength=nclasses)
    best = None
    for trial in range(60):
        gen = np.random.default_rng(seed + trial)
        order = gen.permutation(len(groups))
        order = sorted(order, key=lambda i: -len(groups[i][1]))
        count = np.zeros_like(target)
        assigned = [[] for _ in ROLES]
        for gi in order:
            group, g = groups[gi]
            v = np.bincount(g.label_idx, minlength=nclasses)
            cost = []
            for role in range(4):
                before = ((count[role] - target[role]) ** 2 / np.maximum(target[role], 1)).sum()
                after = ((count[role] + v - target[role]) ** 2 / np.maximum(target[role], 1)).sum()
                cost.append(after - before)
            role = int(gen.choice(np.flatnonzero(np.isclose(cost, min(cost)))))
            count[role] += v
            assigned[role].extend(g.index.tolist())
        if (count == 0).any():
            continue
        score = float(((count - target) ** 2 / np.maximum(target, 1)).sum())
        if best is None or score < best[0]:
            best = (score, assigned)
    if best is None:
        raise ValueError(
            "Cannot put every class in all four roles with the available whole groups; supply predefined roles or more groups."
        )
    return [df.loc[ix] for ix in best[1]]


def split(cfg, records=None):
    df, labels = read_records(cfg) if records is None else records
    settings = cfg["split"]
    if settings.get("mode") == "predefined" and len(settings.get("seeds", [42])) != 1:
        raise ValueError(
            "One fixed predefined allocation needs one outer seed; repeating identical role lists is not a new random split."
        )
    fractions = np.asarray(settings.get("fractions", [0.6, 0.15, 0.15, 0.1]), float)
    if len(fractions) != 4 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Four positive split fractions must sum to 1.")
    cohort = settings.get("internal_cohort", "internal")
    internal = df[df.cohort.eq(cohort)]
    external = df[~df.cohort.eq(cohort)]
    out = output_dir(cfg)
    if internal.empty:
        raise ValueError(f"No records in internal_cohort={cohort}.")
    for seed in settings.get("seeds", [42]):
        if settings.get("mode", "image") == "predefined":
            parts = [internal[internal.role.eq(role)] for role in ROLES]
            if sum(map(len, parts)) != len(internal):
                raise ValueError(
                    "Predefined internal roles must be train/selection/calibration/evaluation."
                )
        else:
            fn = group_split if settings.get("mode") == "group" else image_split
            parts = fn(internal, fractions, int(seed))
        rows = []
        for role, p in zip(ROLES, parts):
            if set(p.label) != set(labels):
                raise ValueError(
                    f"{role} does not contain every class; use larger cohorts or compatible predefined splits."
                )
            rows.append(p.assign(role=role))
        manifest = pd.concat(
            rows + [external.assign(role="external")], ignore_index=True
        ).sort_values("index")
        audit = []
        for key in ["group", "content_hash"]:
            for i, a in enumerate(ROLES):
                aa = set(manifest.loc[manifest.role.eq(a), key])
                for b in ROLES[i + 1 :] + ["external"]:
                    overlap = aa & set(manifest.loc[manifest.role.eq(b), key])
                    audit.append(
                        {"key": key, "role_a": a, "role_b": b, "overlap_count": len(overlap)}
                    )
        bad = [
            a
            for a in audit
            if a["overlap_count"] and (a["key"] == "content_hash" or df.group_provided.any())
        ]
        if bad and settings.get("overlap_policy", "error") == "error":
            raise ValueError(
                f"Cross-role/group content overlap found: {bad}. Correct grouping or explicitly set overlap_policy: report to retain and document it."
            )
        sd = out / f"splits/seed_{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(sd / "manifest.csv", index=False)
        save_json(sd / "overlap_audit.json", audit)
        manifest.groupby(["cohort", "role", "label"], sort=False).size().rename(
            "n"
        ).reset_index().to_csv(sd / "class_counts.csv", index=False)
    return df, labels


def manifest(cfg, seed):
    p = output_dir(cfg) / f"splits/seed_{seed}/manifest.csv"
    if not p.exists():
        split(cfg)
    return pd.read_csv(p, keep_default_na=False, dtype={"id": str, "group": str, "label": str})


def load_image(row):
    if row.get("loader", ""):
        image = plugin(row["loader"])(dict(row))
        return (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        )
    with Image.open(row["path"]) as image:
        return image.convert("RGB")


def crop_transform(cfg):
    p = cfg["preprocessing"]
    return transforms.Compose(
        [transforms.Resize(p.get("resize", 256)), transforms.CenterCrop(p.get("crop", 224))]
    )


def normalize_transform(cfg):
    p = cfg["preprocessing"]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                p.get("mean", [0.485, 0.456, 0.406]), p.get("std", [0.229, 0.224, 0.225])
            ),
        ]
    )


class Images(__import__("torch").utils.data.Dataset):
    def __init__(self, cfg, records, perturb=None):
        self.records = records.reset_index(drop=True)
        self.crop = crop_transform(cfg)
        self.tensor = normalize_transform(cfg)
        self.perturb = perturb

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        im = load_image(self.records.iloc[i])
        if self.perturb:
            im = self.perturb(im)
        return self.tensor(self.crop(im)), i
