import json
import numpy as np
import pandas as pd
import pytest
from otoscopy_ai.config import output_dir
from otoscopy_ai.data import prepare, split, manifest, group_split, read_table


def test_column_mapping_formats_and_deterministic_split(experiment):
    cfg, frame, path = experiment
    for suffix in ["csv", "xlsx", "jsonl"]:
        file = path / f"input.{suffix}"
        if suffix == "xlsx":
            frame.to_excel(file, index=False)
        if suffix == "jsonl":
            frame.to_json(file, orient="records", lines=True)
        cfg["data"]["sources"][0]["table"] = file.name
        rows, classes = prepare(cfg)
        assert len(classes) == 3
        assert rows.path.str.contains("unrelated folder").all()
        split(cfg)
        a = manifest(cfg, 19)
        split(cfg)
        b = manifest(cfg, 19)
        assert a[["id", "role"]].equals(b[["id", "role"]])
        internal = a[a.cohort.eq("internal")]
        assert internal.role.value_counts().to_dict() == {
            "train": 43,
            "selection": 11,
            "calibration": 11,
            "evaluation": 7,
        }
        assert a[a.cohort.eq("other_hospital")].role.eq("external").all()


def test_folder_adapter_arbitrary_layout_and_pattern(experiment):
    cfg, frame, path = experiment
    cfg["data"]["sources"] = [
        {
            "adapter": "folders",
            "root": str(path),
            "glob": "**/*.png",
            "label_regex": r"unrelated folder (?P<label>\d+)/",
            "cohort": "internal",
        }
    ]
    rows, classes = prepare(cfg)
    assert len(rows) == len(frame)
    assert classes == ["0", "1", "2"]


def test_group_stratification_preserves_whole_groups():
    records = []
    for c in range(3):
        for g in range(15):
            for i in range(2):
                records.append({"label_idx": c, "group": f"{c}_{g}"})
    frame = pd.DataFrame(records)
    parts = group_split(frame, [0.6, 0.15, 0.15, 0.1], 11)
    assert sum(map(len, parts)) == len(frame)
    for i, a in enumerate(parts):
        assert set(a.label_idx) == {0, 1, 2}
        for b in parts[i + 1 :]:
            assert not set(a.group) & set(b.group)


def test_duplicate_content_is_detected_between_roles(experiment):
    cfg, frame, path = experiment
    cfg["split"].update(mode="predefined", seeds=[19])
    frame["assigned"] = [
        ["train", "selection", "calibration", "evaluation"][i % 4] for i in range(len(frame))
    ]
    # Force the same content into train and selection while keeping unique record IDs.
    first = frame[frame.site.eq("internal")].index[:2]
    frame.loc[first[1], "resource"] = frame.loc[first[0], "resource"]
    frame.to_csv(path / "input.csv", index=False)
    cfg["data"]["sources"][0]["columns"]["role"] = "assigned"
    prepare(cfg)
    with pytest.raises(ValueError, match="overlap"):
        split(cfg)
    cfg["split"]["overlap_policy"] = "report"
    split(cfg)
    assert (output_dir(cfg) / "splits/seed_19/overlap_audit.json").exists()


def test_missing_labels_rejected(experiment):
    cfg, frame, path = experiment
    frame.loc[0, "category"] = ""
    frame.to_csv(path / "input.csv", index=False)
    with pytest.raises(ValueError, match="Missing label"):
        prepare(cfg)


def test_custom_metadata_and_in_memory_rgb_loader(experiment, monkeypatch):
    cfg, frame, path = experiment
    module = path / "reader_plugin.py"
    module.write_text(
        "import json\nfrom pathlib import Path\nimport numpy as np\n"
        "def records(spec):\n return json.loads((Path(spec['_config_dir'])/'resources.json').read_text())\n"
        "def image(row):\n return np.full((32,32,3),int(row['pixel_value']),dtype=np.uint8)\n",
        encoding="utf-8",
    )
    records = [
        {
            "id": f"{c}_{i}",
            "path": f"store-resource-{c}-{i}",
            "label": str(c),
            "pixel_value": c * 50 + i,
            "content_hash": f"immutable-v1-{c}-{i}",
        }
        for c in range(3)
        for i in range(12)
    ]
    (path / "resources.json").write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.syspath_prepend(str(path))
    cfg["data"]["sources"] = [
        {"adapter": "custom", "factory": "reader_plugin:records", "loader": "reader_plugin:image"}
    ]
    rows, labels = prepare(cfg)
    assert labels == ["0", "1", "2"]
    from otoscopy_ai.data import Images

    image, _ = Images(cfg, rows)[0]
    assert tuple(image.shape) == (3, 32, 32)


def test_predefined_repeats_are_not_counted_as_new_splits(experiment):
    cfg, _, _ = experiment
    prepare(cfg)
    cfg["split"]["mode"] = "predefined"
    with pytest.raises(ValueError, match="identical role lists"):
        split(cfg)
