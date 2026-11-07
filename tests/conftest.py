from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from otoscopy_ai.config import load_config


@pytest.fixture
def experiment(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    cfg["_config_dir"] = str(tmp_path)
    cfg["output"] = "results"
    cfg["split"]["seeds"] = [19, 29]
    cfg["training"].update(max_epochs=2, min_epochs=1, patience=1, log_every=100)
    cfg["explain"].update(max_cases=1, regions=2, full_region_layer_decomposition=True)
    records = []
    gen = np.random.default_rng(6)
    # Three arbitrary classes, non-prescribed filenames/folders and a different schema.
    for cohort, n in [("internal", 24), ("other_hospital", 6)]:
        for c in range(3):
            for i in range(n):
                folder = tmp_path / f"unrelated folder {c}"
                folder.mkdir(exist_ok=True)
                path = folder / f"{cohort} photo {i}.png"
                Image.fromarray(gen.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
                records.append(
                    {
                        "resource": str(path),
                        "category": f"custom_{c}",
                        "key": f"{cohort}_{c}_{i}",
                        "site": cohort,
                    }
                )
    frame = pd.DataFrame(records)
    frame.to_csv(tmp_path / "input.csv", index=False)
    cfg["data"] = {
        "hash_images": True,
        "sources": [
            {
                "adapter": "table",
                "table": "input.csv",
                "columns": {"path": "resource", "label": "category", "id": "key", "cohort": "site"},
            }
        ],
    }
    cfg["experiments"] = {"backbones": ["toy_b"], "layer_encoders": ["toy_b"], "pad_to": {}}
    cfg["evaluation"]["bootstrap_replicates"] = 100
    import torch

    torch.set_num_threads(2)
    return cfg, frame, tmp_path
