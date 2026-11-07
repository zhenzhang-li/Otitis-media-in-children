"""Create clearly synthetic RGB examples; no patient records or medical photographs."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    gen = np.random.default_rng(91)
    for cohort, n in [("internal", 30), ("external_example", 8)]:
        for c, label in enumerate(["pattern_left", "pattern_right"]):
            for i in range(n):
                x = gen.integers(0, 100, size=(32, 32, 3), dtype=np.uint8)
                x[:, c * 16 : (c + 1) * 16, 0] = gen.integers(
                    140, 240, size=(32, 16), dtype=np.uint8
                )
                key = f"{cohort}_{c}_{i:03d}"
                path = root / (key + ".png")
                Image.fromarray(x).save(path)
                records.append(
                    {"image_file": path.name, "target": label, "sample_key": key, "cohort": cohort}
                )
    pd.DataFrame(records).to_csv(root / "records.csv", index=False)
    return root / "records.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "generated"))
    args = parser.parse_args()
    print(generate(args.output))
