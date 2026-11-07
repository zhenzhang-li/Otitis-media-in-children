"""Use as data.sources[].factory: examples.custom_adapter:records.

Replace this function with your database/query/HDF5/metadata format. It returns
canonical records; loading the image itself can be another plugin if necessary.
"""

from pathlib import Path
import json
from PIL import Image


def records(spec):
    path = Path(spec["metadata_file"])
    if not path.is_absolute():
        path = Path(spec["_config_dir"]) / path
    metadata = json.loads(path.read_text("utf-8"))
    return [
        {
            "id": str(r["key"]),
            "path": r["resource"],
            "label": str(r["category"]),
            "group": str(r.get("patient", r["key"])),
            "cohort": r.get("hospital", "internal"),
        }
        for r in metadata["items"]
    ]


def load_image(row):
    # Replace with e.g. an archive member / array-store decoder for resource IDs.
    with Image.open(row["path"]) as im:
        return im.convert("RGB")
