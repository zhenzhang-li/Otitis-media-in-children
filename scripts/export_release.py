"""Create a clean source-only GitHub archive, excluding all runtime artifacts."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "configs", "examples", "scripts", "tests", "docs", ".github"}
FILES = {
    "README.md",
    "README.zh-CN.md",
    "pyproject.toml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CITATION.cff.template",
    "CITATION.cff",
    ".gitignore",
}
EXCLUDED = {
    "__pycache__",
    "generated",
    ".pytest_cache",
    "outputs",
    "weights",
    "data",
    "third_party",
    "format_tools",
}


def release_files():
    result = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED for part in rel.parts):
            continue
        if rel.parts[0] not in DIRECTORIES and rel.as_posix() not in FILES:
            continue
        if path.is_symlink() or path.suffix.lower() in {
            ".pyc",
            ".pt",
            ".pth",
            ".npy",
            ".npz",
            ".safetensors",
            ".xlsx",
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".dcm",
        }:
            raise RuntimeError(f"Unexpected binary/data artifact in source release: {rel}")
        text = path.read_text("utf-8")
        for pattern in [
            r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]",
            r"/(?:home|Users)/[^/\s]+",
            r"ghp_[A-Za-z0-9]{30,}",
            r"hf_[A-Za-z0-9]{30,}",
        ]:
            if re.search(pattern, text):
                raise RuntimeError(f"Possible private absolute path or credential in {rel}")
        result.append(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--directory", help="Also write a clean source folder; it must not exist.")
    args = parser.parse_args()
    target = Path(args.destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = release_files()
    clean = Path(args.directory).resolve() if args.directory else None
    if clean is not None:
        if clean.exists():
            raise FileExistsError(clean)
        clean.mkdir(parents=True)
    manifest = []
    with ZipFile(target, "x", ZIP_DEFLATED) as z:
        for path in paths:
            rel = path.relative_to(ROOT)
            z.write(path, (Path("otoscopy-layerfusion") / rel).as_posix())
            manifest.append(
                {"file": rel.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
            if clean is not None:
                dest = clean / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        z.writestr("otoscopy-layerfusion/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
    if clean is not None:
        (clean / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    with ZipFile(target) as z:
        assert z.testzip() is None
    print(
        json.dumps(
            {
                "archive": str(target),
                "source_files": len(paths),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
