"""Step 1: pull a subset from allenai/pixmo-point-explanations.

Keeps normalized (0-100) point coordinates and the ground-truth explanation
text alongside each downloaded image. Restricted to examples with exactly one
referenced point group, since scoring a single model turn against a single
target point is what steps 5-6 assume.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image

DATASET_NAME = "allenai/pixmo-point-explanations"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "subset"
USER_AGENT = "pixmo-vlm-bench/0.1 (research eval; contact via HF dataset card)"


@dataclass
class Example:
    id: str
    image_path: str
    question: str
    explanation: str
    points: list[list[float]]  # alternate valid (x, y) locations, 0-100 scale


def _download_image(url: str, dest: Path, timeout: float = 10.0) -> bool:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        with Image.open(dest) as im:
            im.verify()
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def build_subset(
    n: int = 150,
    seed: int = 0,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_source_rows: int = 20_000,
) -> Path:
    """Download `n` examples and write out_dir/manifest.jsonl + out_dir/images/."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_source_rows, len(ds))))

    manifest: list[Example] = []
    for row in ds:
        if len(manifest) >= n:
            break
        points = row["points"]
        if len(points) != 1 or not row["alt_text"] or not row["question"].strip():
            continue

        ex_id = row["image_sha256"][:16]
        img_path = images_dir / f"{ex_id}.jpg"
        if not img_path.exists() and not _download_image(row["image_url"], img_path):
            continue

        manifest.append(
            Example(
                id=ex_id,
                image_path=str(img_path.relative_to(out_dir)),
                question=row["question"].strip(),
                explanation=(row["alt_text"][0] or row["inline_text"][0] or "").strip(),
                points=points[0],
            )
        )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for ex in manifest:
            f.write(json.dumps(asdict(ex)) + "\n")

    print(f"wrote {len(manifest)} examples to {manifest_path}")
    return manifest_path


def load_subset(out_dir: Path = DEFAULT_OUT_DIR) -> list[Example]:
    manifest_path = Path(out_dir) / "manifest.jsonl"
    examples = []
    with manifest_path.open() as f:
        for line in f:
            d = json.loads(line)
            examples.append(Example(**d))
    return examples


if __name__ == "__main__":
    t0 = time.time()
    build_subset()
    print(f"done in {time.time() - t0:.1f}s")
