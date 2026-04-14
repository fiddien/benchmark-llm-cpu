"""scripts/make_sample_manifest.py
Utility to build a sample manifest JSON from a folder of images.

Usage:
    python scripts/make_sample_manifest.py data/images --task captioning \
        --prompt "Describe this image." --out data/captioning.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a task manifest from images.")
    parser.add_argument("image_dir", type=Path, help="Directory of images.")
    parser.add_argument("--task", required=True, choices=["captioning", "vqa", "structured_output"])
    parser.add_argument("--prompt", default="", help="Default prompt for all samples.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args()

    image_dir: Path = args.image_dir
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {image_dir}")
        return

    records = [
        {"image": str(img), "prompt": args.prompt, "reference": None}
        for img in images
    ]
    out = args.out or Path("data") / f"{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} samples to {out}")


if __name__ == "__main__":
    main()
