"""Create a portable fingerprint for a YOLO image/label split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def fingerprint_dataset(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    split_reports = {}
    combined_records = []
    for split in SPLITS:
        image_root = dataset_root / "images" / split
        label_root = dataset_root / "labels" / split
        images = sorted(
            path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(path for path in label_root.rglob("*.txt") if path.is_file())
        records = []
        missing_labels = 0
        for image_path in images:
            relative_image = image_path.relative_to(image_root)
            label_path = label_root / relative_image.with_suffix(".txt")
            if not label_path.is_file():
                missing_labels += 1
            records.append(
                f"images/{split}/{relative_image.as_posix()}\t"
                f"labels/{split}/{relative_image.with_suffix('.txt').as_posix()}\t"
                f"{int(label_path.is_file())}"
            )
        payload = ("\n".join(records) + "\n").encode("utf-8")
        split_reports[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": missing_labels,
            "fingerprint_sha256": hashlib.sha256(payload).hexdigest(),
        }
        combined_records.extend(records)
    combined_payload = ("\n".join(combined_records) + "\n").encode("utf-8")
    return {
        "algorithm": "sha256(sorted relative image path, expected label path, label-present flag)",
        "splits": split_reports,
        "combined_fingerprint_sha256": hashlib.sha256(combined_payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(fingerprint_dataset(args.dataset_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
