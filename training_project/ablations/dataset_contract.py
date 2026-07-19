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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataset_content(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    split_reports = {}
    combined_records = []
    image_occurrences: dict[str, list[str]] = {}
    for split in SPLITS:
        image_root = dataset_root / "images" / split
        label_root = dataset_root / "labels" / split
        images = sorted(
            path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        records = []
        empty_labels = 0
        invalid_label_rows = 0
        for image_path in images:
            relative_image = image_path.relative_to(image_root)
            label_path = label_root / relative_image.with_suffix(".txt")
            image_hash = sha256_file(image_path)
            label_hash = sha256_file(label_path) if label_path.is_file() else "MISSING"
            if label_path.is_file():
                rows = [row.strip() for row in label_path.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
                if not rows:
                    empty_labels += 1
                for row in rows:
                    try:
                        values = [float(value) for value in row.split()]
                    except ValueError:
                        invalid_label_rows += 1
                        continue
                    if (
                        len(values) != 5
                        or not values[0].is_integer()
                        or not 0 <= int(values[0]) < 5
                        or any(not 0.0 <= value <= 1.0 for value in values[1:])
                    ):
                        invalid_label_rows += 1
            relative_record = (
                f"images/{split}/{relative_image.as_posix()}\t{image_hash}\t"
                f"labels/{split}/{relative_image.with_suffix('.txt').as_posix()}\t{label_hash}"
            )
            records.append(relative_record)
            image_occurrences.setdefault(image_hash, []).append(f"{split}/{relative_image.as_posix()}")
        payload = ("\n".join(records) + "\n").encode("utf-8")
        split_reports[split] = {
            "content_fingerprint_sha256": hashlib.sha256(payload).hexdigest(),
            "empty_labels": empty_labels,
            "invalid_label_rows": invalid_label_rows,
        }
        combined_records.extend(records)
    duplicate_records = sorted(
        f"{digest}\t{'|'.join(paths)}"
        for digest, paths in image_occurrences.items()
        if len({path.split('/', 1)[0] for path in paths}) > 1
    )
    combined_payload = ("\n".join(combined_records) + "\n").encode("utf-8")
    duplicate_payload = ("\n".join(duplicate_records) + "\n").encode("utf-8")
    return {
        "algorithm": "sha256(relative image path+content, relative label path+content)",
        "splits": split_reports,
        "combined_content_fingerprint_sha256": hashlib.sha256(combined_payload).hexdigest(),
        "cross_split_duplicate_images": len(duplicate_records),
        "cross_split_duplicate_fingerprint_sha256": hashlib.sha256(duplicate_payload).hexdigest(),
        "cross_split_duplicate_records": duplicate_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--content", action="store_true", help="Also hash image/label contents and check leakage.")
    args = parser.parse_args()
    report = fingerprint_dataset(args.dataset_root)
    if args.content:
        report["content"] = fingerprint_dataset_content(args.dataset_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
