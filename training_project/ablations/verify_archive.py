from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("manifest.yaml")
EXPECTED_IDS = ["A0", "B1", "B2", "B3", "B4", "B5"]
HEX64 = re.compile(r"[0-9a-f]{64}")
RUNTIME_STATUSES = {"blocked", "source_verified", "runnable", "verified"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify immutable ablation YAML provenance.")
    parser.add_argument("--check-source", action="store_true", help="Also compare against the external history directory.")
    parser.add_argument("--check-git", action="store_true", help="Also verify that committed HEAD blobs preserve the source bytes.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="Manifest path, primarily for negative tests.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    experiments = manifest["experiments"]
    ids = [item["id"] for item in experiments]
    if ids != EXPECTED_IDS:
        raise RuntimeError(f"Expected ordered experiment ids {EXPECTED_IDS}, got {ids}")

    archive_root_value = Path(manifest["archive_root"])
    if archive_root_value.is_absolute():
        raise RuntimeError("archive_root must be repository-relative")
    archive_root = (ROOT / archive_root_value).resolve()
    try:
        archive_root.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("archive_root escapes the repository") from exc

    checked = []
    expected_yaml_paths = set()
    for item in experiments:
        archive_value = Path(item["archive_path"])
        canonical_value = Path(item["canonical_path"])
        if archive_value.is_absolute() or canonical_value.is_absolute():
            raise RuntimeError(f"Runtime paths must be repository-relative for {item['id']}")
        archive = (ROOT / archive_value).resolve()
        canonical = (ROOT / canonical_value).resolve()
        try:
            archive.relative_to(archive_root)
            canonical.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Path escapes its allowed root for {item['id']}") from exc
        if archive.suffix.lower() not in {".yaml", ".yml"}:
            raise RuntimeError(f"Archive is not YAML for {item['id']}: {archive}")
        if not (item["source_filename"] == Path(item["source_path"]).name == archive.name):
            raise RuntimeError(f"Provenance basename mismatch for {item['id']}")
        if item.get("archive_status") != "verified" or item.get("runtime_status") not in RUNTIME_STATUSES:
            raise RuntimeError(f"Invalid status fields for {item['id']}")
        if not HEX64.fullmatch(item["source_sha256"]):
            raise RuntimeError(f"Invalid source SHA-256 for {item['id']}")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        observed_hash = sha256(archive)
        observed_bytes = archive.stat().st_size
        if observed_hash != item["source_sha256"] or observed_bytes != item["source_bytes"]:
            raise RuntimeError(f"Archive drift for {item['id']}: hash={observed_hash} bytes={observed_bytes}")
        if args.check_source:
            source = Path(item["source_path"])
            if not source.is_file() or sha256(source) != observed_hash:
                raise RuntimeError(f"External source mismatch for {item['id']}: {source}")
        if args.check_git:
            repository_path = archive.relative_to(ROOT).as_posix()
            committed = subprocess.run(
                ["git", "show", f"HEAD:{repository_path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            committed_hash = hashlib.sha256(committed).hexdigest()
            if committed_hash != observed_hash or len(committed) != observed_bytes:
                raise RuntimeError(
                    f"Committed archive drift for {item['id']}: hash={committed_hash} bytes={len(committed)}"
                )
        checked.append({"id": item["id"], "sha256": observed_hash, "bytes": observed_bytes})
        expected_yaml_paths.add(archive)

    actual_yaml_paths = {
        path.resolve()
        for path in archive_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    }
    if actual_yaml_paths != expected_yaml_paths:
        extra = sorted(str(path.relative_to(archive_root)) for path in actual_yaml_paths - expected_yaml_paths)
        missing = sorted(str(path.relative_to(archive_root)) for path in expected_yaml_paths - actual_yaml_paths)
        raise RuntimeError(f"Archive YAML set mismatch: extra={extra} missing={missing}")
    allowed_files = expected_yaml_paths | {(archive_root / "README.md").resolve()}
    actual_files = {path.resolve() for path in archive_root.rglob("*") if path.is_file()}
    if actual_files != allowed_files:
        unexpected = sorted(str(path.relative_to(archive_root)) for path in actual_files - allowed_files)
        raise RuntimeError(f"Unexpected files in immutable archive: {unexpected}")

    b4 = next(item for item in experiments if item["id"] == "B4")
    canonical = (ROOT / b4["canonical_path"]).resolve()
    if sha256(canonical) != b4["source_sha256"]:
        raise RuntimeError("B4 canonical no longer matches the captured historical source")
    weights = manifest["historical_weights"]["artifacts"]
    filenames = [item["filename"] for item in weights]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("Historical weight filenames must be unique")
    for item in weights:
        if Path(item["filename"]).suffix.lower() != ".pt" or not HEX64.fullmatch(item["sha256"]):
            raise RuntimeError(f"Invalid historical weight provenance: {item}")

    print(
        json.dumps(
            {
                "status": "ok",
                "checked": checked,
                "external_source_checked": args.check_source,
                "git_head_checked": args.check_git,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
