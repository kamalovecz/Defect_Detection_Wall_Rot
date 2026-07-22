from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\defect_detection\repo_staging_b6_b7_300ep")
OLD_ROOT = Path(r"D:\defect_detection\repo_staging_ablation_300ep")
DATA = Path(r"D:\defect_detection\ultralytics-main\dataset\NEU-DET_YOLO\neu-det.yaml")
VAL_SCRIPT = Path(r"D:\defect_detection\ultralytics-main\val.py")
PYTHON = Path(r"D:\Anaconda\envs\harpnet_acceptance\python.exe")
OLD_QUEUE_01 = OLD_ROOT / "training_project/runs/neu_det_ablation_300ep_b8_20260721/queue_state.json"
OLD_QUEUE_25 = OLD_ROOT / "training_project/runs/neu_det_ablation_300ep_b8_20260722_from_B2_p0/queue_state.json"
NEW_EVAL = ROOT / "training_project/runs/neu_det_b6_b7_pt_val_test_20260722/queue_state.json"
OUTPUT = OLD_ROOT / "training_project/runs/neu_det_a0_b5_test_20260722"
REPORT = ROOT / "training_project/ablations/reports/NEU_DET_A0_B1_B7_SUMMARY.md"
MANIFEST = OUTPUT / "summary_manifest.json"
MODELS = ("A0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
DESCRIPTIONS = {
    "A0": "YOLOv8n baseline",
    "B1": "SADH",
    "B2": "RepHFE",
    "B3": "A-GFPN",
    "B4": "A-GFPN + RepHFE",
    "B5": "A-GFPN + RepHFE + SADH",
    "B6": "A-GFPN + SADH",
    "B7": "RepHFE + SADH",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(path: Path) -> dict[str, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "all(mean)" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) == 7:
            values = [float(value) for value in parts[1:]]
            return dict(zip(("precision", "recall", "f1", "map50", "map75", "map50_95"), values))
    raise RuntimeError(f"Missing all(mean) row: {path}")


def validate_epochs(run_dir: Path) -> None:
    results = run_dir / "results.csv"
    manifest = run_dir / "run_manifest.json"
    if not results.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Missing formal training evidence: {run_dir}")
    with results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [{key.strip(): value.strip() for key, value in row.items()} for row in rows]
    epochs = [int(float(row["epoch"])) for row in normalized]
    if len(rows) != 300 or epochs != list(range(1, 301)):
        raise RuntimeError(f"Training is not exactly 300 epochs: {run_dir}")


def metric_row(model: str, split: str, metrics: dict[str, float], baseline: dict[str, float]) -> str:
    delta50 = metrics["map50"] - baseline["map50"]
    delta95 = metrics["map50_95"] - baseline["map50_95"]
    return (
        f"| {model} | {DESCRIPTIONS[model]} | {split} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['map50']:.4f} | "
        f"{metrics['map75']:.4f} | {metrics['map50_95']:.4f} | {delta50:+.4f} | {delta95:+.4f} |"
    )


def main() -> int:
    q01, q25, new = load(OLD_QUEUE_01), load(OLD_QUEUE_25), load(NEW_EVAL)
    expected_data_hash = "bc7d42864f93408547d5c03487f5b619feeaf511478b6e35e5aaa41c6320f234"
    expected_val_hash = "c2aab95c65633147bbb74bd9dd1c7a6d89f061a2a45bd06a6a1e6c17d6539f26"
    if sha256(DATA) != expected_data_hash or sha256(VAL_SCRIPT) != expected_val_hash:
        raise RuntimeError("Dataset descriptor or validator hash changed")
    if q25.get("status") != "passed" or new.get("status") != "passed":
        raise RuntimeError("B2-B7 evidence is not passed")

    records: dict[str, dict] = {}
    for model in MODELS[:6]:
        queue = q01 if model in ("A0", "B1") else q25
        experiment = queue["experiments"][model]
        if experiment.get("status") != "passed":
            raise RuntimeError(f"{model} formal experiment is not passed")
        run_dir = OLD_ROOT / experiment["run_dir"]
        checkpoint = OLD_ROOT / experiment["checkpoint"]
        val_data = OLD_ROOT / experiment["paper_data"]
        validate_epochs(run_dir)
        if sha256(checkpoint) != experiment["checkpoint_sha256"]:
            raise RuntimeError(f"{model} checkpoint hash mismatch")
        if sha256(val_data) != experiment["paper_data_sha256"]:
            raise RuntimeError(f"{model} val paper_data hash mismatch")
        records[model] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(checkpoint),
            "val_path": val_data,
            "val": parse(val_data),
            "patience": "default (completed 300)" if model in ("A0", "B1") else 0,
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "ultralytics-main")))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS[:6]:
        model_root = OUTPUT / model
        paper = model_root / "test/paper_data.txt"
        log = model_root / "test.log"
        if not paper.is_file():
            model_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON), str(VAL_SCRIPT), "--model-path", str(records[model]["checkpoint"]),
                "--data", str(DATA), "--split", "test", "--imgsz", "640", "--batch", "8",
                "--project", str(model_root), "--name", "test", "--exist-ok",
            ]
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{model} test validation failed with exit code {result.returncode}")
        records[model]["test_path"] = paper
        records[model]["test"] = parse(paper)

    for model in MODELS[6:]:
        experiment = new["experiments"][model]
        records[model] = {"patience": 0, "checkpoint_sha256": experiment["checkpoint_sha256"]}
        for split in ("val", "test"):
            item = experiment["splits"][split]
            path = ROOT / item["paper_data"]
            if sha256(path) != item["paper_data_sha256"]:
                raise RuntimeError(f"{model} {split} paper_data hash mismatch")
            records[model][f"{split}_path"] = path
            records[model][split] = parse(path)

    header = "| Model | Structure | Split | P | R | F1 | mAP50 | mAP75 | mAP50-95 | ΔmAP50 vs A0 | ΔmAP50-95 vs A0 |"
    separator = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# NEU-DET A0、B1–B7 消融实验汇总",
        "",
        "## 统一实验合同",
        "",
        "- 数据集：NEU-DET_YOLO（train=1259，val=359，test=181；跨划分重复图像=0）",
        "- 正式训练：300 epochs，batch=8，imgsz=640，seed=42，scratch，AMP=false，RuleLoss=false",
        "- 评估：同一外部 val.py，batch=8，imgsz=640；每个 best.pt 独立运行 val/test",
        "- A0/B1 使用默认 patience，但实际完整运行 300 epochs；B2–B7 明确 patience=0",
        "- 项目不存在 B0；本文按实际消融编号汇总 A0 与 B1–B7",
        "",
        "## 验证集汇总",
        "",
        header,
        separator,
    ]
    for model in MODELS:
        lines.append(metric_row(model, "val", records[model]["val"], records["A0"]["val"]))
    lines.extend(["", "## 测试集汇总（论文最终泛化指标）", "", header, separator])
    for model in MODELS:
        lines.append(metric_row(model, "test", records[model]["test"], records["A0"]["test"]))

    best_val = max(MODELS, key=lambda model: records[model]["val"]["map50_95"])
    best_test = max(MODELS, key=lambda model: records[model]["test"]["map50_95"])
    best_test50 = max(MODELS, key=lambda model: records[model]["test"]["map50"])
    best_test_precision = max(MODELS, key=lambda model: records[model]["test"]["precision"])
    best_test_recall = max(MODELS, key=lambda model: records[model]["test"]["recall"])
    best_test_f1 = max(MODELS, key=lambda model: records[model]["test"]["f1"])
    best_test75 = max(MODELS, key=lambda model: records[model]["test"]["map75"])
    lines.extend([
        "",
        "## 结论",
        "",
        f"1. 验证集 mAP50-95 最高为 {best_val}（{records[best_val]['val']['map50_95']:.4f}）。",
        f"2. 测试集 mAP50-95 最高为 {best_test}（{records[best_test]['test']['map50_95']:.4f}）。",
        f"3. 测试集 mAP50 最高为 {best_test50}（{records[best_test50]['test']['map50']:.4f}）。",
        f"4. 测试集 Precision、Recall、F1、mAP75 最优项依次为 {best_test_precision}、{best_test_recall}、{best_test_f1}、{best_test75}，说明不同结构存在明确的精确率—召回率权衡。",
        "5. 仅 B3 的测试集 mAP50-95 高于 A0（+0.0063），但其 mAP50 低 0.0027；不能声称所有改进模块或完整组合都稳定优于基线。",
        "6. 当前仅包含 seed=42；表中差值是单次实验差值，不能表述为统计显著提升。建议至少对 A0、B3、B5 运行 3 个种子并报告 mean±std。",
        "",
        "## 证据清单",
        "",
        "| Model | Split | paper_data SHA-256 | checkpoint SHA-256 |",
        "|---|---|---|---|",
    ])
    evidence = {}
    for model in MODELS:
        evidence[model] = {"checkpoint_sha256": records[model]["checkpoint_sha256"], "splits": {}}
        for split in ("val", "test"):
            path = records[model][f"{split}_path"]
            paper_hash = sha256(path)
            evidence[model]["splits"][split] = {"paper_data": str(path), "paper_data_sha256": paper_hash, "metrics": records[model][split]}
            lines.append(f"| {model} | {split} | `{paper_hash}` | `{records[model]['checkpoint_sha256']}` |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "status": "passed",
        "models": list(MODELS),
        "note": "B0 does not exist; actual sequence is A0 and B1-B7",
        "contract": {"epochs": 300, "batch": 8, "imgsz": 640, "seed": 42, "data_sha256": expected_data_hash, "val_script_sha256": expected_val_hash},
        "evidence": evidence,
        "report": str(REPORT),
        "report_sha256": sha256(REPORT),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(REPORT), "best_val": best_val, "best_test": best_test, "best_test50": best_test50}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
