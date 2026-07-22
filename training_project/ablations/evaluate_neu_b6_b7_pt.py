from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_PROJECT = ROOT / "training_project/runs/neu_det_ablation_b6_b7_300ep_b8_20260722_p0"
QUEUE_STATE = RUN_PROJECT / "queue_state.json"
OUTPUT_ROOT = ROOT / "training_project/runs/neu_det_b6_b7_pt_val_test_20260722"
REPORT_ROOT = ROOT / "training_project/ablations/reports/neu_det_b6_b7_val_test"
DATA = Path(r"D:\defect_detection\ultralytics-main\dataset\NEU-DET_YOLO\neu-det.yaml")
VAL_SCRIPT = Path(r"D:\defect_detection\ultralytics-main\val.py")
MODELS = ("B6", "B7")
SPLITS = ("val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def validation_environment() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "ultralytics-main")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def import_preflight(env: dict[str, str], expected_ultralytics: Path) -> dict[str, str]:
    command = [
        sys.executable,
        "-c",
        (
            "import json, pathlib, defect_modules, ultralytics; "
            "print(json.dumps({'defect_modules': str(pathlib.Path(defect_modules.__file__).resolve()), "
            "'ultralytics': str(pathlib.Path(ultralytics.__file__).resolve().parent)}))"
        ),
    ]
    observed = json.loads(
        subprocess.check_output(command, cwd=VAL_SCRIPT.parent, env=env, text=True).strip()
    )
    expected_modules = (ROOT / "defect_modules/__init__.py").resolve()
    if Path(observed["defect_modules"]) != expected_modules:
        raise RuntimeError(f"Unexpected defect_modules import: {observed['defect_modules']}")
    if Path(observed["ultralytics"]) != expected_ultralytics.resolve():
        raise RuntimeError(f"Unexpected Ultralytics import: {observed['ultralytics']}")
    return observed


def validate_training_run(model_id: str, queue: dict) -> dict:
    experiment = queue["experiments"][model_id]
    if experiment.get("status") != "passed":
        raise RuntimeError(f"{model_id} is not a passed formal experiment")
    run_dir = ROOT / experiment["run_dir"]
    checkpoint = ROOT / experiment["checkpoint"]
    results = run_dir / "results.csv"
    manifest_path = run_dir / "run_manifest.json"
    for path in (checkpoint, results, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing formal artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = manifest["config"]["train"]
    expected = {
        "status": "completed",
        "epochs": 300,
        "batch": 8,
        "imgsz": 640,
        "seed": 42,
        "patience": 0,
        "amp": False,
        "pretrained": False,
        "rule_loss": False,
    }
    observed = {
        "status": manifest.get("status"),
        "epochs": train.get("epochs"),
        "batch": train.get("batch"),
        "imgsz": train.get("imgsz"),
        "seed": train.get("seed"),
        "patience": train.get("patience"),
        "amp": train.get("amp"),
        "pretrained": train.get("pretrained"),
        "rule_loss": manifest["rule_loss"]["enabled"],
    }
    if observed != expected:
        raise RuntimeError(f"{model_id} manifest contract mismatch: {observed}")
    if manifest.get("model_yaml") != experiment["model"]:
        raise RuntimeError(f"{model_id} model YAML path mismatch")
    if manifest.get("model_yaml_sha256") != experiment["model_sha256"]:
        raise RuntimeError(f"{model_id} model YAML hash mismatch")
    with results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise RuntimeError(f"{model_id} results.csv has {len(rows)} rows, expected 300")
    normalized = [{key.strip(): value.strip() for key, value in row.items()} for row in rows]
    epochs = [int(float(row["epoch"])) for row in normalized]
    if epochs != list(range(1, 301)):
        raise RuntimeError(f"{model_id} epoch sequence is not exactly 1..300")
    for row_index, row in enumerate(normalized, start=1):
        for key, value in row.items():
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(f"{model_id} non-finite results.csv value at row {row_index}: {key}={value}")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != experiment["checkpoint_sha256"]:
        raise RuntimeError(f"{model_id} checkpoint hash mismatch")
    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "results": results,
        "results_sha256": sha256(results),
        "manifest": manifest_path,
        "manifest_sha256": sha256(manifest_path),
    }


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def parse_paper_data(path: Path) -> dict:
    rows: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 7:
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        rows[parts[0]] = {
            "precision": values[0],
            "recall": values[1],
            "f1": values[2],
            "map50": values[3],
            "map75": values[4],
            "map50_95": values[5],
        }
    if "all(mean)" not in rows:
        raise RuntimeError(f"Could not parse all(mean) metrics from {path}")
    return {"classes": rows, "overall": rows["all(mean)"]}


def metric_row(label: str, metrics: dict[str, float]) -> str:
    return (
        f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
        f"{metrics['f1']:.4f} | {metrics['map50']:.4f} | {metrics['map75']:.4f} | "
        f"{metrics['map50_95']:.4f} |"
    )


def write_report(model_id: str, training: dict, evaluations: dict[str, dict], queue: dict) -> Path:
    val_metrics = evaluations["val"]["metrics"]
    test_metrics = evaluations["test"]["metrics"]
    delta = {
        key: test_metrics["overall"][key] - val_metrics["overall"][key]
        for key in val_metrics["overall"]
    }
    classes = [name for name in val_metrics["classes"] if name != "all(mean)"]
    lines = [
        f"# {model_id} 在 NEU-DET 验证集与测试集上的 PT 性能报告",
        "",
        "## 实验契约",
        "",
        f"- 模型：`{model_id}`",
        "- 正式训练：300 epochs，batch=8，imgsz=640，seed=42，patience=0",
        "- 权重初始化：scratch；AMP：关闭；RuleLoss：关闭",
        f"- 数据集：`{queue['dataset_contract']['dataset']}`",
        f"- 数据描述：`{DATA}`",
        f"- 检查点：`{relative(training['checkpoint'])}`",
        f"- 检查点 SHA-256：`{training['checkpoint_sha256']}`",
        "",
        "## 总体性能",
        "",
        "| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("val", val_metrics["overall"]),
        metric_row("test", test_metrics["overall"]),
        metric_row("test - val", delta),
        "",
        "## 分类别性能",
        "",
        "| Class | Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name in classes:
        for split in SPLITS:
            metrics = evaluations[split]["metrics"]["classes"][class_name]
            row = metric_row(split, metrics).strip("|").strip()
            lines.append(f"| {class_name} | {row} |")
    lines.extend(
        [
            "",
            "## 产物与哈希",
            "",
            "| Artifact | Path | SHA-256 |",
            "|---|---|---|",
            f"| results.csv | `{relative(training['results'])}` | `{training['results_sha256']}` |",
            f"| run_manifest.json | `{relative(training['manifest'])}` | `{training['manifest_sha256']}` |",
        ]
    )
    for split in SPLITS:
        item = evaluations[split]
        lines.append(f"| {split} paper_data | `{relative(item['paper_data'])}` | `{item['paper_data_sha256']}` |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 本报告使用同一 `best.pt` 分别在官方 val/test split 上独立评估。",
            "- 当前结果仅包含 seed=42，不能用于估计跨随机种子的均值和标准差。",
            "- test 指标用于最终泛化能力陈述；val 指标用于模型选择与消融过程比较。",
            "",
        ]
    )
    report = REPORT_ROOT / f"{model_id}_NEU_DET_VAL_TEST.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output root already exists: {OUTPUT_ROOT}")
    if REPORT_ROOT.exists():
        raise FileExistsError(f"Report root already exists: {REPORT_ROOT}")
    queue = json.loads(QUEUE_STATE.read_text(encoding="utf-8-sig"))
    if queue.get("status") != "passed" or tuple(queue.get("order", ())) != MODELS:
        raise RuntimeError("B6-B7 formal queue has not passed with the expected order")
    if DATA.resolve() != Path(queue["contract"]["data"]).resolve():
        raise RuntimeError("Dataset path does not match the formal queue contract")
    if sha256(DATA) != queue["dataset_contract"]["descriptor_sha256"]:
        raise RuntimeError("Dataset descriptor hash mismatch")
    if sha256(VAL_SCRIPT) != queue["contract"]["val_script_sha256"]:
        raise RuntimeError("External val.py hash mismatch")
    env = validation_environment()
    imports = import_preflight(env, Path(queue["dataset_contract"]["validator"]["python_tree_root"]))
    training = {model_id: validate_training_run(model_id, queue) for model_id in MODELS}

    OUTPUT_ROOT.mkdir(parents=True)
    REPORT_ROOT.mkdir(parents=True)
    state_path = OUTPUT_ROOT / "queue_state.json"
    manifest: dict = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "splits": list(SPLITS),
        "contract": {
            "data": str(DATA),
            "data_sha256": sha256(DATA),
            "val_script": str(VAL_SCRIPT),
            "val_script_sha256": sha256(VAL_SCRIPT),
            "imgsz": 640,
            "batch": 8,
            "python": sys.executable,
            "imports": imports,
        },
        "experiments": {},
    }
    atomic_json(state_path, manifest)
    try:
        for model_id in MODELS:
            model_output = OUTPUT_ROOT / model_id
            model_output.mkdir()
            evaluations: dict[str, dict] = {}
            manifest["experiments"][model_id] = {"status": "running", "splits": {}}
            atomic_json(state_path, manifest)
            for split in SPLITS:
                manifest["experiments"][model_id]["splits"][split] = {"status": "running"}
                atomic_json(state_path, manifest)
                command = [
                    sys.executable,
                    str(VAL_SCRIPT),
                    "--model-path",
                    str(training[model_id]["checkpoint"]),
                    "--data",
                    str(DATA),
                    "--split",
                    split,
                    "--imgsz",
                    "640",
                    "--batch",
                    "8",
                    "--project",
                    str(model_output),
                    "--name",
                    split,
                    "--exist-ok",
                ]
                log_path = model_output / f"{split}.log"
                run_logged(command, log_path, env)
                paper_data = model_output / split / "paper_data.txt"
                if not paper_data.is_file():
                    raise FileNotFoundError(f"Missing paper_data: {paper_data}")
                evaluations[split] = {
                    "paper_data": paper_data,
                    "paper_data_sha256": sha256(paper_data),
                    "log": log_path,
                    "log_sha256": sha256(log_path),
                    "metrics": parse_paper_data(paper_data),
                }
                manifest["experiments"][model_id]["splits"][split] = {
                    "status": "passed",
                    "paper_data": relative(paper_data),
                    "paper_data_sha256": evaluations[split]["paper_data_sha256"],
                    "log": relative(log_path),
                    "log_sha256": evaluations[split]["log_sha256"],
                    "metrics": evaluations[split]["metrics"]["overall"],
                }
                atomic_json(state_path, manifest)
            report = write_report(model_id, training[model_id], evaluations, queue)
            manifest["experiments"][model_id].update(
                {
                    "status": "passed",
                    "checkpoint": relative(training[model_id]["checkpoint"]),
                    "checkpoint_sha256": training[model_id]["checkpoint_sha256"],
                    "report": relative(report),
                    "report_sha256": sha256(report),
                }
            )
            atomic_json(state_path, manifest)
        manifest["status"] = "passed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
