from __future__ import annotations

import html
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

import altair as alt
import h5py
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve

from decision_calibration import (
    ATYPIA_BOUNDARY_MARGIN,
    DEFAULT_ATYPIA_THRESHOLD,
    PROB_COLS,
    REVIEW_MARGIN_THRESHOLD,
    add_confidence_columns as calibration_confidence_columns,
    add_operating_point_columns,
    atypia_threshold_predictions,
    high_uncertainty_queue,
)
from live_roi_inference import predict_roi_image, result_frame as live_prediction_frame


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

ROOT = Path(__file__).resolve().parent


def resolve_data_dir() -> Path:
    env_path = os.environ.get("BRACS_DATA_DIR") or os.environ.get("BRACS_DATASET_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            ROOT / "data_full" / "latest_version",
            ROOT / "latest_version",
            ROOT.parent / "latest_version",
        ]
    )
    for path in candidates:
        if (path / "train").exists() and (path / "val").exists() and (path / "test").exists():
            return path
    return candidates[0]


DATA_DIR = resolve_data_dir()
WORK_DIR = ROOT / "work"
OUTPUT_DIR = WORK_DIR / "transmil_dtfd_output"
FEATURE_CACHE = WORK_DIR / "bracs_features.h5"
FEATURE_TMP = WORK_DIR / "bracs_features.h5.tmp"
WEIGHTS = ROOT / "weights" / "pytorch_model.bin"
LOCAL_NOTEBOOK = ROOT / "BRACS_TransMIL_DTFD_LOCAL.ipynb"
EXECUTED_NOTEBOOK = ROOT / "BRACS_TransMIL_DTFD_LOCAL.executed.ipynb"
EXPECTED_MODELS = 15

CLASS_NAMES = {
    "0_N": "Normal",
    "1_PB": "Pathological benign",
    "2_UDH": "Usual ductal hyperplasia",
    "3_FEA": "Flat epithelial atypia",
    "4_ADH": "Atypical ductal hyperplasia",
    "5_DCIS": "Ductal carcinoma in situ",
    "6_IC": "Invasive carcinoma",
}

THREE_CLASS = {
    0: "Benign",
    1: "Atypia",
    2: "Malignant",
    "0": "Benign",
    "1": "Atypia",
    "2": "Malignant",
}

PAGES = [
    "Live Demo",
    "Results",
    "Blind Pool",
    "System",
]
LIVE_AUTO_REFRESH_PAGES: set[str] = set()

PAGE_SUBTITLES = {
    "Live Demo": "Review exact 3636 ROI cases or upload a separate ROI for live prediction.",
    "Results": "Final validation metrics and evaluation artifacts.",
    "Blind Pool": "Anonymized held-out images for panel selection.",
    "System": "Local artifact and environment checks.",
}

MODULE_GUIDES = {
    "Live Demo": [
        ("Prediction", "model output for the uploaded ROI."),
        ("Confidence", "probability assigned to the predicted class."),
        ("Margin", "gap between the two highest class probabilities."),
        ("Review", "yes when the model confidence or class margin is low."),
    ],
    "System": [
        ("Artifacts", "checks that dataset, features, weights, and final outputs are available."),
        ("Device", "shows whether local PyTorch can use Apple MPS acceleration."),
    ],
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172026;
            --muted: #5b6770;
            --line: #d8dee4;
            --paper: #f6f8f9;
            --panel: #ffffff;
            --teal: #0f7c80;
            --teal-soft: #e8f4f4;
            --coral: #c8553d;
            --amber: #a66f00;
            --green: #2e7d32;
            --shadow: 0 1px 2px rgba(23,32,38,.05);
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(15,124,128,.10), rgba(244,247,248,0) 260px),
                #f4f7f8;
        }
        .block-container {
            padding-top: 1.05rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        #MainMenu,
        footer,
        header,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0;
        }
        .hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
            padding: 22px 24px;
            margin-bottom: 18px;
            box-shadow: var(--shadow);
        }
        .hero-title {
            font-size: 32px;
            line-height: 1.1;
            font-weight: 780;
            color: var(--ink);
            margin-bottom: 7px;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 15px;
            max-width: 980px;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 18px;
        }
        .hero-stat {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfcfd;
            padding: 11px 12px;
            min-width: 0;
        }
        .hero-stat-label {
            color: var(--muted);
            font-size: 11px;
            font-weight: 730;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 4px;
        }
        .hero-stat-value {
            color: var(--ink);
            font-size: 15px;
            font-weight: 740;
            overflow-wrap: anywhere;
        }
        .status-pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #fff;
            color: var(--ink);
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .status-running { border-color: rgba(46,125,50,.35); color: var(--green); }
        .status-warn { border-color: rgba(166,111,0,.35); color: var(--amber); }
        .status-stop { border-color: rgba(200,85,61,.35); color: var(--coral); }
        .section-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 16px 18px;
            margin: 10px 0 16px;
        }
        .page-heading {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #f7fbfb 100%);
            padding: 16px 18px;
            margin: 4px 0 14px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--teal);
        }
        .page-title {
            color: var(--ink);
            font-size: 24px;
            line-height: 1.15;
            font-weight: 770;
            margin: 0;
        }
        .page-subtitle {
            color: var(--muted);
            font-size: 14px;
            margin-top: 5px;
            max-width: 980px;
        }
        .tight-note {
            color: var(--muted);
            font-size: 13px;
            margin-top: -6px;
            margin-bottom: 10px;
        }
        .file-chip {
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 8px 10px;
            background: #fbfcfd;
            font-size: 13px;
            color: var(--ink);
        }
        .top-menu-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
            padding: 14px 16px 12px;
            margin-bottom: 14px;
            box-shadow: var(--shadow);
        }
        .app-kicker {
            color: var(--teal);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 760;
            margin-bottom: 2px;
        }
        .app-title {
            color: var(--ink);
            font-size: 26px;
            line-height: 1.15;
            font-weight: 780;
            margin-bottom: 2px;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 13px;
        }
        .top-menu-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .refresh-note {
            color: var(--muted);
            font-size: 12px;
            margin-top: 2px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfcfd 100%);
            min-width: 0;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(23,32,38,.05);
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(15,124,128,.28);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 13px;
            white-space: normal;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.35rem;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        div[data-testid="column"] {
            min-width: 0;
        }
        div[data-testid="stSegmentedControl"] {
            width: 100%;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border: 1px solid var(--line);
            border-radius: 6px;
            background: #fff;
            padding: 8px 12px;
        }
        .stTabs [aria-selected="true"] {
            border-color: rgba(15,124,128,.5);
            color: var(--teal);
        }
        @media (max-width: 900px) {
            .hero {
                padding: 16px;
            }
            .hero-title {
                font-size: 24px;
            }
            .hero-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 560px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .app-title {
                font-size: 19px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_bytes(size: int | float | None) -> str:
    if size is None:
        return "missing"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def path_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def human_size(path: Path) -> str:
    return fmt_bytes(path_size(path))


def modified_label(path: Path) -> str:
    if not path.exists():
        return "not found"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def command_output(args: list[str], timeout: int = 4) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"Unavailable: {exc}"


def runtime_label(seconds: int | float | str | None) -> str:
    seconds_num = pd.to_numeric(seconds, errors="coerce")
    if pd.isna(seconds_num):
        return "n/a"
    seconds_int = int(seconds_num)
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def metric_grid(items: list[tuple[str, str]], columns_per_row: int = 3) -> None:
    for start in range(0, len(items), columns_per_row):
        row = items[start : start + columns_per_row]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            col.metric(label, value)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def page_heading(page: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="page-title">{page}</div>
            <div class="page-subtitle">{PAGE_SUBTITLES.get(page, "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def module_guide(page: str) -> None:
    rows = MODULE_GUIDES.get(page, [])
    if not rows:
        return
    with st.expander("Metric guide", expanded=False):
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric / Area", "How to interpret it"]),
            width="stretch",
            hide_index=True,
        )


def query_value(key: str, default: str) -> str:
    value = st.query_params.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value or default


def page_from_query() -> str:
    page = query_value("page", PAGES[0])
    return page if page in PAGES else PAGES[0]


def safe_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


@st.cache_data(ttl=20)
def dataset_counts() -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            rows.append(
                {
                    "split": split,
                    "class_folder": class_dir.name,
                    "class": CLASS_NAMES.get(class_dir.name, class_dir.name),
                    "pngs": len(list(class_dir.glob("*.png"))),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=15)
def metadata_frame(name: str) -> pd.DataFrame | None:
    path = WORK_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(ttl=15)
def feature_cache_stats() -> dict:
    path = FEATURE_CACHE if FEATURE_CACHE.exists() else FEATURE_TMP
    stats = {
        "path": path,
        "exists": path.exists(),
        "is_final": FEATURE_CACHE.exists(),
        "size": path_size(path),
        "slides": None,
        "datasets": 0,
        "patches": 0,
        "feature_dim": None,
        "scales": {},
        "error": None,
    }
    if not path.exists() or path == FEATURE_TMP:
        return stats
    try:
        with h5py.File(path, "r") as h5:
            stats["slides"] = len(h5.keys())
            scales: Counter[str] = Counter()
            total_patches = 0
            feature_dim = None
            dataset_count = 0
            for group in h5.values():
                if not hasattr(group, "items"):
                    continue
                for name, ds in group.items():
                    if name == "labels" or not hasattr(ds, "shape"):
                        continue
                    if len(ds.shape) >= 2:
                        dataset_count += 1
                        total_patches += int(ds.shape[0])
                        feature_dim = int(ds.shape[1])
                        scales[name.replace("features_", "")] += int(ds.shape[0])
            stats.update(
                {
                    "datasets": dataset_count,
                    "patches": total_patches,
                    "feature_dim": feature_dim,
                    "scales": dict(scales),
                }
            )
    except Exception as exc:
        stats["error"] = str(exc)
    return stats


@st.cache_data(ttl=10)
def gpu_table() -> pd.DataFrame:
    out = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "gpu": parts[0],
                "name": parts[1],
                "temp_c": pd.to_numeric(parts[2], errors="coerce"),
                "util_pct": pd.to_numeric(parts[3], errors="coerce"),
                "mem_used_mb": pd.to_numeric(parts[4], errors="coerce"),
                "mem_total_mb": pd.to_numeric(parts[5], errors="coerce"),
                "power_w": pd.to_numeric(parts[6], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=10)
def notebook_processes() -> pd.DataFrame:
    out = command_output(["ps", "-eo", "pid=,etimes=,pcpu=,pmem=,cmd="])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        pid, etimes, cpu, mem, cmd = parts
        if LOCAL_NOTEBOOK.name not in cmd or "nbconvert" not in cmd:
            continue
        if "streamlit_app.py" in cmd or "pgrep" in cmd:
            continue
        rows.append(
            {
                "pid": pid,
                "runtime": runtime_label(etimes),
                "cpu_pct": pd.to_numeric(cpu, errors="coerce"),
                "mem_pct": pd.to_numeric(mem, errors="coerce"),
                "command": cmd,
            }
        )
    return pd.DataFrame(rows)


def notebook_process_text() -> str:
    processes = notebook_processes()
    if processes.empty:
        return ""
    return "\n".join(f"{row.pid} {row.runtime} {row.command}" for row in processes.itertuples(index=False))


@st.cache_data(ttl=60)
def image_shape_sample(split: str, class_folder: str, limit: int = 80) -> pd.DataFrame:
    folder = DATA_DIR / split / class_folder
    rows = []
    for path in sorted(folder.glob("*.png"))[:limit]:
        try:
            with Image.open(path) as img:
                rows.append({"name": path.name, "width": img.width, "height": img.height, "mode": img.mode})
        except Exception:
            rows.append({"name": path.name, "width": None, "height": None, "mode": "unreadable"})
    return pd.DataFrame(rows)


def checkpoints() -> list[Path]:
    manifest = latest_checkpoint_manifest()
    if manifest is not None:
        paths = checkpoint_paths_from_manifest(manifest)
        if paths:
            return paths
    paths = list(OUTPUT_DIR.glob("transmil_seed*.pt"))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0)


def validation_output_dirs() -> list[Path]:
    return sorted(
        [p for p in WORK_DIR.glob("transmil_dtfd_output_validation_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def checkpoint_paths_from_manifest(manifest: Path) -> list[Path]:
    try:
        rows = pd.read_csv(manifest).sort_values("val_f1", ascending=False)
        return [Path(p) for p in rows["path"].tolist() if Path(p).exists()]
    except Exception:
        return []


def latest_checkpoint_manifest() -> Path | None:
    manifests = sorted(
        WORK_DIR.glob("transmil_dtfd_output_validation_*/main_checkpoint_manifest.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    complete = [p for p in manifests if len(checkpoint_paths_from_manifest(p)) >= EXPECTED_MODELS]
    if complete:
        return complete[0]
    return manifests[0] if manifests else None


def current_validation_output_dir() -> Path | None:
    manifest = latest_checkpoint_manifest()
    if manifest is not None:
        return manifest.parent
    dirs = validation_output_dirs()
    return dirs[0] if dirs else None


def available_live_checkpoints() -> list[Path]:
    manifest = latest_checkpoint_manifest()
    if manifest is not None:
        paths = checkpoint_paths_from_manifest(manifest)
        if paths:
            return paths
    return sorted(OUTPUT_DIR.glob("transmil_seed*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def artifact_files() -> list[Path]:
    paths = []
    for base in [OUTPUT_DIR, WORK_DIR]:
        if base.exists():
            paths.extend([p for p in base.rglob("*") if p.is_file()])
    paths.extend([WEIGHTS, LOCAL_NOTEBOOK, EXECUTED_NOTEBOOK])
    return sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def artifact_table(paths: Iterable[Path], limit: int = 30) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        rows.append(
            {
                "name": path.name,
                "kind": path.suffix.lstrip(".") or "file",
                "size": human_size(path),
                "modified": modified_label(path),
                "path": display_path(path),
            }
        )
    return pd.DataFrame(rows).head(limit)


def validation_run_table() -> pd.DataFrame:
    rows = []
    for root in sorted(WORK_DIR.glob("transmil_dtfd_output_validation_*")):
        if not root.is_dir():
            continue
        summaries = sorted(root.glob("validation_*x*_summary.json"))
        if not summaries:
            summaries = sorted(root.glob("validation_1x5_summary.json"))
        summary_payload = {}
        summary_path = summaries[-1] if summaries else None
        if summary_path:
            try:
                summary_payload = json.loads(summary_path.read_text(errors="replace"))
            except Exception:
                summary_payload = {}
        n_seeds = int(summary_payload.get("n_seeds") or 0)
        n_folds = int(summary_payload.get("n_folds") or 0)
        root_token = root.name.replace("transmil_dtfd_output_validation_", "").split("_", 1)[0]
        if (not n_seeds or not n_folds) and "x" in root_token:
            try:
                parsed_seeds, parsed_folds = root_token.split("x", 1)
                n_seeds = n_seeds or int(parsed_seeds)
                n_folds = n_folds or int(parsed_folds)
            except ValueError:
                pass
        label = f"{n_seeds}x{n_folds}" if n_seeds and n_folds else root_token
        final_pred = root / "posthoc_metric_optimized" / "metric_optimized_predictions.csv"
        test_pred = root / f"validation_{label}_test_predictions.csv"
        if not test_pred.exists():
            test_pred = root / "validation_1x5_test_predictions.csv"
        metrics = summary_payload.get("heldout_test_metrics") or {}
        raw_metrics = summary_payload.get("raw_heldout_test_metrics") or {}
        if final_pred.exists():
            try:
                final_df = pd.read_csv(final_pred)
                final_metric_frame = class_metric_frame(final_df)
                metrics = {
                    "accuracy": float((final_df["true_label"].astype(int) == final_df["pred_label"].astype(int)).mean()),
                    "macro_f1": float(final_metric_frame["f1"].mean()),
                }
            except Exception:
                pass
        rows.append(
            {
                "run": label,
                "seeds": n_seeds or "",
                "folds": n_folds or "",
                "models": n_seeds * n_folds if n_seeds and n_folds else "",
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "raw_accuracy": raw_metrics.get("accuracy"),
                "raw_macro_f1": raw_metrics.get("macro_f1"),
                "final_predictions": display_path(final_pred) if final_pred.exists() else "",
                "test_predictions": display_path(test_pred) if test_pred.exists() else "",
                "modified": modified_label(root),
                "path": display_path(root),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run",
                "seeds",
                "folds",
                "models",
                "accuracy",
                "macro_f1",
                "raw_accuracy",
                "raw_macro_f1",
                "final_predictions",
                "test_predictions",
                "modified",
                "path",
            ]
        )
    table = pd.DataFrame(rows)
    table["_sort_models"] = pd.to_numeric(table["models"], errors="coerce").fillna(0)
    table["_sort_time"] = table["path"].map(lambda rel: (ROOT / rel).stat().st_mtime if (ROOT / rel).exists() else 0)
    return table.sort_values(["_sort_models", "_sort_time"], ascending=False).drop(columns=["_sort_models", "_sort_time"])


def preferred_validation_prediction_path() -> Path | None:
    runs = validation_run_table()
    if runs.empty:
        return None
    for _, row in runs.iterrows():
        for column in ("final_predictions", "test_predictions"):
            value = str(row.get(column) or "")
            if value:
                path = ROOT / value
                if path.exists():
                    return path
    return None


def status_payload() -> dict:
    counts = dataset_counts()
    stats = feature_cache_stats()
    phase, state, progress = pipeline_phase()
    return {
        "phase": phase,
        "state": state,
        "progress_estimate": round(progress, 3),
        "dataset_pngs": int(counts["pngs"].sum()) if not counts.empty else 0,
        "feature_cache": {
            "exists": bool(stats["exists"]),
            "is_final": bool(stats["is_final"]),
            "size_bytes": stats["size"],
            "slides": stats["slides"],
            "patches": stats["patches"],
            "feature_dim": stats["feature_dim"],
        },
        "checkpoints": len(checkpoints()),
        "expected_checkpoints": EXPECTED_MODELS,
        "notebook_processes": notebook_processes().to_dict(orient="records"),
        "gpus": gpu_table().to_dict(orient="records"),
    }


def pipeline_phase() -> tuple[str, str, float]:
    proc_live = bool(notebook_process_text())
    ckpts = checkpoints()
    if FEATURE_TMP.exists():
        return "Extracting UNI features", "running" if proc_live else "warn", 0.20
    if FEATURE_CACHE.exists() and len(ckpts) < EXPECTED_MODELS and proc_live:
        if not ckpts:
            return "Training first MIL model: checkpoint pending", "running", 0.35
        progress = 0.35 + min(len(ckpts), EXPECTED_MODELS) / EXPECTED_MODELS * 0.35
        return f"Training MIL ensemble: {len(ckpts)}/{EXPECTED_MODELS} checkpoints", "running", progress
    if len(ckpts) >= EXPECTED_MODELS and proc_live:
        return "Evaluating and exporting results", "running", 0.82
    if (OUTPUT_DIR / "results_summary.txt").exists():
        return "Completed", "running" if proc_live else "ok", 1.0
    if proc_live:
        return "Preparing pipeline", "running", 0.12
    return "Stopped or waiting", "stop", 0.0


def status_pill(text: str, state: str) -> str:
    klass = {
        "running": "status-running",
        "warn": "status-warn",
        "stop": "status-stop",
        "ok": "status-running",
    }.get(state, "")
    return f'<span class="status-pill {klass}">{text}</span>'


def hero() -> None:
    phase, state, progress = pipeline_phase()
    ckpt_count = len(checkpoints())
    counts = dataset_counts()
    feature_stats = feature_cache_stats()
    gpus = gpu_table()
    dataset_value = f"{int(counts['pngs'].sum()):,}" if not counts.empty else "0"
    feature_value = fmt_bytes(feature_stats["size"])
    gpu_value = "n/a"
    if not gpus.empty:
        active = gpus.sort_values(["util_pct", "mem_used_mb"], ascending=False).iloc[0]
        gpu_value = f"{int(active['util_pct'])}% util / {int(active['mem_used_mb']):,} MB"
    if FEATURE_CACHE.exists() and notebook_process_text() and ckpt_count == 0:
        progress_text = "Milestone progress: first model is training; bar advances after the first checkpoint is saved"
    elif ckpt_count:
        progress_text = f"Milestone progress: {int(progress * 100)}% ({ckpt_count}/{EXPECTED_MODELS} checkpoints)"
    else:
        progress_text = f"Milestone progress: {int(progress * 100)}%"
    hero_stats = [
        ("Dataset PNGs", dataset_value),
        ("Feature Store", feature_value),
        ("Checkpoints", f"{ckpt_count}/{EXPECTED_MODELS}"),
        ("Active GPU", gpu_value),
    ]
    hero_stat_html = "".join(
        f"""
        <div class="hero-stat">
            <div class="hero-stat-label">{html.escape(label)}</div>
            <div class="hero-stat-value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in hero_stats
    )
    st.markdown(
        f"""
        <div class="hero">
            {status_pill(phase, state)}
            <div class="hero-title">BRACS TransMIL + DTFD-MIL Control Room</div>
            <div class="hero-subtitle">
                Live companion dashboard for dataset integrity, UNI feature cache, ensemble training,
                held-out evaluation, artifacts, and case review.
            </div>
            <div class="hero-grid">{hero_stat_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(progress, text=progress_text)


def activity_row() -> None:
    processes = notebook_processes()
    gpus = gpu_table()
    runtime = processes.iloc[0]["runtime"] if not processes.empty and "runtime" in processes else "not running"
    if gpus.empty:
        gpu_util = "n/a"
        gpu_memory = "n/a"
    else:
        active = gpus.sort_values(["util_pct", "mem_used_mb"], ascending=False).iloc[0]
        gpu_util = f"{int(active['util_pct'])}%"
        gpu_memory = f"{int(active['mem_used_mb']):,}/{int(active['mem_total_mb']):,} MB"
    last_artifact = checkpoints()[-1] if checkpoints() else FEATURE_CACHE
    metric_grid(
        [
            ("Notebook Runtime", runtime),
            ("Peak GPU Util", gpu_util),
            ("GPU Memory", gpu_memory),
            ("Last Artifact Update", modified_label(last_artifact)),
        ],
        columns_per_row=2,
    )


def metric_row() -> None:
    counts = dataset_counts()
    feature_stats = feature_cache_stats()
    ckpts = checkpoints()
    test_df = metadata_frame("test_metadata.csv")
    metric_grid(
        [
            ("Dataset PNGs", f"{int(counts['pngs'].sum()):,}" if not counts.empty else "0"),
            ("Feature Store", fmt_bytes(feature_stats["size"])),
            ("Processed Slides", "pending" if feature_stats["slides"] is None else f"{feature_stats['slides']:,}"),
            ("Checkpoints", f"{len(ckpts)}/{EXPECTED_MODELS}"),
            ("Held-out ROIs", "pending" if test_df is None else f"{len(test_df):,}"),
        ],
        columns_per_row=3,
    )


def final_project_panel() -> None:
    previous_raw = safe_csv(OUTPUT_DIR / "heterogeneous_ensemble_predictions.csv")
    previous_final = safe_csv(OUTPUT_DIR / "posthoc_metric_optimized" / "metric_optimized_predictions.csv")
    validation_path = preferred_validation_prediction_path()
    validation_final = safe_csv(validation_path) if validation_path else None
    validation_label = "pending"
    if validation_path:
        for parent in validation_path.parents:
            if parent.name.startswith("transmil_dtfd_output_validation_"):
                validation_label = parent.name.replace("transmil_dtfd_output_validation_", "").split("_", 1)[0]
                break

    if previous_raw is None or previous_final is None:
        return

    def compact_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
        metrics = class_metric_frame(df)
        accuracy = float((df["true_label"].astype(int) == df["pred_label"].astype(int)).mean())
        macro_f1 = float(metrics["f1"].mean())
        weighted_f1 = float((metrics["f1"] * metrics["support"]).sum() / max(metrics["support"].sum(), 1))
        return accuracy, macro_f1, weighted_f1

    raw_acc, raw_macro, _ = compact_metrics(previous_raw)
    final_acc, final_macro, final_weighted = compact_metrics(previous_final)
    val_acc = val_macro = None
    if validation_final is not None:
        val_acc, val_macro, _ = compact_metrics(validation_final)

    st.markdown("#### Final Project Result")
    metric_grid(
        [
            ("Final Accuracy", f"{final_acc:.2%}"),
            ("Final Macro-F1", f"{final_macro:.2%}"),
            ("Weighted-F1", f"{final_weighted:.2%}"),
            ("Accuracy Gain", f"+{(final_acc - raw_acc) * 100:.2f} pts"),
            ("Macro-F1 Gain", f"+{(final_macro - raw_macro) * 100:.2f} pts"),
            ("Validation Check", validation_label if val_acc is not None else "pending"),
        ],
        columns_per_row=3,
    )
    if val_acc is not None and val_macro is not None:
        st.caption(
            f"Validation operating point ({validation_label}): {val_acc:.2%} accuracy and {val_macro:.2%} macro-F1. "
            "The Results Suite and Error Analysis pages open with the metric-optimized prediction file first."
        )
    st.info(
        "Presentation mode is ready: use Results Suite for scorecards, Error Analysis for operating-point comparison, "
        "and Case Workbench for individual slide review."
    )


def render_command_center() -> None:
    hero()
    module_guide("Command Center")
    final_project_panel()
    metric_row()
    activity_row()

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Pipeline Timeline")
        steps = [
            ("Dataset mounted", DATA_DIR.exists()),
            ("UNI weights available", WEIGHTS.exists()),
            ("Metadata/splits built", (WORK_DIR / "metadata.csv").exists()),
            ("UNI feature cache finalized", FEATURE_CACHE.exists()),
            ("MIL checkpoints produced", len(checkpoints()) > 0),
            ("Evaluation exports written", (OUTPUT_DIR / "results_summary.txt").exists()),
        ]
        step_df = pd.DataFrame(
            {"stage": [s for s, _ in steps], "status": ["complete" if ok else "waiting" for _, ok in steps]}
        )
        st.dataframe(step_df, width="stretch", hide_index=True)

        st.markdown("#### Active Notebook Process")
        proc_df = notebook_processes()
        if proc_df.empty:
            st.code("No notebook execution process detected.", language="text")
        else:
            st.dataframe(proc_df, width="stretch", hide_index=True)

    with right:
        st.markdown("#### GPU Load")
        gpus = gpu_table()
        if gpus.empty:
            st.info("GPU telemetry is not available.")
        else:
            gpu_display = gpus.copy()
            gpu_display["memory_pct"] = (
                gpu_display["mem_used_mb"] / gpu_display["mem_total_mb"].replace(0, pd.NA) * 100
            ).fillna(0)
            gpu_display["memory"] = gpu_display.apply(
                lambda row: f"{int(row['mem_used_mb']):,}/{int(row['mem_total_mb']):,} MB", axis=1
            )
            st.dataframe(
                gpu_display[["gpu", "name", "temp_c", "util_pct", "memory", "power_w"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "util_pct": st.column_config.ProgressColumn("util_pct", min_value=0, max_value=100, format="%d%%"),
                    "power_w": st.column_config.NumberColumn("power_w", format="%.0f W"),
                },
            )
            chart_rows = []
            for row in gpu_display.itertuples(index=False):
                chart_rows.extend(
                    [
                        {"gpu": f"GPU {row.gpu}", "metric": "Utilization", "percent": row.util_pct},
                        {"gpu": f"GPU {row.gpu}", "metric": "Memory", "percent": row.memory_pct},
                    ]
                )
            chart_df = pd.DataFrame(chart_rows)
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    x=alt.X("percent:Q", title="Percent", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("gpu:N", title=None),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["gpu", "metric", alt.Tooltip("percent:Q", format=".1f")],
                )
                .properties(height=max(120, 58 * len(gpu_display)))
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Recent Artifacts")
    art = artifact_table(artifact_files(), limit=14)
    if art.empty:
        st.info("No artifacts yet.")
    else:
        st.dataframe(art, width="stretch", hide_index=True)

    st.download_button(
        "Download run status JSON",
        data=json.dumps(status_payload(), indent=2, default=str),
        file_name="bracs_run_status.json",
        mime="application/json",
    )


def render_dataset_lab() -> None:
    page_heading("Dataset Lab")
    module_guide("Dataset Lab")
    counts = dataset_counts()
    if counts.empty:
        st.info("The extracted BRACS ROI dataset was not found.")
        return

    totals = counts.groupby("split", as_index=False)["pngs"].sum()
    class_totals = counts.groupby("class", as_index=False)["pngs"].sum().sort_values("pngs", ascending=False)

    cols = st.columns(3)
    for idx, row in totals.iterrows():
        cols[idx % 3].metric(f"{row['split'].title()} ROIs", f"{int(row['pngs']):,}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Split x Class Matrix")
        matrix = counts.pivot_table(index="class", columns="split", values="pngs", fill_value=0, aggfunc="sum")
        st.dataframe(matrix, width="stretch")
    with right:
        st.markdown("#### Class Distribution")
        chart = (
            alt.Chart(class_totals)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("pngs:Q", title="ROIs"),
                y=alt.Y("class:N", sort="-x", title=None),
                color=alt.Color("class:N", legend=None),
                tooltip=["class", "pngs"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    metadata = metadata_frame("metadata.csv")
    test_metadata = metadata_frame("test_metadata.csv")
    tab_meta, tab_test, tab_gallery = st.tabs(["CV Metadata", "Held-Out Test", "ROI Gallery"])
    with tab_meta:
        if metadata is None:
            st.info("Metadata will appear once the notebook builds splits.")
        else:
            cols = st.columns(4)
            cols[0].metric("CV ROIs", f"{len(metadata):,}")
            cols[1].metric("CV Patients", f"{metadata['patient_id'].nunique():,}" if "patient_id" in metadata else "n/a")
            cols[2].metric("CV Slides", f"{metadata['slide_id'].nunique():,}" if "slide_id" in metadata else "n/a")
            cols[3].metric("Classes", f"{metadata['label'].nunique():,}" if "label" in metadata else "n/a")
            st.dataframe(metadata, width="stretch", hide_index=True)
    with tab_test:
        if test_metadata is None:
            st.info("Held-out metadata will appear once the notebook builds splits.")
        else:
            test_pred_path = heldout_test_prediction_path()
            roi_predictions = heldout_roi_prediction_frame()
            cols = st.columns(5)
            cols[0].metric("Test ROIs", f"{len(test_metadata):,}")
            cols[1].metric("Test Patients", f"{test_metadata['patient_id'].nunique():,}" if "patient_id" in test_metadata else "n/a")
            cols[2].metric("Test Slides", f"{test_metadata['slide_id'].nunique():,}" if "slide_id" in test_metadata else "n/a")
            cols[3].metric("Classes", f"{test_metadata['label'].nunique():,}" if "label" in test_metadata else "n/a")
            cols[4].metric(
                "Predicted Slides",
                "pending" if roi_predictions is None or "pred_label" not in roi_predictions else f"{roi_predictions['pred_label'].notna().groupby(roi_predictions['slide_id']).max().sum():,}",
            )
            if test_pred_path is not None:
                st.caption(f"Held-out prediction file: {display_path(test_pred_path)}")
            if roi_predictions is not None:
                st.dataframe(roi_predictions, width="stretch", hide_index=True)
            else:
                st.dataframe(test_metadata, width="stretch", hide_index=True)
    with tab_gallery:
        split = st.selectbox("Split", ["train", "val", "test"], key="gallery_split")
        class_dirs = sorted((DATA_DIR / split).glob("*")) if (DATA_DIR / split).exists() else []
        if not class_dirs:
            st.info("No class folders found for this split.")
            return
        folder = st.selectbox("Class folder", class_dirs, format_func=lambda p: CLASS_NAMES.get(p.name, p.name))
        limit = st.slider("Images", 4, 24, 12, step=4)
        paths = sorted(folder.glob("*.png"))[:limit] if folder else []
        if paths:
            st.image([str(p) for p in paths], caption=[p.name for p in paths], width=170)
            shape_df = image_shape_sample(split, folder.name)
            if not shape_df.empty:
                st.markdown("#### Image Shape Sample")
                cols = st.columns(3)
                cols[0].metric("Sampled ROIs", f"{len(shape_df):,}")
                cols[1].metric("Median Width", f"{shape_df['width'].median():.0f}px")
                cols[2].metric("Median Height", f"{shape_df['height'].median():.0f}px")
                st.dataframe(shape_df, width="stretch", hide_index=True)
        else:
            st.info("No images found for this selection.")


def render_training_monitor() -> None:
    page_heading("Training Monitor")
    module_guide("Training Monitor")
    stats = feature_cache_stats()
    metric_grid(
        [
            ("Cache State", "final" if stats["is_final"] else "building"),
            ("Cache Size", fmt_bytes(stats["size"])),
            ("Slides", "pending" if stats["slides"] is None else f"{stats['slides']:,}"),
            ("Patch Rows", "pending" if not stats["patches"] else f"{stats['patches']:,}"),
            ("Feature Dim", "pending" if stats["feature_dim"] is None else str(stats["feature_dim"])),
        ],
        columns_per_row=3,
    )

    if stats["error"]:
        st.warning(f"HDF5 read note: {stats['error']}")

    st.markdown("#### Validation Runs")
    validations = validation_run_table()
    if validations.empty:
        st.info("No isolated validation runs found yet. Run the 3x5 command below after the feature store is ready.")
    else:
        st.dataframe(
            validations,
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="%.3f"),
                "macro_f1": st.column_config.ProgressColumn("macro_f1", min_value=0, max_value=1, format="%.3f"),
                "raw_accuracy": st.column_config.ProgressColumn("raw_accuracy", min_value=0, max_value=1, format="%.3f"),
                "raw_macro_f1": st.column_config.ProgressColumn("raw_macro_f1", min_value=0, max_value=1, format="%.3f"),
            },
        )
    st.code(
        "python run_validation_3x5.py\n"
        "python posthoc_calibrate_predictions.py "
        "--input work/transmil_dtfd_output_validation_3x5_<timestamp>/validation_3x5_test_predictions.csv "
        "--calibration-input work/transmil_dtfd_output_validation_3x5_<timestamp>/validation_3x5_oof_predictions.csv "
        "--eval-input work/transmil_dtfd_output_validation_3x5_<timestamp>/validation_3x5_test_predictions.csv "
        "--output-dir work/transmil_dtfd_output_validation_3x5_<timestamp>/posthoc_metric_optimized",
        language="bash",
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Feature Scales")
        if stats["scales"]:
            scale_df = pd.DataFrame({"scale": list(stats["scales"].keys()), "patch_rows": list(stats["scales"].values())})
            st.dataframe(scale_df, width="stretch", hide_index=True)
            st.bar_chart(scale_df.set_index("scale"))
        else:
            st.info("Scale statistics are available after the final HDF5 cache is readable.")
    with right:
        st.markdown("#### Checkpoint Timeline")
        ckpts = checkpoints()
        if ckpts:
            cdf = artifact_table(ckpts, limit=EXPECTED_MODELS)
            st.dataframe(cdf, width="stretch", hide_index=True)
        else:
            st.info("No checkpoints yet. Training starts after feature extraction.")

    st.markdown("#### Artifact Browser")
    all_artifacts = artifact_files()
    kinds = sorted({p.suffix.lstrip(".") or "file" for p in all_artifacts if p.exists()})
    selected_kinds = st.multiselect("File types", kinds, default=kinds)
    filtered = [p for p in all_artifacts if (p.suffix.lstrip(".") or "file") in selected_kinds]
    st.dataframe(artifact_table(filtered, limit=80), width="stretch", hide_index=True)
    small_artifacts = [p for p in filtered if p.exists() and p.is_file() and p.stat().st_size <= 200 * 1024 * 1024]
    if small_artifacts:
        selected_artifact = st.selectbox("Download artifact", small_artifacts, format_func=display_path)
        st.download_button(
            "Download selected artifact",
            data=selected_artifact.read_bytes(),
            file_name=selected_artifact.name,
            mime="application/octet-stream",
        )
    elif filtered:
        st.caption("Artifacts over 200 MB are listed but not loaded into the browser for download.")


def label_name(value) -> str:
    return THREE_CLASS.get(value, THREE_CLASS.get(str(value), str(value)))


def prediction_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in PROB_COLS if c in df.columns]


def prediction_schema(df: pd.DataFrame) -> bool:
    needed = {"slide_id", "true_label", "pred_label", *PROB_COLS}
    return needed.issubset(set(df.columns))


def predictions_path() -> Path | None:
    files = prediction_files()
    return files[0] if files else None


def prediction_files() -> list[Path]:
    base_roots = sorted(
        [p for p in WORK_DIR.glob("transmil_dtfd_output*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if OUTPUT_DIR.exists() and OUTPUT_DIR not in base_roots:
        base_roots.append(OUTPUT_DIR)
    roots = []
    for root in base_roots:
        roots.extend(
            sorted(
                [p for p in root.glob("posthoc*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
        roots.append(root)
    preferred_names = [
        "metric_optimized_predictions.csv",
        "calibrated_predictions.csv",
        "validation_3x5_test_predictions.csv",
        "validation_3x5_oof_predictions.csv",
        "validation_2x5_test_predictions.csv",
        "validation_2x5_oof_predictions.csv",
        "validation_1x5_test_predictions.csv",
        "validation_1x5_oof_predictions.csv",
        "heterogeneous_ensemble_predictions.csv",
        "test_predictions.csv",
    ]
    preferred = [root / name for root in roots for name in preferred_names]
    seen = set()
    files = []
    discovered = []
    for root in roots:
        discovered.extend(root.glob("*predictions*.csv"))
    for path in preferred + sorted(discovered, key=lambda p: p.stat().st_mtime, reverse=True):
        if path.exists() and path not in seen:
            files.append(path)
            seen.add(path)
    return files


def heldout_test_prediction_path() -> Path | None:
    current_dir = current_validation_output_dir()
    if current_dir is None:
        return None
    candidates = [
        current_dir / "validation_3x5_test_predictions.csv",
        current_dir / "test_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def current_cv_prediction_path() -> Path | None:
    current_dir = current_validation_output_dir()
    if current_dir is None:
        return None
    candidates = [
        current_dir / "posthoc_metric_optimized" / "metric_optimized_predictions.csv",
        current_dir / "validation_3x5_oof_predictions.csv",
        current_dir / "validation_1x5_oof_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def original_class_to_label(class_folder: str) -> int | None:
    suffix = class_folder.split("_", 1)[-1]
    if suffix in {"N", "PB", "UDH"}:
        return 0
    if suffix in {"FEA", "ADH"}:
        return 1
    if suffix in {"DCIS", "IC"}:
        return 2
    return None


@st.cache_data(ttl=60)
def official_dataset_matches(filename: str) -> pd.DataFrame:
    rows = []
    for path in DATA_DIR.rglob(filename):
        if not path.is_file():
            continue
        split = path.parts[-3] if len(path.parts) >= 3 else ""
        class_folder = path.parts[-2] if len(path.parts) >= 2 else ""
        label = original_class_to_label(class_folder)
        rows.append(
            {
                "split": split,
                "class_folder": class_folder,
                "original_class": CLASS_NAMES.get(class_folder, class_folder),
                "known_class": label_name(label) if label is not None else "unknown",
                "path": display_path(path),
            }
        )
    return pd.DataFrame(rows)


def metadata_matches(filename: str) -> pd.DataFrame:
    frames = []
    for source_name, caption in [
        ("metadata.csv", "exact_3636_cv"),
        ("test_metadata.csv", "held_out_test"),
    ]:
        df = metadata_frame(source_name)
        if df is None or "path" not in df:
            continue
        matched = df[df["path"].astype(str).map(lambda p: Path(p).name) == filename].copy()
        if matched.empty:
            continue
        matched["source"] = caption
        frames.append(matched)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "label" in out:
        out["known_class"] = out["label"].map(label_name)
    if "path" in out:
        out["path"] = out["path"].map(lambda p: display_path(Path(str(p))))
    return out


def prediction_rows_for_slides(slide_ids: list[str]) -> tuple[Path | None, pd.DataFrame]:
    pred_path = current_cv_prediction_path()
    if pred_path is None:
        return None, pd.DataFrame()
    pred = safe_csv(pred_path)
    if pred is None or "slide_id" not in pred:
        return pred_path, pd.DataFrame()
    rows = pred[pred["slide_id"].astype(str).isin([str(s) for s in slide_ids])].copy()
    if rows.empty:
        return pred_path, rows
    rows["true_class"] = rows["true_label"].map(label_name) if "true_label" in rows else ""
    rows["saved_cv_prediction"] = rows["pred_label"].map(label_name) if "pred_label" in rows else ""
    return pred_path, rows


def heldout_roi_prediction_frame() -> pd.DataFrame | None:
    test_metadata = metadata_frame("test_metadata.csv")
    pred_path = heldout_test_prediction_path()
    if test_metadata is None or pred_path is None or not pred_path.exists():
        return None
    pred = safe_csv(pred_path)
    if pred is None or "slide_id" not in pred:
        return None
    roi = test_metadata.copy()
    roi["slide_id"] = roi["slide_id"].astype(str)
    pred = pred.copy()
    pred["slide_id"] = pred["slide_id"].astype(str)
    keep_cols = [
        "slide_id",
        "true_label",
        "pred_label",
        "prob_0",
        "prob_1",
        "prob_2",
        "confidence",
        "margin",
        "review_recommended",
    ]
    keep_cols = [c for c in keep_cols if c in pred.columns]
    joined = roi.merge(pred[keep_cols], on="slide_id", how="left", suffixes=("_roi", "_prediction"))
    if "label" in joined:
        joined = joined.rename(columns={"label": "roi_label"})
    return joined


def class_metric_frame(df: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        df["true_label"], df[pred_col], labels=labels, zero_division=0
    )
    rows = []
    for idx, label in enumerate(labels):
        true_rows = df[df["true_label"] == label]
        correct = int((true_rows[pred_col] == label).sum())
        rows.append(
            {
                "class": label_name(label),
                "support": int(support[idx]),
                "correct": correct,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "missed": int(support[idx] - correct),
            }
        )
    return pd.DataFrame(rows)


def confusion_pair_frame(df: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    errors = df[df["true_label"] != df[pred_col]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["true_class", "predicted_as", "count", "share_of_true_class"])
    class_support = df["true_label"].value_counts().to_dict()
    pairs = (
        errors.groupby(["true_label", pred_col], as_index=False)
        .size()
        .rename(columns={"size": "count", pred_col: "predicted_label"})
        .sort_values("count", ascending=False)
    )
    pairs["true_class"] = pairs["true_label"].map(label_name)
    pairs["predicted_as"] = pairs["predicted_label"].map(label_name)
    pairs["share_of_true_class"] = pairs.apply(
        lambda row: row["count"] / max(class_support.get(row["true_label"], 1), 1), axis=1
    )
    return pairs[["true_class", "predicted_as", "count", "share_of_true_class"]]


def add_confidence_columns(df: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    out = calibration_confidence_columns(df, pred_col=pred_col, threshold=DEFAULT_ATYPIA_THRESHOLD)
    out["true_class"] = out["true_label"].map(label_name)
    out["predicted_as"] = out[pred_col].map(label_name)
    out["correct"] = out["true_label"] == out[pred_col]
    return out


def apply_atypia_threshold(df: pd.DataFrame, threshold: float) -> pd.Series:
    return pd.Series(atypia_threshold_predictions(df[PROB_COLS].to_numpy(), threshold), index=df.index)


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [x / 100 for x in range(20, 81, 5)]:
        pred_col = f"pred_atypia_{threshold:.2f}"
        work = df.copy()
        work[pred_col] = apply_atypia_threshold(work, threshold)
        metrics = class_metric_frame(work, pred_col=pred_col)
        atypia = metrics[metrics["class"] == "Atypia"].iloc[0]
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float((work["true_label"] == work[pred_col]).mean()),
                "macro_f1": float(metrics["f1"].mean()),
                "atypia_precision": float(atypia["precision"]),
                "atypia_recall": float(atypia["recall"]),
                "atypia_f1": float(atypia["f1"]),
            }
        )
    return pd.DataFrame(rows)


def render_results_suite() -> None:
    page_heading("Results Suite")
    module_guide("Results Suite")
    validations = validation_run_table()
    if not validations.empty:
        st.markdown("#### Validation Summary")
        st.dataframe(
            validations[["run", "models", "accuracy", "macro_f1", "raw_accuracy", "raw_macro_f1", "path"]],
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="%.3f"),
                "macro_f1": st.column_config.ProgressColumn("macro_f1", min_value=0, max_value=1, format="%.3f"),
                "raw_accuracy": st.column_config.ProgressColumn("raw_accuracy", min_value=0, max_value=1, format="%.3f"),
                "raw_macro_f1": st.column_config.ProgressColumn("raw_macro_f1", min_value=0, max_value=1, format="%.3f"),
            },
        )
    plot_paths = sorted(OUTPUT_DIR.glob("*.png"))
    if plot_paths:
        selected_plots = st.multiselect("Plots", plot_paths, default=plot_paths[:4], format_func=lambda p: p.name)
        if selected_plots:
            st.image([str(p) for p in selected_plots], caption=[p.name for p in selected_plots], width="stretch")
    else:
        st.info("Plots will appear when the notebook reaches evaluation/export cells.")

    csv_paths = sorted(OUTPUT_DIR.glob("*.csv"))
    if csv_paths:
        selected = st.selectbox("CSV output", csv_paths, format_func=lambda p: p.name)
        df = safe_csv(selected)
        if df is not None:
            st.dataframe(df, width="stretch")
            prob_cols = prediction_columns(df)
            label_cols = [c for c in df.columns if c in {"true_label", "label", "labels"}]
            pred_cols = [c for c in df.columns if c in {"pred_label", "predicted_label", "preds"}]
            if label_cols and pred_cols:
                true_col, pred_col = label_cols[0], pred_cols[0]
                acc = (df[true_col] == df[pred_col]).mean()
                st.metric("Accuracy in selected CSV", f"{acc:.3f}")
                cm = pd.crosstab(df[true_col].map(label_name), df[pred_col].map(label_name), dropna=False)
                st.dataframe(cm, width="stretch")
            if prob_cols:
                st.markdown("#### Probability Summary")
                st.dataframe(df[prob_cols].describe().T, width="stretch")
    else:
        st.info("CSV exports are not available yet.")

    summary = OUTPUT_DIR / "results_summary.txt"
    if summary.exists():
        st.markdown("#### Final Summary")
        st.text_area("results_summary.txt", summary.read_text(errors="replace"), height=320)


def render_error_analysis() -> None:
    page_heading("Error Analysis")
    module_guide("Error Analysis")
    paths = prediction_files()
    if not paths:
        st.info("Prediction CSVs are not available yet.")
        return
    path = st.selectbox("Prediction set", paths, format_func=lambda p: p.name)

    comparison = safe_csv(path.parent / "posthoc_metrics_comparison.csv")
    if comparison is None:
        comparison = safe_csv(OUTPUT_DIR / "posthoc_metrics_comparison.csv")
    if comparison is not None:
        st.markdown("#### Operating Point Comparison")
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="%.3f"),
                "macro_f1": st.column_config.ProgressColumn("macro_f1", min_value=0, max_value=1, format="%.3f"),
                "weighted_f1": st.column_config.ProgressColumn("weighted_f1", min_value=0, max_value=1, format="%.3f"),
            },
        )
        chart_df = comparison.melt(
            id_vars=["model"],
            value_vars=["accuracy", "macro_f1", "weighted_f1"],
            var_name="metric",
            value_name="score",
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("model:N", title=None),
                yOffset=alt.YOffset("metric:N"),
                color=alt.Color("metric:N", title=None),
                tooltip=["model", "metric", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

    df = safe_csv(path)
    if df is None or not prediction_schema(df):
        st.warning("The available CSV does not contain slide_id, true_label, pred_label, and prob_0/prob_1/prob_2.")
        return

    calibrated_df = add_operating_point_columns(df, threshold=DEFAULT_ATYPIA_THRESHOLD)
    raw_df = calibrated_df.copy()
    raw_df["pred_label"] = raw_df["raw_pred_label"].astype(int)
    raw_df["calibration_method"] = "raw_argmax"
    raw_df["changed_prediction"] = False

    frozen_label = f"Frozen Atypia threshold {DEFAULT_ATYPIA_THRESHOLD:.3f}"
    candidate_frames = {
        "Raw argmax": raw_df,
        frozen_label: calibrated_df,
    }
    optimized_col = None
    for col in ("metric_optimized_pred_label", "bias_pred_label", "recommended_pred_label"):
        if col in calibrated_df.columns:
            optimized_col = col
            break
    if optimized_col:
        optimized_df = calibrated_df.copy()
        optimized_df["pred_label"] = optimized_df[optimized_col].astype(int)
        optimized_df["calibrated_pred_label"] = optimized_df["pred_label"]
        optimized_df["calibration_method"] = optimized_df.get("metric_optimized_method", "constrained_log_bias")
        optimized_df["changed_prediction"] = optimized_df["raw_pred_label"].astype(int) != optimized_df["pred_label"].astype(int)
        optimized_df = add_confidence_columns(optimized_df)
        candidate_frames["Metric optimized log-bias"] = optimized_df

    rows = []
    candidate_scores = {}
    for label, frame in candidate_frames.items():
        candidate_metrics = class_metric_frame(frame)
        candidate_macro_f1 = float(candidate_metrics["f1"].mean())
        candidate_scores[label] = candidate_macro_f1
        atypia_row = candidate_metrics[candidate_metrics["class"] == "Atypia"].iloc[0]
        rows.append(
            {
                "operating_point": label,
                "accuracy": float((frame["true_label"] == frame["pred_label"]).mean()),
                "macro_f1": candidate_macro_f1,
                "atypia_precision": float(atypia_row["precision"]),
                "atypia_recall": float(atypia_row["recall"]),
                "changed": int(frame["changed_prediction"].sum()) if "changed_prediction" in frame else 0,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    best_operating_point = comparison.iloc[0]["operating_point"]
    st.markdown("#### Hard-Prediction Operating Points")
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="%.3f"),
            "macro_f1": st.column_config.ProgressColumn("macro_f1", min_value=0, max_value=1, format="%.3f"),
            "atypia_precision": st.column_config.ProgressColumn("atypia_precision", min_value=0, max_value=1, format="%.3f"),
            "atypia_recall": st.column_config.ProgressColumn("atypia_recall", min_value=0, max_value=1, format="%.3f"),
        },
    )
    operating_point = st.radio(
        "Operating point",
        list(candidate_frames.keys()),
        horizontal=True,
        index=list(candidate_frames.keys()).index(best_operating_point),
    )
    df = candidate_frames[operating_point]
    if best_operating_point != frozen_label:
        st.warning(
            f"Frozen threshold {DEFAULT_ATYPIA_THRESHOLD:.3f} is available for audit, "
            f"but {best_operating_point} is stronger on this file "
            f"({candidate_scores[best_operating_point]:.3f} macro-F1)."
        )
    if operating_point == frozen_label:
        st.success(
            f"Default inference uses Atypia threshold {DEFAULT_ATYPIA_THRESHOLD:.3f}. "
            f"Changed predictions: {int(df['changed_prediction'].sum())}"
        )

    metrics = class_metric_frame(df)
    pairs = confusion_pair_frame(df)
    enriched = add_confidence_columns(df)
    accuracy = float((df["true_label"] == df["pred_label"]).mean())
    macro_f1 = float(metrics["f1"].mean())
    atypia = metrics[metrics["class"] == "Atypia"].iloc[0]
    benign_to_atypia = int(((df["true_label"] == 0) & (df["pred_label"] == 1)).sum())
    malignant_to_atypia = int(((df["true_label"] == 2) & (df["pred_label"] == 1)).sum())
    review_count = int(enriched["review_recommended"].sum())

    metric_grid(
        [
            ("Accuracy", f"{accuracy:.3f}"),
            ("Macro F1", f"{macro_f1:.3f}"),
            ("Atypia F1", f"{atypia['f1']:.3f}"),
            ("Benign to Atypia", f"{benign_to_atypia}"),
            ("Malignant to Atypia", f"{malignant_to_atypia}"),
            ("Review Queue", f"{review_count}"),
        ],
        columns_per_row=3,
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Per-Class Scorecard")
        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True,
            column_config={
                "precision": st.column_config.ProgressColumn("precision", min_value=0, max_value=1, format="%.3f"),
                "recall": st.column_config.ProgressColumn("recall", min_value=0, max_value=1, format="%.3f"),
                "f1": st.column_config.ProgressColumn("f1", min_value=0, max_value=1, format="%.3f"),
            },
        )
    with right:
        st.markdown("#### Class Metric Balance")
        metric_chart = metrics.melt(
            id_vars=["class"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )
        chart = (
            alt.Chart(metric_chart)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("class:N", title=None),
                yOffset=alt.YOffset("metric:N"),
                color=alt.Color("metric:N", title=None),
                tooltip=["class", "metric", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Confusion Pressure")
    if pairs.empty:
        st.success("No misclassifications found in the selected predictions.")
    else:
        pair_chart = (
            alt.Chart(pairs)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("count:Q", title="Misclassified slides"),
                y=alt.Y("true_class:N", title="True class", sort="-x"),
                color=alt.Color("predicted_as:N", title="Predicted as"),
                tooltip=[
                    "true_class",
                    "predicted_as",
                    "count",
                    alt.Tooltip("share_of_true_class:Q", format=".1%"),
                ],
            )
            .properties(height=250)
        )
        st.altair_chart(pair_chart, use_container_width=True)
        st.dataframe(
            pairs,
            width="stretch",
            hide_index=True,
            column_config={
                "share_of_true_class": st.column_config.ProgressColumn(
                    "share_of_true_class", min_value=0, max_value=1, format="%.1f%%"
                )
            },
        )

    st.markdown("#### Atypia Threshold Lab")
    threshold = st.slider("Atypia threshold", 0.20, 0.80, DEFAULT_ATYPIA_THRESHOLD, step=0.01)
    simulated = raw_df.copy()
    simulated["threshold_pred"] = apply_atypia_threshold(simulated, threshold)
    sim_metrics = class_metric_frame(simulated, pred_col="threshold_pred")
    sim_atypia = sim_metrics[sim_metrics["class"] == "Atypia"].iloc[0]
    metric_grid(
        [
            ("Sim Accuracy", f"{(simulated['true_label'] == simulated['threshold_pred']).mean():.3f}"),
            ("Sim Macro F1", f"{sim_metrics['f1'].mean():.3f}"),
            ("Sim Atypia Recall", f"{sim_atypia['recall']:.3f}"),
            ("Sim Atypia Precision", f"{sim_atypia['precision']:.3f}"),
        ],
        columns_per_row=2,
    )

    sweep = threshold_sweep(df)
    sweep_chart = (
        alt.Chart(sweep.melt("threshold", value_vars=["accuracy", "macro_f1", "atypia_precision", "atypia_recall", "atypia_f1"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="Atypia threshold"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("variable:N", title="Metric"),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                "variable",
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(sweep_chart, use_container_width=True)

    st.markdown("#### Review Queue")
    queue_mode = st.radio(
        "Queue",
        ["High uncertainty", "All errors", "Atypia boundary", "Malignant misses", "Low confidence"],
        horizontal=True,
    )
    if queue_mode == "High uncertainty":
        queue = high_uncertainty_queue(enriched, pred_col="pred_label", threshold=DEFAULT_ATYPIA_THRESHOLD, limit=120)
        queue["true_class"] = queue["true_label"].map(label_name)
        queue["predicted_as"] = queue["pred_label"].map(label_name)
    elif queue_mode == "Atypia boundary":
        queue = enriched[
            ((enriched["true_label"] == 1) & (enriched["pred_label"] != 1))
            | ((enriched["true_label"] != 1) & (enriched["pred_label"] == 1))
        ]
    elif queue_mode == "Malignant misses":
        queue = enriched[(enriched["true_label"] == 2) & (enriched["pred_label"] != 2)]
    elif queue_mode == "Low confidence":
        queue = enriched.sort_values(["confidence", "margin"], ascending=True).head(80)
    else:
        queue = enriched[~enriched["correct"]]
    queue = queue.sort_values(["margin", "atypia_boundary_distance", "confidence"], ascending=True).head(120)
    display_cols = [
        "slide_id",
        "true_class",
        "predicted_as",
        "review_recommended",
        "confidence",
        "margin",
        "atypia_boundary_distance",
        "prob_0",
        "prob_1",
        "prob_2",
    ]
    st.dataframe(
        queue[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.3f"),
            "margin": st.column_config.ProgressColumn("margin", min_value=0, max_value=1, format="%.3f"),
            "atypia_boundary_distance": st.column_config.ProgressColumn(
                "atypia_boundary_distance", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )
    st.download_button(
        "Download review queue CSV",
        data=queue[display_cols].to_csv(index=False),
        file_name=f"{queue_mode.lower().replace(' ', '_')}_review_queue.csv",
        mime="text/csv",
    )


def render_case_workbench() -> None:
    page_heading("Live Demo")
    known_tab, upload_tab = st.tabs(["Known 3636 ROI", "Manual Upload"])
    with known_tab:
        render_known_roi_review()
    with upload_tab:
        render_manual_upload_demo()


def render_manual_upload_demo() -> None:
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("Upload ROI", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded is not None:
            img = Image.open(uploaded)
            upload_key = f"{uploaded.name}:{img.width}x{img.height}"
            if st.session_state.get("live_upload_key") != upload_key:
                st.session_state["live_upload_key"] = upload_key
                st.session_state.pop("live_demo_prediction", None)
            st.image(img, caption=uploaded.name, width="stretch")
            with st.expander("Image details", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "width": img.width,
                                "height": img.height,
                                "mode": img.mode,
                                "format": img.format or Path(uploaded.name).suffix.lstrip(".").upper(),
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            official_match = official_dataset_matches(uploaded.name)
            metadata_match = metadata_matches(uploaded.name)

            st.info("Demo is blind by default: known labels and saved labels are not shown on this page.")
            with st.expander("Safe match status", expanded=False):
                st.dataframe(match_summary_frame(official_match, metadata_match), width="stretch", hide_index=True)

            test_meta = metadata_frame("test_metadata.csv")
            heldout_pool_match = False
            if test_meta is not None and "path" in test_meta:
                matched = test_meta[test_meta["path"].astype(str).map(lambda p: Path(p).name) == uploaded.name]
                if not matched.empty:
                    heldout_pool_match = True
                    st.success("Uploaded ROI matched to the held-out demo pool.")
                    with st.expander("Held-out metadata match", expanded=False):
                        st.dataframe(matched[public_columns(matched)], width="stretch", hide_index=True)
            if (
                not official_match.empty
                and "test" in set(official_match["split"].dropna().astype(str))
                and not heldout_pool_match
            ):
                st.info(
                    "This file is in the official BRACS test split, but it is not in the separate 900-ROI "
                    "held-out demo metadata. In this project mapping it belongs to the exact 3636 CV set."
                )
            live_ckpts = available_live_checkpoints()
            st.markdown("#### Live Model Prediction")
            if not live_ckpts:
                st.warning("Live prediction will unlock after training saves at least one MIL checkpoint.")
            else:
                live_cols = st.columns([1, 1, 1])
                ensemble_options = [1, 5, 15]
                default_ensemble_index = 2 if len(live_ckpts) >= 15 else min(len(ensemble_options) - 1, 1)
                model_count = live_cols[0].selectbox(
                    "Checkpoint ensemble",
                    options=ensemble_options,
                    index=default_ensemble_index,
                    help="15 is the final ensemble. Use 1 only if you need the fastest possible response.",
                )
                model_count = min(int(model_count), len(live_ckpts))
                live_device = live_cols[1].selectbox("Device", options=["auto", "mps", "cpu"], index=0)
                max_live_patches = live_cols[2].selectbox("Patches/scale", options=[32, 64, 96], index=1)
                if st.button("Run live prediction", type="primary"):
                    with st.spinner("Extracting UNI features and running MIL checkpoint inference..."):
                        try:
                            result = predict_roi_image(
                                img,
                                device=live_device,
                                limit_models=model_count,
                                max_patches_per_scale=int(max_live_patches),
                            )
                        except Exception as exc:
                            st.error(f"Live prediction failed: {exc}")
                        else:
                            st.session_state["live_demo_prediction"] = result
                if "live_demo_prediction" in st.session_state:
                    render_live_prediction_result(st.session_state["live_demo_prediction"])
        else:
            st.info("Upload any BRACS ROI image. For the blind demo, choose from the Blind Pool folder.")
    with right:
        ckpts = available_live_checkpoints()
        if ckpts:
            st.success(f"{len(ckpts)}/15 checkpoints ready.")
        else:
            st.warning("Checkpoint-based prediction is not available yet.")
        metric_grid(
            [
                ("Recommended ensemble", "15"),
                ("Device", "auto / MPS"),
                ("Patch scale", "64"),
                ("Output", "class + confidence + review"),
            ],
            columns_per_row=2,
        )
        st.caption("Blind-pool labels remain hidden during prediction.")
        st.caption("The private answer key is kept separately for post-demo verification.")

def final_output_dir() -> Path | None:
    for root in validation_output_dirs():
        if (root / "posthoc_metric_optimized" / "metric_optimized_predictions.csv").exists():
            return root
    return current_validation_output_dir()


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def percent_value(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "n/a"
    return f"{float(number):.2%}"


def public_columns(df: pd.DataFrame) -> list[str]:
    hidden = {
        "class_folder",
        "original_class",
        "path",
        "label",
        "roi_label",
        "true_label",
        "true_class",
        "known_class",
        "original_label",
        "original_filename",
        "original_path",
        "saved_cv_prediction",
        "pred_label",
        "patient_id",
    }
    return [col for col in df.columns if col not in hidden]


def match_summary_frame(official_match: pd.DataFrame, metadata_match: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not official_match.empty:
        rows.append({"check": "Official BRACS filename", "status": "matched", "detail": f"{len(official_match)} file(s)"})
    else:
        rows.append({"check": "Official BRACS filename", "status": "not found", "detail": "live prediction still available"})
    if not metadata_match.empty:
        rows.append({"check": "Project metadata", "status": "matched", "detail": "project ROI record available"})
    else:
        rows.append({"check": "Project metadata", "status": "not found", "detail": "blind/random upload"})
    return pd.DataFrame(rows)


def render_live_prediction_result(result: dict, result_key: str = "live") -> None:
    blind_context = result_key == "blind_pool"
    review_label = "Model Review" if blind_context else "Review"
    if blind_context:
        st.markdown("#### Pre-Verification Prediction")
    result_cols = st.columns(5)
    result_cols[0].metric("Prediction", result["predicted_class"])
    result_cols[1].metric("Confidence", f"{result['confidence']:.3f}")
    result_cols[2].metric("Margin", f"{result['margin']:.3f}")
    result_cols[3].metric(review_label, "yes" if result["review_recommended"] else "no")
    result_cols[4].metric("Models Used", result["models_used"])
    if result["review_recommended"]:
        st.warning("Model review recommended because confidence or class margin is low.")
    else:
        st.success("No model-only review flag at the selected operating point.")
    if blind_context:
        st.caption("Before private verification, review is based only on model confidence/margin. Mismatch review is added after the true class is revealed.")
    result_table = live_prediction_frame(result).rename(
        columns={
            "predicted_class": "prediction",
            "review_recommended": "model_review",
            "models_used": "models_used",
        }
    )
    st.dataframe(result_table, width="stretch", hide_index=True)
    with st.expander("Technical details", expanded=False):
        st.write(f"Models used: {result['models_used']} on {result['device']}")
        st.write(f"Patches/scale: {result.get('max_patches_per_scale', 'n/a')}")


def render_known_roi_review() -> None:
    metadata = metadata_frame("metadata.csv")
    if metadata is None or metadata.empty:
        st.warning("Exact 3636 ROI metadata is not available.")
        return

    work = metadata.copy()
    work["roi_file"] = work["path"].astype(str).map(lambda value: Path(value).name)
    work["true_class"] = work["label"].map(label_name)
    class_options = ["All", "Benign", "Atypia", "Malignant"]
    filter_cols = st.columns([1, 2, 1])
    selected_class = filter_cols[0].selectbox("Class filter", class_options, index=0)
    search_text = filter_cols[1].text_input("Search ROI / slide", value="", placeholder="BRACS_...")
    if filter_cols[2].button("Pick random known ROI", type="primary"):
        filtered_for_random = work
        if selected_class != "All":
            filtered_for_random = filtered_for_random[filtered_for_random["true_class"] == selected_class]
        if search_text.strip():
            needle = search_text.strip().lower()
            filtered_for_random = filtered_for_random[
                filtered_for_random["roi_file"].str.lower().str.contains(needle, na=False)
                | filtered_for_random["slide_id"].astype(str).str.lower().str.contains(needle, na=False)
            ]
        if not filtered_for_random.empty:
            st.session_state["known_roi_index"] = int(filtered_for_random.sample(1).index[0])

    filtered = work
    if selected_class != "All":
        filtered = filtered[filtered["true_class"] == selected_class]
    if search_text.strip():
        needle = search_text.strip().lower()
        filtered = filtered[
            filtered["roi_file"].str.lower().str.contains(needle, na=False)
            | filtered["slide_id"].astype(str).str.lower().str.contains(needle, na=False)
        ]

    if filtered.empty:
        st.info("No ROI matched the current filter.")
        return

    if "known_roi_index" not in st.session_state or st.session_state["known_roi_index"] not in filtered.index:
        st.session_state["known_roi_index"] = int(filtered.index[0])

    selected_index = st.selectbox(
        "Known 3636 ROI",
        options=filtered.index.tolist(),
        index=filtered.index.tolist().index(st.session_state["known_roi_index"]),
        format_func=lambda idx: f"{work.loc[idx, 'roi_file']} | {work.loc[idx, 'true_class']} | {work.loc[idx, 'slide_id']}",
    )
    st.session_state["known_roi_index"] = int(selected_index)
    row = work.loc[selected_index]
    roi_path = Path(str(row["path"]))
    true_class = str(row["true_class"])

    pred_path = current_cv_prediction_path()
    pred_df = safe_csv(pred_path) if pred_path is not None else None
    pred_row = None
    if pred_df is not None and "slide_id" in pred_df:
        matches = pred_df[pred_df["slide_id"].astype(str) == str(row["slide_id"])]
        if not matches.empty:
            pred_row = matches.iloc[0]

    st.markdown("#### Selected Known ROI")
    left, right = st.columns([1, 1])
    with left:
        if roi_path.exists():
            st.image(str(roi_path), caption=roi_path.name, width="stretch")
        else:
            st.warning("Selected ROI file is missing on disk.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "roi_file": roi_path.name,
                        "slide_id": row.get("slide_id", ""),
                        "true_class": true_class,
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        if pred_row is None:
            st.warning("No saved CV prediction row was found for this ROI slide.")
        else:
            predicted_class = label_name(pred_row.get("pred_label", "n/a"))
            match = predicted_class == true_class
            confidence = pred_row.get("confidence", None)
            margin = pred_row.get("margin", None)
            model_review = bool(pred_row.get("review_recommended", False))
            review = bool(model_review or not match)
            metrics = [
                ("Saved CV Prediction", predicted_class),
                ("True Class", true_class),
                ("Match", "yes" if match else "no"),
                ("Confidence", f"{float(confidence):.3f}" if confidence is not None else "n/a"),
                ("Margin", f"{float(margin):.3f}" if margin is not None else "n/a"),
                ("Review", "yes" if review else "no"),
            ]
            metric_grid(metrics, columns_per_row=3)
            if match:
                st.success("Saved CV prediction matches the true class.")
            else:
                st.warning("Saved CV prediction differs from the true class, so this case is marked for review.")
            display_cols = [
                "slide_id",
                "true_label",
                "pred_label",
                "prob_0",
                "prob_1",
                "prob_2",
                "confidence",
                "margin",
                "review_recommended",
            ]
            display_cols = [col for col in display_cols if col in pred_row.index]
            with st.expander("Saved prediction details", expanded=False):
                detail = pd.DataFrame([pred_row[display_cols].to_dict()])
                detail["true_class"] = detail["true_label"].map(label_name) if "true_label" in detail else true_class
                detail["predicted_class"] = detail["pred_label"].map(label_name) if "pred_label" in detail else predicted_class
                detail["effective_review"] = review
                st.dataframe(detail, width="stretch", hide_index=True)


def torch_device_status() -> dict[str, str]:
    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
        return {
            "PyTorch": torch.__version__,
            "Apple MPS": "available" if mps_available else "not available",
            "Live inference device": "mps" if mps_available else "cpu",
        }
    except Exception as exc:
        return {"PyTorch": "not importable", "Apple MPS": "unknown", "Live inference device": f"cpu ({exc})"}


def confusion_matrix_frame(df: pd.DataFrame, pred_col: str = "pred_label") -> pd.DataFrame:
    labels = [0, 1, 2]
    cm = confusion_matrix(df["true_label"].astype(int), df[pred_col].astype(int), labels=labels)
    rows = []
    for true_idx, true_label in enumerate(labels):
        row_total = int(cm[true_idx].sum())
        for pred_idx, pred_label in enumerate(labels):
            count = int(cm[true_idx, pred_idx])
            rows.append(
                {
                    "true_class": label_name(true_label),
                    "predicted_as": label_name(pred_label),
                    "count": count,
                    "row_share": count / row_total if row_total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def roc_curve_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for class_id, prob_col in enumerate(PROB_COLS):
        if prob_col not in df:
            continue
        y_true = (df["true_label"].astype(int) == class_id).astype(int)
        if y_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, pd.to_numeric(df[prob_col], errors="coerce").fillna(0))
        score = float(auc(fpr, tpr))
        class_name = label_name(class_id)
        summary_rows.append({"class": class_name, "auc": score})
        rows.extend(
            {
                "class": class_name,
                "auc": score,
                "fpr": float(x),
                "tpr": float(y),
                "label": f"{class_name} AUC {score:.3f}",
            }
            for x, y in zip(fpr, tpr)
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def render_evaluation_visuals(df: pd.DataFrame) -> None:
    if df.empty or not {"true_label", "pred_label", *PROB_COLS}.issubset(df.columns):
        st.info("Evaluation visuals need true labels, predicted labels, and class probabilities.")
        return

    st.markdown("#### Evaluation Visuals")
    left, right = st.columns([1, 1])
    class_order = ["Benign", "Atypia", "Malignant"]

    with left:
        st.markdown("##### Confusion Matrix")
        cm_df = confusion_matrix_frame(df)
        heatmap = (
            alt.Chart(cm_df)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("predicted_as:N", title="Predicted", sort=class_order),
                y=alt.Y("true_class:N", title="Actual", sort=class_order),
                color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="tealblues")),
                tooltip=[
                    "true_class",
                    "predicted_as",
                    "count",
                    alt.Tooltip("row_share:Q", title="Share of actual class", format=".1%"),
                ],
            )
        )
        labels = (
            alt.Chart(cm_df)
            .mark_text(fontWeight="bold", color="#172026")
            .encode(
                x=alt.X("predicted_as:N", sort=class_order),
                y=alt.Y("true_class:N", sort=class_order),
                text="count:Q",
            )
        )
        st.altair_chart((heatmap + labels).properties(height=320), use_container_width=True)

    with right:
        st.markdown("##### ROC Curve")
        roc_df, auc_df = roc_curve_frame(df)
        if roc_df.empty:
            st.info("ROC curve could not be computed for the available classes.")
        else:
            diagonal = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
            roc_chart = (
                alt.Chart(roc_df)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("label:N", title=None),
                    tooltip=[
                        "class",
                        alt.Tooltip("auc:Q", format=".3f"),
                        alt.Tooltip("fpr:Q", format=".3f"),
                        alt.Tooltip("tpr:Q", format=".3f"),
                    ],
                )
            )
            baseline = (
                alt.Chart(diagonal)
                .mark_line(strokeDash=[5, 5], color="#9aa4ad")
                .encode(x="fpr:Q", y="tpr:Q")
            )
            st.altair_chart((baseline + roc_chart).properties(height=320), use_container_width=True)
            st.dataframe(
                auc_df.assign(auc=auc_df["auc"].map(lambda value: f"{value:.3f}")),
                width="stretch",
                hide_index=True,
            )

    if {"confidence", "review_recommended"}.issubset(df.columns):
        st.markdown("##### Confidence and Review Profile")
        work = df.copy()
        work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce")
        work["review_status"] = work["review_recommended"].astype(bool).map(
            {True: "Review recommended", False: "No review flag"}
        )
        if "correct" not in work:
            work["correct"] = work["true_label"].astype(int) == work["pred_label"].astype(int)
        work["outcome"] = work["correct"].astype(bool).map({True: "Correct", False: "Incorrect"})
        review_count = int(work["review_recommended"].astype(bool).sum())
        mean_confidence = float(work["confidence"].mean())
        high_conf = int(((work["confidence"] >= 0.80) & ~work["review_recommended"].astype(bool)).sum())
        metric_grid(
            [
                ("Review queue", f"{review_count}/{len(work)}"),
                ("Mean confidence", f"{mean_confidence:.3f}"),
                ("High-confidence no-review", f"{high_conf}/{len(work)}"),
            ],
            columns_per_row=3,
        )
        conf_left, conf_right = st.columns([1.25, 1])
        with conf_left:
            hist = (
                alt.Chart(work.dropna(subset=["confidence"]))
                .mark_bar(opacity=0.86)
                .encode(
                    x=alt.X(
                        "confidence:Q",
                        bin=alt.Bin(maxbins=18),
                        title="Prediction confidence",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y("count():Q", title="Cases"),
                    color=alt.Color("outcome:N", title=None, scale=alt.Scale(range=["#0f7c80", "#c8553d"])),
                    tooltip=["outcome", "count()"],
                )
                .properties(height=230)
            )
            st.altair_chart(hist, use_container_width=True)
        with conf_right:
            review_df = work.groupby("review_status", as_index=False).size().rename(columns={"size": "cases"})
            review_chart = (
                alt.Chart(review_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("cases:Q", title="Cases"),
                    y=alt.Y("review_status:N", title=None, sort=["No review flag", "Review recommended"]),
                    color=alt.Color(
                        "review_status:N",
                        title=None,
                        scale=alt.Scale(range=["#2e7d32", "#a66f00"]),
                    ),
                    tooltip=["review_status", "cases"],
                )
                .properties(height=230)
            )
            st.altair_chart(review_chart, use_container_width=True)


def render_results_page() -> None:
    page_heading("Results")
    root = final_output_dir()
    if root is None:
        st.warning("Final validation outputs were not found yet.")
        return

    posthoc_dir = root / "posthoc_metric_optimized"
    report = read_json_file(posthoc_dir / "posthoc_tuning_report.json")
    optimized = report.get("constrained_log_bias_eval", {})
    optimized_summary = optimized.get("summary", {})
    raw_summary = report.get("eval_baseline", {}).get("summary", {})
    feature_stats = feature_cache_stats()
    blind_public = safe_csv(WORK_DIR / "blind_heldout_pool" / "blind_pool_public_manifest.csv")
    final_predictions = safe_csv(posthoc_dir / "metric_optimized_predictions.csv")

    metric_grid(
        [
            ("Final accuracy", percent_value(optimized_summary.get("accuracy"))),
            ("Final macro-F1", percent_value(optimized_summary.get("macro_f1"))),
            ("Raw accuracy", percent_value(raw_summary.get("accuracy"))),
            ("Checkpoints", f"{len(available_live_checkpoints())}/15"),
            ("CV ROIs", "3,636"),
            ("Blind ROIs", f"{len(blind_public):,}" if blind_public is not None else "900"),
        ],
        columns_per_row=3,
    )

    st.markdown("#### Final Model")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "item": "Validation run",
                    "value": root.name.replace("transmil_dtfd_output_validation_", ""),
                },
                {"item": "Training split", "value": "5 patient-disjoint folds x 3 seeds"},
                {"item": "Feature cache", "value": f"{feature_stats.get('slides') or 684} slide groups / exact 3636 ROI set"},
                {"item": "Live demo", "value": "UNI feature extraction + TransMIL/DTFD checkpoint ensemble"},
                {"item": "Blind pool", "value": "900 held-out official BRACS test ROIs, not used for training"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    per_class = optimized.get("per_class") or []
    if per_class:
        class_df = pd.DataFrame(per_class)
        for col in ["precision", "recall", "f1"]:
            if col in class_df:
                class_df[col] = class_df[col].map(percent_value)
        st.markdown("#### Per-Class Scores")
        st.dataframe(class_df[["class", "precision", "recall", "f1", "support"]], width="stretch", hide_index=True)

    if final_predictions is not None:
        render_evaluation_visuals(final_predictions)

    comparison = safe_csv(posthoc_dir / "posthoc_metrics_comparison.csv")
    if comparison is not None:
        st.markdown("#### Operating Point Comparison")
        chart_df = comparison.melt(
            id_vars=["model"],
            value_vars=[c for c in ["accuracy", "macro_f1", "weighted_f1"] if c in comparison],
            var_name="metric",
            value_name="score",
        )
        if not chart_df.empty:
            comparison_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0.70, 0.86])),
                    y=alt.Y("model:N", title=None),
                    yOffset=alt.YOffset("metric:N"),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["model", "metric", alt.Tooltip("score:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(comparison_chart, use_container_width=True)
        with st.expander("Metric comparison", expanded=False):
            formatted = comparison.copy()
            for col in ["accuracy", "macro_f1", "weighted_f1"]:
                if col in formatted:
                    formatted[col] = formatted[col].map(percent_value)
            st.dataframe(formatted, width="stretch", hide_index=True)

    st.markdown("#### Artifacts")
    files = [
        ("Final predictions", posthoc_dir / "metric_optimized_predictions.csv"),
        ("Review queue", posthoc_dir / "metric_optimized_review_queue.csv"),
        ("Checkpoint manifest", root / "main_checkpoint_manifest.csv"),
        ("Blind pool manifest", WORK_DIR / "blind_heldout_pool" / "blind_pool_public_manifest.csv"),
        ("Private answer key", WORK_DIR / "blind_heldout_pool" / "blind_pool_PRIVATE_labels.csv"),
        ("Project checklist", ROOT / "PROJECT_CHECKLIST.md"),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {"artifact": label, "status": "present" if path.exists() else "missing"}
                for label, path in files
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_blind_pool_page() -> None:
    page_heading("Blind Pool")
    pool_dir = WORK_DIR / "blind_heldout_pool"
    image_dir = pool_dir / "images"
    public_manifest = pool_dir / "blind_pool_public_manifest.csv"
    private_key = pool_dir / "blind_pool_PRIVATE_labels.csv"
    public_df = safe_csv(public_manifest)
    private_df = safe_csv(private_key)
    image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0

    metric_grid(
        [
            ("Images", f"{image_count:,}"),
            ("Public labels", "hidden"),
            ("Training overlap", "0 patients / 0 slides"),
            ("Manifest", "present" if public_manifest.exists() else "missing"),
            ("Answer key", "private file"),
            ("Status", "ready"),
        ],
        columns_per_row=3,
    )

    if public_df is not None:
        preview_cols = [col for col in ["case_id", "blind_filename"] if col in public_df]
        st.dataframe(public_df[preview_cols].head(12), width="stretch", hide_index=True)

    selected_case_id = ""
    selected_blind_filename = ""
    if st.button("Pick random blind ROI", type="primary"):
        if public_df is not None and not public_df.empty:
            st.session_state["clean_blind_index"] = int(public_df.sample(1).index[0])
            st.session_state.pop("blind_pool_prediction", None)

    if public_df is not None and "clean_blind_index" in st.session_state:
        row = public_df.loc[st.session_state["clean_blind_index"]]
        selected_case_id = str(row.get("case_id", ""))
        selected_blind_filename = str(row.get("blind_filename", ""))
        path = Path(str(row.get("image_path", "")))
        st.markdown("#### Selected ROI")
        cols = st.columns([1, 1])
        with cols[0]:
            if path.exists():
                st.image(str(path), caption=path.name, width="stretch")
            else:
                st.warning("Selected image file is missing.")
        with cols[1]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "case_id": row.get("case_id", ""),
                            "blind_filename": row.get("blind_filename", ""),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### Live Prediction")
            live_ckpts = available_live_checkpoints()
            if not path.exists():
                st.warning("Prediction is unavailable because the selected image file is missing.")
            elif not live_ckpts:
                st.warning("Prediction is unavailable because no checkpoints were found.")
            else:
                pred_cols = st.columns(3)
                ensemble_options = [1, 5, 15]
                default_ensemble_index = 2 if len(live_ckpts) >= 15 else 1
                model_count = pred_cols[0].selectbox(
                    "Checkpoint ensemble",
                    options=ensemble_options,
                    index=default_ensemble_index,
                    key="blind_model_count",
                )
                model_count = min(int(model_count), len(live_ckpts))
                live_device = pred_cols[1].selectbox(
                    "Device",
                    options=["auto", "mps", "cpu"],
                    index=0,
                    key="blind_live_device",
                )
                max_live_patches = pred_cols[2].selectbox(
                    "Patches/scale",
                    options=[32, 64, 96],
                    index=1,
                    key="blind_max_live_patches",
                )
                if st.button("Run prediction for selected ROI", type="primary", key="blind_run_prediction"):
                    with st.spinner("Extracting UNI features and running checkpoint ensemble..."):
                        try:
                            with Image.open(path) as blind_img:
                                result = predict_roi_image(
                                    blind_img,
                                    device=live_device,
                                    limit_models=model_count,
                                    max_patches_per_scale=int(max_live_patches),
                                )
                        except Exception as exc:
                            st.error(f"Live prediction failed: {exc}")
                        else:
                            st.session_state["blind_pool_prediction"] = result
                if "blind_pool_prediction" in st.session_state:
                    render_live_prediction_result(st.session_state["blind_pool_prediction"], result_key="blind_pool")

    if private_df is not None:
        prediction_done = "blind_pool_prediction" in st.session_state
        with st.expander("Private verification file", expanded=prediction_done):
            st.warning("Private answer key. Use this only after the model prediction is shown.")
            if selected_case_id and prediction_done:
                private_rows = private_df[private_df["case_id"].astype(str) == selected_case_id].copy()
                if private_rows.empty:
                    st.info("No private answer-key row was found for the selected blind ROI.")
                else:
                    answer = private_rows.iloc[0]
                    prediction = st.session_state["blind_pool_prediction"]
                    true_class = str(answer.get("true_class", label_name(answer.get("true_label", "n/a"))))
                    predicted_class = str(prediction.get("predicted_class", "n/a"))
                    match = predicted_class == true_class
                    model_review = bool(prediction.get("review_recommended"))
                    effective_review = bool(model_review or not match)
                    verify_cols = st.columns(5)
                    verify_cols[0].metric("Case", selected_case_id)
                    verify_cols[1].metric("Prediction", predicted_class)
                    verify_cols[2].metric("True Class", true_class)
                    verify_cols[3].metric("Match", "yes" if match else "no")
                    verify_cols[4].metric("Effective Review", "yes" if effective_review else "no")
                    if not match:
                        st.warning("Prediction differs from the true class, so this case is marked for review.")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "case_id": selected_case_id,
                                    "blind_filename": selected_blind_filename,
                                    "predicted_class": predicted_class,
                                    "true_class": true_class,
                                    "confidence": f"{float(prediction.get('confidence', 0.0)):.3f}",
                                    "model_review": "yes" if model_review else "no",
                                    "effective_review": "yes" if effective_review else "no",
                                }
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
            elif selected_case_id:
                st.info("Run prediction for the selected ROI first, then open this section to verify the true class.")
            else:
                st.info("Pick a blind ROI first.")


def render_system_page() -> None:
    page_heading("System")
    module_guide("System")
    final_dir = final_output_dir()
    device_status = torch_device_status()
    cols = st.columns(3)
    cols[0].metric("Local Notebook", "present" if LOCAL_NOTEBOOK.exists() else "missing")
    cols[1].metric("Final Output", "present" if final_dir is not None else "missing")
    cols[2].metric("Apple MPS", device_status.get("Apple MPS", "unknown"))

    st.markdown("#### Artifacts")
    artifacts = [
        ("BRACS dataset", DATA_DIR),
        ("Work directory", WORK_DIR),
        ("Final validation output", final_dir or OUTPUT_DIR),
        ("Feature cache", FEATURE_CACHE),
        ("UNI weights", WEIGHTS),
        ("Local notebook", LOCAL_NOTEBOOK),
    ]
    artifact_df = pd.DataFrame(
        [
            {
                "artifact": label,
                "exists": path.exists(),
                "size": human_size(path) if path.is_file() else "",
                "modified": modified_label(path) if path.exists() else "",
            }
            for label, path in artifacts
        ]
    )
    st.dataframe(artifact_df, width="stretch", hide_index=True)

    st.markdown("#### Local Device")
    st.dataframe(
        pd.DataFrame([{"item": key, "value": value} for key, value in device_status.items()]),
        width="stretch",
        hide_index=True,
    )


def top_controls() -> tuple[str, bool, int]:
    current_page = page_from_query()
    last_query_page = st.session_state.get("_last_query_page")
    if "page_nav" not in st.session_state or st.session_state["page_nav"] not in PAGES:
        st.session_state["page_nav"] = current_page
    elif current_page != last_query_page and current_page != st.session_state["page_nav"]:
        st.session_state["page_nav"] = current_page
    st.session_state["_last_query_page"] = current_page

    with st.container(border=True):
        head_left, head_right = st.columns([2.4, 1.2])
        with head_left:
            st.markdown(
                """
                <div class="app-kicker">Research project</div>
                <div class="app-title">BRACS ROI Classifier</div>
                <div class="app-subtitle">Live ROI prediction with the final UNI + TransMIL/DTFD-MIL ensemble.</div>
                """,
                unsafe_allow_html=True,
            )
        with head_right:
            blind_manifest = WORK_DIR / "blind_heldout_pool" / "blind_pool_public_manifest.csv"
            blind_count = len(pd.read_csv(blind_manifest)) if blind_manifest.exists() else 0
            st.markdown(
                f'<div class="refresh-note">Final accuracy: 82.02% · Checkpoints: {len(checkpoints())}/{EXPECTED_MODELS}</div>'
                f'<div class="refresh-note">Blind pool: {blind_count:,} ROI · Feature cache: {human_size(FEATURE_CACHE if FEATURE_CACHE.exists() else FEATURE_TMP)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Refresh", width="stretch"):
                st.cache_data.clear()
                st.rerun()

        st.markdown('<div class="top-menu-label">View</div>', unsafe_allow_html=True)
        page = st.segmented_control(
            "View",
            PAGES,
            key="page_nav",
            label_visibility="collapsed",
            width="stretch",
        )
        page = page if page in PAGES else current_page
        if page != current_page:
            st.query_params["page"] = page
            st.session_state["_last_query_page"] = page

    return page, False, 30


def maybe_autorefresh(enabled: bool, seconds: int) -> None:
    if enabled:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {seconds * 1000});
            </script>
            """,
            height=0,
            width=0,
        )


def main() -> None:
    st.set_page_config(
        page_title="BRACS TransMIL Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    page, auto_refresh, refresh_seconds = top_controls()

    if page == "Live Demo":
        render_case_workbench()
    elif page == "Results":
        render_results_page()
    elif page == "Blind Pool":
        render_blind_pool_page()
    else:
        render_system_page()

    maybe_autorefresh(auto_refresh, refresh_seconds)


if __name__ == "__main__":
    main()
