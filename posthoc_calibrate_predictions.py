#!/usr/bin/env python3
"""Post-hoc decision calibration for BRACS ensemble predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from decision_calibration import (
    CLASS_NAME_BY_ID,
    DEFAULT_ATYPIA_THRESHOLD,
    PROB_COLS,
    add_operating_point_columns,
    add_confidence_columns,
    atypia_threshold_predictions,
    high_uncertainty_queue,
    log_bias_predictions,
)

CLASS_NAMES = CLASS_NAME_BY_ID


def metrics_frame(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    rows = []
    for idx, label in enumerate([0, 1, 2]):
        rows.append(
            {
                "class_id": label,
                "class": CLASS_NAMES[label],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
    return summary, pd.DataFrame(rows)


def threshold_predictions(probs: np.ndarray, atypia_threshold: float) -> np.ndarray:
    return atypia_threshold_predictions(probs, atypia_threshold)


def bias_predictions(probs: np.ndarray, biases: tuple[float, float, float]) -> np.ndarray:
    return log_bias_predictions(probs, biases)


def tune_atypia_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict, pd.DataFrame, np.ndarray]:
    rows = []
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = threshold_predictions(probs, float(threshold))
        summary, per_class = metrics_frame(y_true, pred)
        atypia = per_class[per_class["class_id"] == 1].iloc[0]
        row = {
            "threshold": float(threshold),
            **summary,
            "atypia_precision": float(atypia["precision"]),
            "atypia_recall": float(atypia["recall"]),
            "atypia_f1": float(atypia["f1"]),
        }
        rows.append(row)
        key = (row["macro_f1"], row["accuracy"], row["atypia_f1"])
        if best is None or key > best[0]:
            best = (key, row, pred)
    assert best is not None
    return best[1], pd.DataFrame(rows), best[2]


def tune_log_bias(
    y_true: np.ndarray,
    probs: np.ndarray,
    baseline_pred: np.ndarray,
    min_atypia_recall: float | None = None,
    max_malignant_recall_drop: float = 0.03,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    baseline_summary, baseline_per_class = metrics_frame(y_true, baseline_pred)
    baseline_recall = baseline_per_class.set_index("class_id")["recall"].to_dict()
    effective_min_atypia_recall = (
        float(baseline_recall[1]) if min_atypia_recall is None else float(min_atypia_recall)
    )
    rows = []
    best = None

    # Biases are relative; keep class 0 at zero and tune Atypia/Malignant offsets.
    for bias_1 in np.linspace(-0.55, 0.30, 171):
        for bias_2 in np.linspace(-0.55, 0.30, 171):
            biases = (0.0, float(bias_1), float(bias_2))
            pred = bias_predictions(probs, biases)
            summary, per_class = metrics_frame(y_true, pred)
            recall = per_class.set_index("class_id")["recall"].to_dict()
            constrained = (
                recall[1] >= effective_min_atypia_recall
                and recall[2] >= baseline_recall[2] - max_malignant_recall_drop
                and summary["accuracy"] >= baseline_summary["accuracy"]
            )
            row = {
                "bias_0": biases[0],
                "bias_1": biases[1],
                "bias_2": biases[2],
                "constrained": bool(constrained),
                **summary,
                "benign_recall": float(recall[0]),
                "atypia_recall": float(recall[1]),
                "malignant_recall": float(recall[2]),
            }
            rows.append(row)
            if constrained:
                key = (row["macro_f1"], row["accuracy"], row["atypia_recall"])
                if best is None or key > best[0]:
                    best = (key, row, pred)

    if best is None:
        table = pd.DataFrame(rows)
        row = table.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0].to_dict()
        pred = bias_predictions(probs, (row["bias_0"], row["bias_1"], row["bias_2"]))
        best = ((row["macro_f1"], row["accuracy"], row["atypia_recall"]), row, pred)
    best[1]["effective_min_atypia_recall"] = effective_min_atypia_recall
    return best[1], pd.DataFrame(rows), best[2]


def review_queue(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = high_uncertainty_queue(df, pred_col=pred_col, limit=None)
    out["true_class"] = out["true_label"].map(CLASS_NAMES)
    out["predicted_as"] = out[pred_col].map(CLASS_NAMES)
    out["correct"] = out["true_label"] == out[pred_col]
    return out.sort_values(
        ["review_recommended", "correct", "margin", "atypia_boundary_distance"],
        ascending=[False, True, True, True],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("work/transmil_dtfd_output/heterogeneous_ensemble_predictions.csv"),
    )
    parser.add_argument(
        "--calibration-input",
        type=Path,
        default=None,
        help="Prediction CSV used to tune thresholds/biases. Defaults to --input.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Prediction CSV to transform/report after tuning. Defaults to --input.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work/transmil_dtfd_output"))
    parser.add_argument(
        "--min-atypia-recall",
        type=float,
        default=None,
        help="Minimum Atypia recall for log-bias tuning. Defaults to preserving raw baseline recall.",
    )
    parser.add_argument("--max-malignant-recall-drop", type=float, default=0.03)
    parser.add_argument("--atypia-threshold", type=float, default=DEFAULT_ATYPIA_THRESHOLD)
    args = parser.parse_args()

    calibration_input = args.calibration_input or args.input
    eval_input = args.eval_input or args.input
    calibration_df = pd.read_csv(calibration_input)
    eval_df = pd.read_csv(eval_input)
    required = {"slide_id", "true_label", "pred_label", *PROB_COLS}
    missing_calibration = required - set(calibration_df.columns)
    missing_eval = required - set(eval_df.columns)
    if missing_calibration:
        raise SystemExit(f"Missing required calibration columns: {sorted(missing_calibration)}")
    if missing_eval:
        raise SystemExit(f"Missing required eval columns: {sorted(missing_eval)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_y = calibration_df["true_label"].to_numpy()
    calibration_probs = calibration_df[PROB_COLS].to_numpy()
    calibration_baseline_col = "raw_pred_label" if "raw_pred_label" in calibration_df.columns else "pred_label"
    eval_baseline_col = "raw_pred_label" if "raw_pred_label" in eval_df.columns else "pred_label"
    calibration_baseline_pred = calibration_df[calibration_baseline_col].to_numpy()
    eval_y = eval_df["true_label"].to_numpy()
    eval_probs = eval_df[PROB_COLS].to_numpy()
    eval_baseline_pred = eval_df[eval_baseline_col].to_numpy()

    calibration_baseline_summary, calibration_baseline_per_class = metrics_frame(
        calibration_y, calibration_baseline_pred)
    threshold_best, threshold_sweep, _ = tune_atypia_threshold(calibration_y, calibration_probs)
    bias_best, bias_sweep, _ = tune_log_bias(
        calibration_y,
        calibration_probs,
        calibration_baseline_pred,
        min_atypia_recall=args.min_atypia_recall,
        max_malignant_recall_drop=args.max_malignant_recall_drop,
    )

    fixed_threshold_pred = threshold_predictions(eval_probs, args.atypia_threshold)
    fixed_threshold_summary, fixed_threshold_per_class = metrics_frame(eval_y, fixed_threshold_pred)
    threshold_eval_pred = threshold_predictions(eval_probs, threshold_best["threshold"])
    threshold_eval_summary, threshold_eval_per_class = metrics_frame(eval_y, threshold_eval_pred)
    bias_pred = bias_predictions(
        eval_probs, (bias_best["bias_0"], bias_best["bias_1"], bias_best["bias_2"]))
    baseline_summary, baseline_per_class = metrics_frame(eval_y, eval_baseline_pred)
    calibrated_summary, calibrated_per_class = metrics_frame(eval_y, bias_pred)

    calibrated = add_operating_point_columns(eval_df, threshold=args.atypia_threshold)
    calibrated["bias_pred_label"] = bias_pred
    calibrated["metric_optimized_pred_label"] = bias_pred
    calibrated["recommended_pred_label"] = bias_pred
    calibrated["bias_0"] = bias_best["bias_0"]
    calibrated["bias_1"] = bias_best["bias_1"]
    calibrated["bias_2"] = bias_best["bias_2"]
    calibrated["metric_optimized_method"] = "constrained_log_bias"
    calibrated["metric_optimized_changed"] = eval_baseline_pred != bias_pred

    optimized = calibrated.copy()
    optimized["frozen_pred_label"] = optimized["pred_label"].astype(int)
    optimized["pred_label"] = bias_pred
    optimized["calibrated_pred_label"] = bias_pred
    optimized["calibration_method"] = "constrained_log_bias"
    optimized["changed_prediction"] = optimized["raw_pred_label"].astype(int) != optimized["pred_label"].astype(int)
    optimized = add_confidence_columns(optimized, pred_col="pred_label", threshold=args.atypia_threshold)

    calibrated_path = args.output_dir / "calibrated_predictions.csv"
    optimized_path = args.output_dir / "metric_optimized_predictions.csv"
    threshold_path = args.output_dir / "threshold_tuning_sweep.csv"
    bias_path = args.output_dir / "bias_tuning_sweep.csv"
    report_path = args.output_dir / "posthoc_tuning_report.json"
    review_path = args.output_dir / "calibrated_error_review_queue.csv"
    optimized_review_path = args.output_dir / "metric_optimized_review_queue.csv"
    metrics_path = args.output_dir / "posthoc_metrics_comparison.csv"

    calibrated.to_csv(calibrated_path, index=False)
    optimized.to_csv(optimized_path, index=False)
    threshold_sweep.to_csv(threshold_path, index=False)
    bias_sweep.to_csv(bias_path, index=False)
    review_queue(calibrated, "pred_label").to_csv(review_path, index=False)
    review_queue(optimized, "pred_label").to_csv(optimized_review_path, index=False)

    metrics_comparison = pd.DataFrame(
        [
            {"model": "raw_argmax", **baseline_summary},
            {
                "model": f"frozen_atypia_threshold_{args.atypia_threshold:.3f}",
                **fixed_threshold_summary,
            },
            {
                "model": f"atypia_threshold_{threshold_best['threshold']:.3f}",
                **threshold_eval_summary,
            },
            {"model": "constrained_log_bias", **calibrated_summary},
        ]
    )
    metrics_comparison.to_csv(metrics_path, index=False)

    report = {
        "input": str(args.input),
        "calibration_input": str(calibration_input),
        "eval_input": str(eval_input),
        "calibration_baseline_column": calibration_baseline_col,
        "eval_baseline_column": eval_baseline_col,
        "outputs": {
            "calibrated_predictions": str(calibrated_path),
            "metric_optimized_predictions": str(optimized_path),
            "threshold_sweep": str(threshold_path),
            "bias_sweep": str(bias_path),
            "metrics_comparison": str(metrics_path),
            "review_queue": str(review_path),
            "metric_optimized_review_queue": str(optimized_review_path),
        },
        "calibration_baseline": {
            "summary": calibration_baseline_summary,
            "per_class": calibration_baseline_per_class.to_dict(orient="records"),
        },
        "eval_baseline": {"summary": baseline_summary, "per_class": baseline_per_class.to_dict(orient="records")},
        "best_atypia_threshold": threshold_best,
        "frozen_atypia_threshold": {
            "threshold": args.atypia_threshold,
            "eval_summary": fixed_threshold_summary,
            "eval_per_class": fixed_threshold_per_class.to_dict(orient="records"),
        },
        "best_log_bias": bias_best,
        "log_bias_constraints": {
            "min_atypia_recall": bias_best.get("effective_min_atypia_recall", args.min_atypia_recall),
            "max_malignant_recall_drop": args.max_malignant_recall_drop,
        },
        "threshold_eval": {
            "summary": threshold_eval_summary,
            "per_class": threshold_eval_per_class.to_dict(orient="records"),
        },
        "calibrated": {
            "summary": fixed_threshold_summary,
            "per_class": fixed_threshold_per_class.to_dict(orient="records"),
            "changed_predictions": int(calibrated["changed_prediction"].sum()),
        },
        "constrained_log_bias_eval": {
            "summary": calibrated_summary,
            "per_class": calibrated_per_class.to_dict(orient="records"),
            "changed_predictions": int((eval_baseline_pred != bias_pred).sum()),
        },
        "caution": "Tune on OOF/CV predictions and apply to held-out predictions for an honest operating-point check. Do not tune on the held-out test set for clinical claims.",
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
