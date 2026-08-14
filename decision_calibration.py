#!/usr/bin/env python3
"""Shared decision calibration helpers for BRACS 3-class predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_ATYPIA_THRESHOLD = 0.44
REVIEW_MARGIN_THRESHOLD = 0.10
ATYPIA_BOUNDARY_MARGIN = 0.05
PROB_COLS = ["prob_0", "prob_1", "prob_2"]
CLASS_NAME_BY_ID = {0: "Benign", 1: "Atypia", 2: "Malignant"}


def ensure_prob_cols(df: pd.DataFrame) -> None:
    missing = set(PROB_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")


def raw_argmax_predictions(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs).argmax(axis=1).astype(int)


def atypia_threshold_predictions(
    probs: np.ndarray,
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
) -> np.ndarray:
    probs = np.asarray(probs)
    non_atypia = np.where(probs[:, 0] >= probs[:, 2], 0, 2)
    return np.where(probs[:, 1] >= float(threshold), 1, non_atypia).astype(int)


def log_bias_predictions(
    probs: np.ndarray,
    biases: tuple[float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    scores = np.log(np.clip(np.asarray(probs), 1e-9, 1.0)) + np.asarray(biases, dtype=float)
    return scores.argmax(axis=1).astype(int)


def add_operating_point_columns(
    df: pd.DataFrame,
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
    pred_col: str = "pred_label",
) -> pd.DataFrame:
    ensure_prob_cols(df)
    out = df.copy()
    probs = out[PROB_COLS].to_numpy()
    raw = raw_argmax_predictions(probs)
    calibrated = atypia_threshold_predictions(probs, threshold)
    if "raw_pred_label" not in out.columns:
        out["raw_pred_label"] = out[pred_col].to_numpy() if pred_col in out.columns else raw
    out["calibrated_pred_label"] = calibrated
    out[pred_col] = calibrated
    out["calibration_method"] = f"frozen_atypia_threshold_{threshold:.3f}"
    out["atypia_threshold"] = float(threshold)
    out["changed_prediction"] = out["raw_pred_label"].astype(int) != out["calibrated_pred_label"].astype(int)
    return add_confidence_columns(out, pred_col=pred_col, threshold=threshold)


def add_confidence_columns(
    df: pd.DataFrame,
    pred_col: str = "pred_label",
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
) -> pd.DataFrame:
    ensure_prob_cols(df)
    out = df.copy()
    probs = out[PROB_COLS].to_numpy()
    sorted_probs = np.sort(probs, axis=1)
    raw_pred = raw_argmax_predictions(probs)
    active_pred = out[pred_col].to_numpy().astype(int) if pred_col in out.columns else raw_pred
    out["raw_confidence"] = probs[np.arange(len(out)), raw_pred]
    out["confidence"] = probs[np.arange(len(out)), active_pred]
    out["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["atypia_boundary_distance"] = np.abs(probs[:, 1] - float(threshold))
    out["review_recommended"] = (
        (out["margin"] <= REVIEW_MARGIN_THRESHOLD)
        | (out["atypia_boundary_distance"] <= ATYPIA_BOUNDARY_MARGIN)
    )
    return out


def high_uncertainty_queue(
    df: pd.DataFrame,
    pred_col: str = "pred_label",
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
    limit: int | None = 120,
) -> pd.DataFrame:
    out = add_confidence_columns(df, pred_col=pred_col, threshold=threshold)
    queue = out[out["review_recommended"]].copy()
    if queue.empty:
        queue = out.copy()
    queue = queue.sort_values(
        ["review_recommended", "margin", "atypia_boundary_distance", "confidence"],
        ascending=[False, True, True, True],
    )
    return queue.head(limit) if limit is not None else queue
