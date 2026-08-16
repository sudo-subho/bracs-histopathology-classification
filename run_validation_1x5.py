#!/usr/bin/env python3
"""Run an isolated n-seed x n-fold TransMIL+DTFD validation from patched notebook cells."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from decision_calibration import (
    DEFAULT_ATYPIA_THRESHOLD,
    add_operating_point_columns,
    atypia_threshold_predictions,
)

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "BRACS_TransMIL_DTFD_LOCAL.ipynb"


def exec_notebook_cells(cell_ids: list[str]) -> dict:
    nb = json.loads(NOTEBOOK.read_text())
    wanted = set(cell_ids)
    ns: dict = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code" or cell.get("id") not in wanted:
            continue
        source = "".join(cell.get("source", []))
        # Trusted project notebook cells are executed to reuse the training workflow.
        # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
        exec(compile(source, f"{NOTEBOOK.name}:{cell.get('id')}", "exec"), ns)  # nosec B102  # nosemgrep
    return ns


def safe_torch_load(torch_module, path, map_location="cpu"):
    """Load trusted tensor checkpoints without enabling arbitrary pickle objects."""
    try:
        return torch_module.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location=map_location)  # nosec B614  # nosemgrep


def validate_cache(cache_path: Path, expected_ids: set[str], scales: list[int], feature_dim: int) -> None:
    import h5py
    import numpy as np

    with h5py.File(cache_path, "r") as h5:
        keys = set(h5.keys())
        missing = sorted(expected_ids - keys)
        if missing:
            raise RuntimeError(f"Feature cache missing {len(missing)} expected slides; first={missing[0]}")
        for sid in sorted(expected_ids):
            grp = h5[sid]
            if "labels" not in grp:
                raise RuntimeError(f"Feature cache missing labels for {sid}")
            for scale in scales:
                ds = f"features_{scale}" if len(scales) > 1 else "features"
                if ds not in grp:
                    raise RuntimeError(f"Feature cache missing {sid}/{ds}")
                arr = grp[ds][:]
                if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != feature_dim:
                    raise RuntimeError(f"Invalid feature shape for {sid}/{ds}: {arr.shape}")
                if np.isclose(arr, 0).all(axis=1).any():
                    raise RuntimeError(f"All-zero placeholder feature row in {sid}/{ds}")


def aggregate_oof(all_results: dict, np):
    from collections import defaultdict

    slide_probs = defaultdict(list)
    slide_labels = {}
    for fold_results in all_results.values():
        for result in fold_results.values():
            for sid, prob, label in zip(result["ids"], result["probs"], result["labels"]):
                slide_probs[sid].append(prob)
                slide_labels[sid] = int(label)
    ids = sorted(slide_probs.keys())
    probs = np.array([np.mean(slide_probs[sid], axis=0) for sid in ids])
    labels = np.array([slide_labels[sid] for sid in ids])
    preds = probs.argmax(axis=1)
    return ids, probs, labels, preds


def prediction_frame(pd, ids, labels, probs, raw_preds, threshold: float):
    pred_df = pd.DataFrame({"slide_id": ids, "true_label": labels, "pred_label": raw_preds})
    for i in range(probs.shape[1]):
        pred_df[f"prob_{i}"] = probs[:, i]
    return add_operating_point_columns(pred_df, threshold=threshold)


def run_heldout_test(
    ns: dict,
    output_dir: Path,
    n_bootstrap: int = 1000,
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
    artifact_prefix: str = "validation_1x5",
):
    import gc
    import os

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    cfg = ns["CFG"]
    device = cfg["device"]
    manifest_path = output_dir / "main_checkpoint_manifest.csv"
    if not manifest_path.exists():
        print("Held-out test skipped: main_checkpoint_manifest.csv is missing")
        return None
    manifest = pd.read_csv(manifest_path).sort_values("fold_seed")
    ckpt_files = [Path(p) for p in manifest["path"].tolist() if Path(p).exists()]
    if not ckpt_files:
        print("Held-out test skipped: no checkpoints listed in manifest exist")
        return None

    test_df = ns["test_df"]
    test_slides = sorted(test_df["slide_id"].unique().tolist())
    test_ds = ns["MILDataset"](
        cfg["feature_cache"],
        test_slides,
        max_patches=cfg["max_patches_per_bag"],
        scale_sizes=cfg.get("multi_scale_sizes", [224]),
    )
    if len(cfg.get("multi_scale_sizes", [224])) > 1:
        collate_fn = lambda b: ns["padded_collate_multiscale"](b, max_patches=cfg["max_patches_per_bag"])
    else:
        collate_fn = lambda b: ns["padded_collate"](b, max_patches=cfg["max_patches_per_bag"])
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)

    slide_probs: dict[str, list] = {}
    slide_labels: dict[str, int] = {}
    for ckpt_path in ckpt_files:
        ckpt = safe_torch_load(torch, ckpt_path, map_location="cpu")
        if ckpt.get("checkpoint_role") not in (None, "main"):
            print(f"Skipping non-main checkpoint: {ckpt_path}")
            continue
        model = ns["TransMIL"](
            in_dim=cfg["feature_dim"],
            n_classes=cfg["num_classes"],
            dim=cfg["transmil_dim"],
            n_layers=cfg["transmil_layers"],
            num_heads=cfg["transmil_heads"],
            num_landmarks=cfg["transmil_landmarks"],
            dropout=cfg["dropout"],
            max_len=cfg["max_patches_per_bag"],
        )
        if cfg["n_pseudo_bags"] > 0:
            model = ns["DTFDWrapper"](model, n_pseudo=cfg["n_pseudo_bags"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        if cfg.get("use_swa", False) and ckpt.get("swa_state"):
            swa = ns["SWAWrapper"](model, start_epoch=cfg["swa_start"], swa_lr=cfg["swa_lr"])
            swa.swa_state = ckpt["swa_state"]
            swa.n_averaged = int(ckpt.get("swa_n_averaged", 1) or 1)
            swa.apply_to(model)
        elif cfg.get("use_ema", False) and ckpt.get("ema_state"):
            ema = ns["EMAWrapper"](model, decay=cfg["ema_decay"])
            ema.load_state_dict(ckpt["ema_state"])
            ema.apply_to(model)
        with torch.inference_mode():
            for feats, labels, ids, mask in loader:
                if isinstance(feats, dict):
                    feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
                    mask = {k: v.to(device, non_blocking=True) for k, v in mask.items()}
                else:
                    feats = feats.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                out = model(feats, mask)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                probs = F.softmax(out, dim=-1).cpu().numpy()
                for i, sid in enumerate(ids):
                    slide_probs.setdefault(sid, []).append(probs[i])
                    slide_labels[sid] = int(labels[i].item() if hasattr(labels[i], "item") else labels[i])
        del model, ckpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids = sorted(slide_probs.keys())
    test_probs = np.array([np.mean(slide_probs[sid], axis=0) for sid in test_ids])
    test_labels = np.array([slide_labels[sid] for sid in test_ids])
    raw_test_preds = test_probs.argmax(axis=1)
    calibrated_test_preds = atypia_threshold_predictions(test_probs, threshold)
    raw_metrics, raw_per_class = ns["compute_all_metrics"](test_labels, raw_test_preds, test_probs, cfg["class_names"], n_bootstrap)
    metrics, per_class = ns["compute_all_metrics"](test_labels, calibrated_test_preds, test_probs, cfg["class_names"], n_bootstrap)
    test_df_out = prediction_frame(pd, test_ids, test_labels, test_probs, raw_test_preds, threshold)
    test_payload = {
        "metrics": metrics,
        "per_class": per_class,
        "raw_metrics": raw_metrics,
        "raw_per_class": raw_per_class,
        "atypia_threshold": threshold,
    }
    test_df_out.to_csv(output_dir / f"{artifact_prefix}_test_predictions.csv", index=False)
    (output_dir / f"{artifact_prefix}_test_metrics.json").write_text(json.dumps(test_payload, indent=2))
    if artifact_prefix != "validation_1x5":
        test_df_out.to_csv(output_dir / "validation_1x5_test_predictions.csv", index=False)
        (output_dir / "validation_1x5_test_metrics.json").write_text(json.dumps(test_payload, indent=2))
    return metrics, raw_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--single-gpu", action="store_true", help="Disable DataParallel even when multiple CUDA devices are visible.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for macro-F1 confidence intervals.")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--atypia-threshold", type=float, default=DEFAULT_ATYPIA_THRESHOLD)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--atypia-weight-scale", type=float, default=0.90)
    args = parser.parse_args()

    cells = ["co001", "co002", "co003", "co005", "co011", "co012", "co013", "co015", "co020", "co021"]
    ns = exec_notebook_cells(cells)
    cfg = ns["CFG"]
    if args.output_dir is None:
        args.output_dir = ROOT / "work" / f"transmil_dtfd_output_validation_{args.n_seeds}x{args.n_folds}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)
    cfg["feature_cache"] = str((ROOT / "work" / "bracs_features.h5").resolve())
    cfg["n_seeds"] = int(args.n_seeds)
    cfg["n_folds"] = int(args.n_folds)
    cfg["run_tag"] = "main"
    cfg["atypia_threshold"] = float(args.atypia_threshold)
    cfg["class_weight_power"] = float(args.class_weight_power)
    cfg["class_weight_scales"] = [1.0, float(args.atypia_weight_scale), 1.0]

    import numpy as np
    import pandas as pd
    import torch

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cfg["use_dataparallel"] = visible_gpus > 1 and not args.single_gpu
    if visible_gpus:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(visible_gpus)]
        print(f"Visible CUDA devices: {gpu_names}")
    print(f"DataParallel enabled: {cfg['use_dataparallel']}")

    expected_ids = set(ns["trainval_df"]["slide_id"]).union(set(ns["test_df"]["slide_id"]))
    validate_cache(Path(cfg["feature_cache"]), expected_ids, list(cfg.get("multi_scale_sizes", [224])), int(cfg.get("feature_dim", 1024)))

    if args.n_folds != len(ns["splits"]):
        ns["splits"] = ns["create_bracs_splits"](str(ROOT / "work" / "metadata.csv"), n_folds=args.n_folds, seed=cfg["seed"])

    print(f"Running isolated {args.n_seeds} x {args.n_folds} validation into {output_dir}")
    artifact_prefix = f"validation_{args.n_seeds}x{args.n_folds}"
    start = time.time()
    all_results = ns["run_multi_seed_cv"](ns["splits"], cfg["feature_cache"], cfg["device"])
    ids, probs, labels, raw_preds = aggregate_oof(all_results, np)
    calibrated_preds = atypia_threshold_predictions(probs, args.atypia_threshold)
    raw_metrics, raw_per_class = ns["compute_all_metrics"](labels, raw_preds, probs, cfg["class_names"], args.bootstrap)
    metrics, per_class = ns["compute_all_metrics"](labels, calibrated_preds, probs, cfg["class_names"], args.bootstrap)

    prediction_frame(pd, ids, labels, probs, raw_preds, args.atypia_threshold).to_csv(
        output_dir / f"validation_{args.n_seeds}x{args.n_folds}_oof_predictions.csv", index=False)
    # Keep the historical filename for dashboards/scripts that look for it.
    prediction_frame(pd, ids, labels, probs, raw_preds, args.atypia_threshold).to_csv(
        output_dir / "validation_1x5_oof_predictions.csv", index=False)
    summary = {
        "elapsed_seconds": time.time() - start,
        "output_dir": str(output_dir),
        "n_seeds": args.n_seeds,
        "n_folds": args.n_folds,
        "operating_point": f"frozen_atypia_threshold_{args.atypia_threshold:.3f}",
        "atypia_threshold": args.atypia_threshold,
        "class_weight_power": args.class_weight_power,
        "atypia_weight_scale": args.atypia_weight_scale,
        "oof_metrics": metrics,
        "oof_per_class": per_class,
        "raw_oof_metrics": raw_metrics,
        "raw_oof_per_class": raw_per_class,
    }
    if not args.skip_test:
        summary["heldout_test_metrics"], summary["raw_heldout_test_metrics"] = run_heldout_test(
            ns, output_dir, args.bootstrap, args.atypia_threshold, artifact_prefix)
    (output_dir / f"{artifact_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    if artifact_prefix != "validation_1x5":
        (output_dir / "validation_1x5_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nValidation {args.n_seeds}x{args.n_folds} OOF metrics (frozen Atypia threshold):")
    for key, value in metrics.items():
        if "ci" not in key:
            print(f"  {key}: {value}")
    print(f"\nValidation {args.n_seeds}x{args.n_folds} raw OOF metrics:")
    for key, value in raw_metrics.items():
        if "ci" not in key:
            print(f"  {key}: {value}")
    if summary.get("heldout_test_metrics"):
        print(f"\nValidation {args.n_seeds}x{args.n_folds} held-out test metrics (frozen Atypia threshold):")
        for key, value in summary["heldout_test_metrics"].items():
            if "ci" not in key:
                print(f"  {key}: {value}")
        print(f"\nValidation {args.n_seeds}x{args.n_folds} raw held-out test metrics:")
        for key, value in summary["raw_heldout_test_metrics"].items():
            if "ci" not in key:
                print(f"  {key}: {value}")
    print(f"\nArtifacts: {output_dir}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
