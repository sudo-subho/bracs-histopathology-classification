#!/usr/bin/env python3
"""Rebuild the BRACS feature cache using the localized, patched notebook cells."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "BRACS_TransMIL_DTFD_LOCAL.ipynb"


def exec_notebook_cells(cell_ids: set[str]) -> dict:
    nb = json.loads(NOTEBOOK.read_text())
    ns: dict = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code" or cell.get("id") not in cell_ids:
            continue
        source = "".join(cell.get("source", []))
        # Trusted project notebook cells are executed to reuse the training workflow.
        # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
        exec(compile(source, f"{NOTEBOOK.name}:{cell.get('id')}", "exec"), ns)  # nosec B102  # nosemgrep
    return ns


def validate_cache(cache_path: Path, expected_ids: set[str], scales: list[int], feature_dim: int) -> None:
    import h5py
    import numpy as np

    with h5py.File(cache_path, "r") as h5:
        keys = set(h5.keys())
        missing = sorted(expected_ids - keys)
        if missing:
            raise RuntimeError(f"Cache missing {len(missing)} expected slide groups; first={missing[0]}")
        for sid in sorted(keys):
            grp = h5[sid]
            if "labels" not in grp:
                raise RuntimeError(f"Cache missing labels for {sid}")
            for scale in scales:
                ds = f"features_{scale}" if len(scales) > 1 else "features"
                if ds not in grp:
                    raise RuntimeError(f"Cache missing {sid}/{ds}")
                arr = grp[ds][:]
                if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != feature_dim:
                    raise RuntimeError(f"Invalid shape for {sid}/{ds}: {arr.shape}")
                if np.isclose(arr, 0).all(axis=1).any():
                    raise RuntimeError(f"All-zero placeholder feature row in {sid}/{ds}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-backup", action="store_true", help="Skip copying the old cache before rebuild.")
    args = parser.parse_args()

    ns = exec_notebook_cells({"co001", "co002", "co003", "co005", "co007", "co008", "co009"})
    cfg = ns["CFG"]
    cache_path = Path(cfg["feature_cache"])
    scales = list(cfg.get("multi_scale_sizes", [224]))
    feature_dim = int(cfg.get("feature_dim", 1024))

    if cache_path.exists() and not args.no_backup:
        backup = cache_path.with_name(f"{cache_path.stem}_before_small_roi_fix_{time.strftime('%Y%m%d_%H%M%S')}{cache_path.suffix}")
        print(f"Backing up existing cache to {backup}")
        shutil.copy2(cache_path, backup)

    def slide_label_fn(path):
        import os

        fname = os.path.basename(path)
        parts = fname.replace(".png", "").split("_")
        if len(parts) >= 3 and parts[0] == "BRACS":
            slide_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            label = ns["class_to_idx"].get(parts[2], 0)
        else:
            slide_id = f"SLIDE_{parts[0]}_{parts[1]}"
            label = ns["class_to_idx"].get(parts[1], 0)
        return slide_id, label

    expected_ids = set(ns["trainval_df"]["slide_id"]).union(set(ns["test_df"]["slide_id"]))
    device = cfg["device"]
    print(f"Rebuilding {cache_path} for {len(expected_ids)} expected slide groups on {device}")
    encoder = ns["load_uni"](device)
    transform = ns["get_uni_transform"]()
    ns["extract_and_cache_features"](encoder, transform, ns["roi_deduped"], str(cache_path), slide_label_fn, device)

    validate_cache(cache_path, expected_ids, scales, feature_dim)
    print(f"Validated rebuilt cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
