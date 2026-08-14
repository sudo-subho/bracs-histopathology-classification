#!/usr/bin/env python3
"""Convenience entry point for the recommended 3-seed x 5-fold validation run."""

from __future__ import annotations

import sys

from run_validation_1x5 import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--n-seeds", "3", "--n-folds", "5", *sys.argv[1:]]
    raise SystemExit(main())
