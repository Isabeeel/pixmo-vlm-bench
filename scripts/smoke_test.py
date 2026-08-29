#!/usr/bin/env python3
"""Validate the pipeline end to end (generate -> parse -> score) on CPU,
using a tiny model that is NOT one of the 5 benchmark candidates. This only
proves the plumbing works; it says nothing about the real candidates'
quality - see models.SMOKE_MODEL.

    python scripts/smoke_test.py [n_examples]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from pixmo_bench.data import DEFAULT_OUT_DIR
from pixmo_bench.infer import run_model
from pixmo_bench.models import SMOKE_MODEL
from pixmo_bench.report import score_model

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    raw_path = run_model(SMOKE_MODEL, DEFAULT_OUT_DIR, limit=n, torch_dtype=torch.float32)
    print(f"wrote {raw_path}")

    for line in raw_path.read_text().splitlines():
        print(line)

    report = score_model(SMOKE_MODEL.key, DEFAULT_OUT_DIR)
    print(report)
