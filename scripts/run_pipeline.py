#!/usr/bin/env python3
"""Steps 2-7 entry point - needs torch + a GPU (pip install -r requirements.txt first).

    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixmo_bench.data import DEFAULT_OUT_DIR
from pixmo_bench.infer import run_model
from pixmo_bench.models import MODEL_REGISTRY
from pixmo_bench.report import build_report, print_report

if __name__ == "__main__":
    for spec in MODEL_REGISTRY:
        print(f"running {spec.key} ({spec.repo_id}) ...")
        run_model(spec, DEFAULT_OUT_DIR)

    reports = build_report(DEFAULT_OUT_DIR)
    print_report(reports)
