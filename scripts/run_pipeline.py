#!/usr/bin/env python3
"""Steps 2-7 entry point - needs torch + a GPU (pip install -r requirements.txt first).

    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixmo_bench.data import DEFAULT_OUT_DIR, load_subset
from pixmo_bench.infer import RESULTS_DIR, run_model
from pixmo_bench.models import MODEL_REGISTRY
from pixmo_bench.report import build_report, print_report

if __name__ == "__main__":
    n_examples = len(load_subset(DEFAULT_OUT_DIR))
    failed = []
    for spec in MODEL_REGISTRY:
        out_path = RESULTS_DIR / f"{spec.key}.raw.jsonl"
        if out_path.exists() and sum(1 for _ in out_path.open()) >= n_examples:
            print(f"skipping {spec.key}: {out_path} already has {n_examples}+ results")
            continue
        print(f"running {spec.key} ({spec.repo_id}) ...")
        try:
            run_model(spec, DEFAULT_OUT_DIR)
        except Exception as e:
            print(f"!! {spec.key} failed, skipping: {e!r}")
            failed.append(spec.key)

    reports = build_report(DEFAULT_OUT_DIR, only_keys=[s.key for s in MODEL_REGISTRY if s.key not in failed])
    print_report(reports)
    if failed:
        print(f"\nskipped (failed to run): {', '.join(failed)}")
