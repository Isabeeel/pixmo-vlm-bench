#!/usr/bin/env python3
"""Step 1 entry point - runnable now, no GPU needed.

    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixmo_bench.data import build_subset

if __name__ == "__main__":
    build_subset(n=150, seed=0)
