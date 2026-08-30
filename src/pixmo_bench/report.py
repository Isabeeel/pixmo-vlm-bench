"""Step 7: roll up per-model results into one comparison table."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .data import Example, load_subset
from .models import MODEL_REGISTRY
from .parsing import parse_output
from .scoring import lexical_overlap_score, point_hit, rescale_to_percent

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class ModelReport:
    key: str
    n: int
    parse_failure_rate: float
    pointing_accuracy: float
    # Same as pointing_accuracy but rescales point-shaped outputs that are
    # out of the requested 0-100 range (e.g. Qwen3-VL's native 0-1000 scale)
    # before scoring, instead of counting them as misses. parse_failure_rate
    # and pointing_accuracy stay strict - this is a supplementary metric.
    pointing_accuracy_lenient: float
    explanation_score: float
    mean_latency_s: float


def score_model(key: str, data_dir: Path, results_dir: Path = RESULTS_DIR) -> ModelReport:
    examples: dict[str, Example] = {ex.id: ex for ex in load_subset(data_dir)}
    raw_path = results_dir / f"{key}.raw.jsonl"

    n = 0
    parse_failures = 0
    hits = 0
    lenient_hits = 0
    expl_scores: list[float] = []
    latencies: list[float] = []

    with raw_path.open() as f:
        for line in f:
            row = json.loads(line)
            ex = examples[row["id"]]
            n += 1
            latencies.append(row["latency_s"])

            parsed = parse_output(row["output"])
            if not parsed.ok:
                parse_failures += 1
            elif point_hit(parsed.x, parsed.y, ex.points):
                hits += 1

            if parsed.x is not None and parsed.y is not None:
                rescaled = rescale_to_percent(parsed.x, parsed.y)
                if rescaled is not None and point_hit(rescaled[0], rescaled[1], ex.points):
                    lenient_hits += 1

            if parsed.ok:
                expl_scores.append(lexical_overlap_score(parsed.explanation, ex.explanation).score)

    return ModelReport(
        key=key,
        n=n,
        parse_failure_rate=parse_failures / n if n else float("nan"),
        pointing_accuracy=hits / n if n else float("nan"),
        pointing_accuracy_lenient=lenient_hits / n if n else float("nan"),
        explanation_score=statistics.fmean(expl_scores) if expl_scores else 0.0,
        mean_latency_s=statistics.fmean(latencies) if latencies else float("nan"),
    )


def build_report(
    data_dir: Path, results_dir: Path = RESULTS_DIR, only_keys: list[str] | None = None
) -> list[ModelReport]:
    keys = only_keys if only_keys is not None else [spec.key for spec in MODEL_REGISTRY]
    reports = [score_model(key, data_dir, results_dir) for key in keys]
    reports.sort(key=lambda r: r.pointing_accuracy, reverse=True)
    return reports


def print_report(reports: list[ModelReport]) -> None:
    header = (
        f"{'model':<16}{'n':>5}{'point_acc':>11}{'point_acc*':>12}"
        f"{'parse_fail':>12}{'expl_score':>12}{'lat_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.key:<16}{r.n:>5}{r.pointing_accuracy:>11.2%}{r.pointing_accuracy_lenient:>12.2%}"
            f"{r.parse_failure_rate:>12.2%}{r.explanation_score:>12.2f}{r.mean_latency_s:>8.2f}"
        )
    print("\n* point_acc(lenient): rescales out-of-range points (e.g. a model's native 0-1000 scale) before scoring")
