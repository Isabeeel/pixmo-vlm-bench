"""Steps 5-6: pointing accuracy and explanation quality.

Pointing: this dataset split has annotated points, not segmentation masks, so
"inside the region" is approximated as distance-under-threshold against every
ground-truth alternative point (a referenced object sometimes has more than
one valid point, e.g. "Muslims" appears once). Threshold is in the same
0-100 normalized units as the points themselves.

Explanation quality: a real judgment call ("is the reasoning consistent with
the image") needs either a human or an LLM judge. `lexical_overlap_score` is
a cheap, dependency-free placeholder so the pipeline runs end to end; swap in
`llm_judge_score` (uses the Anthropic API) once an API key is available.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_HIT_RADIUS = 7.0  # % of image diagonal-normalized scale; see note above


def point_hit(pred_x: float, pred_y: float, gt_points: list[list[float]], radius: float = DEFAULT_HIT_RADIUS) -> bool:
    return any(math.hypot(pred_x - gx, pred_y - gy) <= radius for gx, gy in gt_points)


@dataclass
class ExplanationScore:
    score: float  # 0-1
    method: str


def lexical_overlap_score(prediction: str, ground_truth: str) -> ExplanationScore:
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return ExplanationScore(score=0.0, method="lexical_overlap")
    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
    return ExplanationScore(score=min(1.0, overlap), method="lexical_overlap")


def llm_judge_score(prediction: str, ground_truth: str, question: str) -> ExplanationScore:
    """Optional: replace lexical_overlap_score with a real judge call.

    Requires ANTHROPIC_API_KEY. Left unimplemented here on purpose - wire up
    the `anthropic` client once this pipeline is unblocked and that dependency
    is worth adding.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; fall back to lexical_overlap_score")
    raise NotImplementedError("wire up anthropic.Anthropic().messages.create(...) here")
