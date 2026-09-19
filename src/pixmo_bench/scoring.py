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
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HIT_RADIUS = 7.0  # % of image diagonal-normalized scale; see note above


def point_hit(pred_x: float, pred_y: float, gt_points: list[list[float]], radius: float = DEFAULT_HIT_RADIUS) -> bool:
    return any(math.hypot(pred_x - gx, pred_y - gy) <= radius for gx, gy in gt_points)


def rescale_to_percent(x: float, y: float) -> tuple[float, float] | None:
    """Best-effort fix-up for models that ignore the requested 0-100 scale
    and answer in their own native convention instead - observed with
    Qwen3-VL, which mostly reverts to its training-time 0-1000 grounding
    scale (occasionally 0-1) despite the prompt asking for percentages.

    Only used for the supplementary "lenient" pointing-accuracy metric in
    report.py; parsing.ParsedOutput.ok / parse_failure_rate are unaffected,
    so "didn't follow the requested format" stays visible as its own signal.
    """
    if x <= 1 and y <= 1:
        return x * 100, y * 100  # 0-1 normalized
    if x <= 100 and y <= 100:
        return x, y  # already matches the requested scale
    if x <= 1000 and y <= 1000:
        return x / 10, y / 10  # native 0-1000 grounding scale (e.g. Qwen-VL)
    return None


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


_JUDGE_MODEL = "claude-sonnet-5"

_JUDGE_PROMPT = """You are grading a vision-language model's explanation for why it \
pointed at a location in the attached image, in response to the question below.

Question: {question}
Reference explanation (written by a human annotator who saw the image): {ground_truth}
Model's explanation: {prediction}

Look at the image yourself and judge whether the model's explanation is consistent \
with it: does it correctly identify the same object/region the question asks about, \
with reasoning that actually holds up against the image? Ignore wording differences \
from the reference - judge the model's explanation against the image directly, using \
the reference only as a hint about what the correct answer is.

Respond with ONLY a single number from 0 to 1 (e.g. "0.8"):
1.0 = correct object/region, reasoning holds up against the image
0.5 = right general area but reasoning is vague, unsupported, or partly wrong
0.0 = wrong object/region, or reasoning contradicts the image"""


def llm_judge_score(
    prediction: str, ground_truth: str, question: str, image_path: "Path | None" = None
) -> ExplanationScore:
    """Judge explanation quality with Claude instead of lexical_overlap_score.

    Requires ANTHROPIC_API_KEY. Claude is itself a VLM, so when `image_path` is
    given the judge looks at the actual image rather than only comparing two
    strings - a real check of "is the reasoning consistent with the image",
    which lexical_overlap_score can't do at all.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; fall back to lexical_overlap_score")

    import base64

    import anthropic
    from PIL import Image

    client = anthropic.Anthropic()
    text = _JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, prediction=prediction)

    content: list[dict] = []
    if image_path is not None:
        image_path = Path(image_path)
        # data.py saves every downloaded image as "<id>.jpg" regardless of its
        # actual format, so the file extension can't be trusted - some are
        # really PNG/BMP/WEBP/etc. and Claude's API rejects a mismatched
        # media_type (or a format it doesn't support at all, e.g. BMP).
        # Detect the real format by opening the file, and re-encode to PNG
        # for anything outside the API's supported set instead of guessing.
        with Image.open(image_path) as im:
            fmt = im.format
            if fmt in ("JPEG", "PNG", "GIF", "WEBP"):
                media_type = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}[fmt]
                image_bytes = image_path.read_bytes()
            else:
                import io

                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="PNG")
                media_type = "image/png"
                image_bytes = buf.getvalue()
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                },
            }
        )
    content.append({"type": "text", "text": text})

    response = client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    # Claude can return a leading "thinking" block before the actual text
    # block, so pick out the text block by type rather than assuming index 0.
    reply = next((block.text for block in response.content if block.type == "text"), "").strip()
    match = re.search(r"(\d*\.?\d+)", reply)
    score = max(0.0, min(1.0, float(match.group(1)))) if match else 0.0
    return ExplanationScore(score=score, method=f"llm_judge:{_JUDGE_MODEL}")
