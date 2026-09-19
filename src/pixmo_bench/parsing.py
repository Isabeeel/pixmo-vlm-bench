"""Step 4: parse model output into a (x, y) point + explanation, or a miss.

Tracks parse-failure rate as its own metric rather than folding it into
pointing accuracy: a model that never emits a point should show up as
"couldn't parse", not as "pointed at the wrong place".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_POINT_RE = re.compile(r"POINT\s*:\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", re.IGNORECASE)
# Non-greedy + lookahead so this stops before a following "POINT:" line
# instead of swallowing it - the prompt now asks for EXPLANATION before
# POINT, but the regex still works if a model emits the old POINT-first order.
_EXPLANATION_RE = re.compile(r"EXPLANATION\s*:\s*(.+?)(?=\s*POINT\s*:|\Z)", re.IGNORECASE | re.DOTALL)

# fallback for models that ignore the requested format but still emit
# something point-shaped, e.g. Molmo's native <point x=".." y="..">
_NATIVE_POINT_RE = re.compile(r'x\s*=\s*"?(-?\d+(?:\.\d+)?)"?\s+y\s*=\s*"?(-?\d+(?:\.\d+)?)"?')


@dataclass
class ParsedOutput:
    ok: bool
    x: float | None = None
    y: float | None = None
    explanation: str = ""
    raw: str = ""


def parse_output(raw_text: str) -> ParsedOutput:
    """`ok` is strict: requires the requested POINT:(x,y)/EXPLANATION: format
    *and* x,y in the requested 0-100 range. x/y are still populated when a
    point-shaped pattern was found but out of range (e.g. a model reverting
    to its own native 0-1 or 0-1000 grounding scale instead of the requested
    percentage scale) - see scoring.rescale_to_percent, which uses these to
    compute a separate, lenient pointing-accuracy metric without changing
    what counts as a strict parse failure.
    """
    text = raw_text.strip()

    point_match = _POINT_RE.search(text)
    if point_match is None:
        point_match = _NATIVE_POINT_RE.search(text)
    if point_match is None:
        return ParsedOutput(ok=False, raw=text)

    x, y = float(point_match.group(1)), float(point_match.group(2))
    in_range = 0 <= x <= 100 and 0 <= y <= 100

    expl_match = _EXPLANATION_RE.search(text)
    explanation = expl_match.group(1).strip() if expl_match else text[point_match.end():].strip()

    return ParsedOutput(ok=in_range, x=x, y=y, explanation=explanation, raw=text)
