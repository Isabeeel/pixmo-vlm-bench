"""Shared zero-shot prompt template (step 2/3).

All five models get the exact same instruction and the exact same required
output format, rather than each model's native grounding syntax. This keeps
scoring uniform across models with very different pointing conventions
(Molmo's native <point> tags vs. models with no pointing format at all), and
turns "didn't use its native format" into what step 4 measures on purpose:
parse-failure rate as a real, comparable metric instead of noise.

Two changes from the original version, both aimed at parse_failure_rate
(mainly Qwen3-VL's 89%, which mostly reverts to its native 0-1000 grounding
scale instead of the requested 0-100 percentage):
- EXPLANATION now comes before POINT in the required output, so the model's
  own generated reasoning tokens are already on the page before it commits to
  a coordinate (the old order asked for reasoning-then-pointing in prose, but
  still made POINT the first line the model had to autoregressively produce).
- Added a one-shot example pinning down the 0-100 scale concretely, instead
  of only describing it in words. parsing._EXPLANATION_RE was updated to stop
  at a following "POINT:" line so it doesn't swallow it now that POINT comes
  second.
"""

PROMPT_TEMPLATE = """Look at the image and answer the question below.

Question: {question}

Reason about where in the image the answer is located, then give a single
point. Respond in exactly this format, with nothing before or after it:

EXPLANATION: <one or two sentences>
POINT: (x, y)

x and y are percentages of image width/height, 0-100 (not pixels, not
0-1000), x from the left edge, y from the top edge.

Example:
Question: Where is the person wearing a red hat?
EXPLANATION: The person wearing a red hat is standing near the center-left of the image, next to the food stand.
POINT: (38, 52)"""


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())
