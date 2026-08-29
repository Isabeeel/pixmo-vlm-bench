"""Shared zero-shot prompt template (step 2/3).

All five models get the exact same instruction and the exact same required
output format, rather than each model's native grounding syntax. This keeps
scoring uniform across models with very different pointing conventions
(Molmo's native <point> tags vs. models with no pointing format at all), and
turns "didn't use its native format" into what step 4 measures on purpose:
parse-failure rate as a real, comparable metric instead of noise.
"""

PROMPT_TEMPLATE = """Look at the image and answer the question below.

Question: {question}

First find the single location in the image that answers the question, then
explain your reasoning. Respond in exactly this format, with nothing before
or after it:

POINT: (x, y)
EXPLANATION: <one or two sentences>

x and y are percentages of image width/height, 0-100, x from the left edge,
y from the top edge."""


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())
