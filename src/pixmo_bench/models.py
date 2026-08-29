"""Step 2: candidate model registry.

Zero-shot, no fine-tuning. Every model is loaded through
transformers.AutoModelForImageTextToText / AutoProcessor with the shared
prompt from prompts.py, so adding a model is just adding a ModelSpec entry.

Requires torch + transformers + accelerate, which this sandbox does not have
yet (no GPU). See infer.py for where these specs get used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    params_b: float
    is_reference: bool = False
    trust_remote_code: bool = False
    notes: str = ""


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="molmo2-4b",
        repo_id="allenai/Molmo2-4B",
        params_b=4.0,
        is_reference=True,
        trust_remote_code=True,
        notes="reference baseline, direct successor of the model PixMo was built for",
    ),
    ModelSpec(
        key="internvl3.5-2b",
        repo_id="OpenGVLab/InternVL3_5-2B-Instruct",
        params_b=2.0,
        trust_remote_code=True,
    ),
    ModelSpec(
        key="smolvlm2-2.2b",
        repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        params_b=2.2,
        notes="no native pointing format; expect a higher parse-failure rate",
    ),
    ModelSpec(
        key="qwen3-vl-2b",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        params_b=2.0,
    ),
    ModelSpec(
        key="qwen3-vl-4b",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        params_b=4.0,
    ),
]


def get_model(key: str) -> ModelSpec:
    for spec in MODEL_REGISTRY:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown model key: {key}")


# Not one of the 5 benchmark candidates - used by scripts/smoke_test.py to
# validate the pipeline end to end (prompt -> generate -> parse -> score) on
# CPU, on hardware too small to run any of the real candidates.
SMOKE_MODEL = ModelSpec(
    key="smolvlm-256m",
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    params_b=0.256,
    notes="CPU smoke-test model only, not a benchmark candidate",
)
