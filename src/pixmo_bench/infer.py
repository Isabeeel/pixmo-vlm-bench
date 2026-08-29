"""Step 3: run zero-shot inference for one model over the prepared subset.

Not runnable in this sandbox yet (no GPU / torch installed - see
requirements.txt). Written now so the only work left once a GPU is available
is `pip install -r requirements.txt` and `python scripts/run_pipeline.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .data import Example, load_subset
from .models import ModelSpec
from .prompts import build_prompt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def run_model(
    spec: ModelSpec,
    data_dir: Path,
    out_path: Path | None = None,
    limit: int | None = None,
    torch_dtype: "torch.dtype | None" = None,
) -> Path:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    examples: list[Example] = load_subset(data_dir)
    if limit is not None:
        examples = examples[:limit]
    out_path = out_path or RESULTS_DIR / f"{spec.key}.raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.repo_id,
        trust_remote_code=spec.trust_remote_code,
        torch_dtype=torch_dtype or torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    with out_path.open("w") as f:
        for ex in examples:
            image = Image.open(Path(data_dir) / ex.image_path).convert("RGB")
            prompt = build_prompt(ex.question)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=image, text=chat_text, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            latency_s = time.time() - t0

            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]

            f.write(json.dumps({"id": ex.id, "output": text, "latency_s": latency_s}) + "\n")

    return out_path
