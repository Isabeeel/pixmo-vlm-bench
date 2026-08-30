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
    examples: list[Example] = load_subset(data_dir)
    if limit is not None:
        examples = examples[:limit]
    out_path = out_path or RESULTS_DIR / f"{spec.key}.raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if spec.api == "internvl_chat":
        _run_internvl_chat(spec, examples, data_dir, out_path)
        return out_path

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

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


# --- InternVL-Chat custom API path ---------------------------------------
#
# OpenGVLab's InternVL3.5 repo still ships the legacy "InternVL-Chat" custom
# modeling class (config class InternVLChatConfig) rather than the natively
# transformers-integrated InternVLConfig/InternVLForConditionalGeneration, so
# it isn't loadable through AutoModelForImageTextToText and doesn't take the
# usual AutoProcessor + generate() calling convention. Loading code below
# follows the model card's documented quick-start exactly (AutoModel +
# model.chat(), with the dynamic image-tiling preprocessing it expects).

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            (i % cols + 1) * image_size,
            (i // cols + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_load_image(image_path: Path, input_size: int = 448, max_num: int = 12):
    import torch

    image = Image.open(image_path).convert("RGB")
    transform = _internvl_build_transform(input_size)
    tiles = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def _run_internvl_chat(spec: ModelSpec, examples: list[Example], data_dir: Path, out_path: Path) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        spec.repo_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=200, do_sample=False)

    with out_path.open("w") as f:
        for ex in examples:
            pixel_values = _internvl_load_image(Path(data_dir) / ex.image_path).to(torch.bfloat16).to(model.device)
            question = "<image>\n" + build_prompt(ex.question)

            t0 = time.time()
            with torch.inference_mode():
                text = model.chat(tokenizer, pixel_values, question, generation_config)
            latency_s = time.time() - t0

            f.write(json.dumps({"id": ex.id, "output": text, "latency_s": latency_s}) + "\n")
