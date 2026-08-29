# PixMo-Point VLM Benchmark

Zero-shot pointing + explanation-quality eval for small (<=5B) VLMs on
`allenai/pixmo-point-explanations`. See the pipeline plan artifact for the
7-step design; this repo implements it.

## Status

- **Step 1 (data subset) - done.** `data/subset/manifest.jsonl` + `data/subset/images/`
  (150 examples, seed=0, gitignored but fully reproducible - see below).
- **Steps 2-7 (model loading, inference, scoring, report) - code written, not
  yet run.** This sandbox has no GPU and no `torch` installed
  (`nvidia-smi` fails, `pip list` has no torch/transformers). Once a GPU box
  is available: `pip install -r requirements.txt && python scripts/run_pipeline.py`.

## Layout

```
src/pixmo_bench/
  data.py       step 1 - pull + cache the subset from HF
  prompts.py    shared zero-shot prompt (same text for all 5 models)
  models.py     the 5 candidate model specs (repo ids, notes)
  infer.py      step 3 - run one model over the subset (needs torch)
  parsing.py    step 4 - extract POINT/EXPLANATION, track parse failures
  scoring.py    steps 5-6 - pointing hit test + explanation score
  report.py     step 7 - per-model comparison table
scripts/
  prepare_data.py   run step 1 (works now)
  run_pipeline.py   run steps 2-7 (needs GPU)
```

## Design notes / decisions made while implementing

- **Single-point examples only.** The source dataset can have multiple
  point-references per response; subset is filtered to exactly one so a
  single model turn maps to a single scorable target.
- **Common output format instead of native grounding syntax.** Rather than
  parsing each model's own pointing convention (Molmo's `<point x y>`,
  whatever InternVL/Qwen3-VL use natively), every model is asked to answer as
  `POINT: (x, y)` / `EXPLANATION: ...`. This is what makes parse-failure rate
  a meaningful, comparable metric instead of 5 different parsers with 5
  different failure modes.
- **Pointing hit = distance under threshold**, not exact match or mask
  containment - this split only has point annotations, no segmentation
  masks, so `scoring.DEFAULT_HIT_RADIUS` (7.0 on the 0-100 scale) is a
  stand-in for "inside the region." Tune this once real outputs exist.
- **Explanation scoring is a placeholder** (`lexical_overlap_score`, token
  overlap, no dependencies). The plan calls for judging logical consistency
  with the image, which really wants an LLM judge - `scoring.llm_judge_score`
  is stubbed for that, gated on `ANTHROPIC_API_KEY`.

## Models

| key | repo id | params |
|---|---|---|
| molmo2-4b (reference) | allenai/Molmo2-4B | 4B |
| internvl3.5-2b | OpenGVLab/InternVL3_5-2B-Instruct | 2B |
| smolvlm2-2.2b | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B |
| qwen3-vl-2b | Qwen/Qwen3-VL-2B-Instruct | 2B |
| qwen3-vl-4b | Qwen/Qwen3-VL-4B-Instruct | 4B |
