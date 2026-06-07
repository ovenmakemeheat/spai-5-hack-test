# Math VQA Pipeline

Primary implementation:

- `notebook/math_vqa/math_vqa_full_pipeline.ipynb`

The notebook is self-contained for this project: it does not import `src.math_vqa` or any local
pipeline module. Gemma, Qwen, and prior predictor classes are defined inside the notebook.

Gemma defaults to `google/gemma-3-4b-it`. Model access may require accepting the license on
Hugging Face and authenticating with `HF_TOKEN`.

Fast smoke test without loading a VLM: set `BACKEND = "prior"` and `LIMIT = 3` in the notebook
config cell, then run all cells.

Default outputs:

- `outputs/math_vqa/images_preprocessed/`
- `outputs/math_vqa/valid_predictions_<run-tag>.jsonl`
- `outputs/math_vqa/valid_metrics_<run-tag>.json`
- `outputs/math_vqa/test_predictions_<run-tag>.jsonl`
- `submission_math_vqa.csv`

The JSONL prediction caches are resumable. Their filenames include backend, model, prompt variants,
and preprocessing settings so prior-backend smoke tests do not collide with Gemma or Qwen runs.
