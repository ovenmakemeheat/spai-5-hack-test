# Thai Math VQA Challenge Dataset Summary

Research snapshot: 2026-06-06

Local source inspected:

- `dataset/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen.zip`
- Extracted placeholder directory: `dataset/Individual-Test-Math-VQA-Challenge`

The requested long dataset path does not exist as an extracted directory in this workspace. The usable dataset is the zip archive above. The archive contains the actual CSV files and images.

## Task Overview

This is a Thai math visual question answering task. Each sample points to a math-problem image, and the model must return a free-form answer string. The answer space is not a fixed class list: labels include integers, decimals, Thai units, expressions, LaTeX fragments, equations, and a small number of non-answer markers.

The competition-style workflow is:

1. Read `train.csv` for labeled examples.
2. Read `test.csv` for unlabeled examples.
3. Predict one `answer` per test `id`.
4. Write a CSV with the same structure as `sample_submission.csv`.

## Archive Layout

```text
super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen.zip
├── train.csv
├── test.csv
├── sample_submission.csv
└── images/
    └── images/
        ├── 0.jpg
        ├── 1.jpg
        └── ... 699.jpg
```

Important path detail:

- CSV rows use paths like `images/0.jpg`.
- The archive stores files as `images/images/0.jpg`.
- When loading from the zip or an extracted archive, prepend the extra `images/` level or normalize paths explicitly.

## Files and Schema

| File | Rows | Columns | Purpose |
|---|---:|---|---|
| `train.csv` | 280 | `id`, `image_path`, `answer` | Labeled training data |
| `test.csv` | 420 | `id`, `image_path` | Test data to predict |
| `sample_submission.csv` | 420 | `id`, `answer` | Required submission format |
| `images/images/*.jpg` | 700 | image files | One JPEG per ID |

There are 700 unique IDs total. Train and test IDs do not overlap, and together they cover IDs `0..699` with no missing IDs.

## Example Rows

`train.csv` examples:

| id | image_path | answer |
|---:|---|---|
| 0 | `images/0.jpg` | `20 ตารางเซนติเมตร` |
| 102 | `images/102.jpg` | `76` |
| 104 | `images/104.jpg` | `45` |
| 105 | `images/105.jpg` | `192` |
| 113 | `images/113.jpg` | `<n/a>` |

`test.csv` examples:

| id | image_path |
|---:|---|
| 1 | `images/1.jpg` |
| 10 | `images/10.jpg` |
| 100 | `images/100.jpg` |
| 101 | `images/101.jpg` |

`sample_submission.csv` uses a constant placeholder answer of `2` for every row. It should be treated only as a formatting template, not as a meaningful baseline.

## Image Profile

All 700 images are RGB JPEGs.

| Property | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|
| Width | 506 | 1257 | 1456 | 1542 | 1841 |
| Height | 70 | 185 | 247 | 410 | 860 |
| File size, bytes | 13,112 | 35,064 | 54,102 | 82,630 | 203,691 |

The images are usually wide and relatively short, consistent with cropped problem statements, diagrams, or answer regions rather than full-page scans. A preprocessing pipeline should preserve small Thai text, mathematical symbols, line drawings, and geometry labels.

## Answer Distribution

There are 198 unique training answers across 280 labeled rows, so most labels are rare. The most common answers are simple integers:

| Answer | Count |
|---|---:|
| `1` | 13 |
| `2` | 12 |
| `3` | 11 |
| `4` | 10 |
| `5` | 9 |
| `8` | 5 |
| `36` | 4 |
| `6` | 3 |
| `14` | 3 |
| `45` | 3 |
| `17` | 3 |
| `80 องศา` | 3 |

Answer type profile:

| Type | Count | Notes |
|---|---:|---|
| Pure integers | 164 | Most common format |
| Decimal-only answers | 5 | Example: `3.75` style |
| LaTeX/math markup present | 28 | Examples include `\frac`, `\sqrt`, `$...$` |
| Non-simple numeric strings | 111 | Includes Thai units, equations, expressions, text |
| Literal `<n/a>` | 2 | Needs explicit handling |

Representative non-numeric labels:

- `20 ตารางเซนติเมตร`
- `1432 จำนวน`
- `15 หน่วย`
- `$7 + 3\sqrt{5}$`
- `l(l+k)`
- `x=70,y=24`
- `80 องศา`
- `จุดนี้ไม่สามารถจะอยู่ในจตุภาคที่ 3 ได้`

## Modeling Implications

This should be treated as open-answer generation or answer extraction, not ordinary image classification.

Main difficulties:

- The training set is very small for fine-tuning a vision-language model.
- The output format matters. `20`, `20 หน่วย`, and `20 ตารางเซนติเมตร` may be considered different strings depending on the competition evaluator.
- Mathematical expressions may need LaTeX-compatible formatting.
- Thai units and Thai explanatory text appear in answers.
- Some samples may require OCR, geometry reasoning, counting, algebra, or equation solving.

## Recommended Baselines

### 1. Strong zero-shot or few-shot VLM baseline

Use a frontier multimodal reasoning model or a strong open VLM. Prompt it to:

- read all visible Thai text and math notation,
- solve the problem,
- output only the final answer string,
- preserve Thai units if present or implied,
- preserve LaTeX-style fractions/radicals when exact form is required.

This is the fastest useful baseline because there are only 420 test samples.

### 2. OCR-assisted VLM pipeline

For each image:

1. Create a high-resolution normalized image.
2. Run OCR for Thai/math text when available.
3. Feed both image and OCR text to the VLM.
4. Ask for concise final answer only.
5. Normalize the predicted string.

This helps when small text is hard for the VLM, but OCR can also introduce errors in mathematical notation. Keep the image in the prompt even when OCR is used.

### 3. Self-consistency or ensemble

For each test image:

1. Query the same VLM with 2-5 prompt variants.
2. Optionally query a second VLM.
3. Normalize candidate answers.
4. Majority vote exact matches.
5. For disagreements, ask a verifier model to select the most plausible answer.

This is practical because the test set is only 420 rows.

### 4. Fine-tuning, only if enough extra data is added

Fine-tuning on 280 rows alone is high risk. If fine-tuning is attempted, use public visual math datasets for pretraining/adaptation, then optionally fine-tune lightly on the local train set. Avoid overfitting answer string style.

## Preprocessing Checklist

- Do not downscale aggressively. Preserve text, thin diagram lines, superscripts, subscripts, and fraction bars.
- Pad to a model-friendly canvas instead of cropping blindly.
- Use contrast enhancement carefully; over-thresholding can erase geometry labels and light grid lines.
- Keep aspect ratio.
- For very wide images, tile or use high-resolution input if the VLM supports it.
- Normalize answer strings after prediction:
  - trim whitespace,
  - normalize repeated spaces,
  - standardize Thai unit spacing,
  - preserve exact math symbols where possible,
  - do not strip units unless the metric is known to ignore units.

## Validation Strategy

Because only 280 labels are available, use repeated splits rather than trusting one validation split:

- Stratify loosely by answer type, not exact answer.
- Keep a held-out set with integer, Thai-unit, LaTeX, and text answers.
- Measure exact match after the same normalization used for submissions.
- Track error categories manually: OCR failure, wrong reasoning, unit mismatch, formatting mismatch, ambiguous/no-answer case.

## Output Format

The submission must contain:

```csv
id,answer
1,2
10,2
...
```

Preserve the row order from `sample_submission.csv` unless the competition explicitly allows arbitrary ordering.

