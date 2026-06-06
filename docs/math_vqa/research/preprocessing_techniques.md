# Math VQA Preprocessing Techniques

Research snapshot: 2026-06-06

Task focus: improve answer accuracy for Thai math VQA images without destroying text, mathematical notation, or diagram geometry.

## Core Principle

For modern VLMs, preprocessing should make the image easier to read without changing its mathematical content. Aggressive thresholding, cropping, compression, or resizing can hurt more than help because math images often contain thin lines, small labels, superscripts, decimal points, and fraction bars.

## Recommended Pipeline

### 1. Path Normalization

The local CSV path is `images/{id}.jpg`, while the archive path is `images/images/{id}.jpg`.

Normalize this once in the data loader:

```python
zip_path = "images/" + row["image_path"]
```

or after extraction:

```python
image_path = root / "images" / row["image_path"]
```

depending on the extraction layout.

### 2. Preserve Native Resolution

Most local images are wide problem crops:

- median width: 1456 px
- median height: 247 px
- maximum width: 1841 px
- maximum height: 860 px

Avoid resizing everything to a square 224/336 image for VLM inference. If a model requires a fixed input size internally, let the model processor handle it, or use high-resolution settings where supported.

### 3. Pad, Do Not Blind-Crop

Use white or light-gray padding to fit model aspect-ratio requirements. Blind cropping is risky because:

- problem text may sit near borders,
- geometry labels may be outside the main diagram,
- answer choices or units may be at the far right,
- Thai tone marks and small math symbols are easy to cut off.

### 4. Contrast and Sharpness

Use mild enhancement only:

- auto-orient image,
- convert to RGB,
- optional mild contrast enhancement,
- optional unsharp mask,
- optional grayscale copy for OCR.

Avoid heavy binarization as the default. It can remove light grid lines, dashed construction lines, or gray answer-choice marks.

### 5. OCR as Auxiliary Context

Run OCR when available, but keep the original image in the VLM prompt. OCR is helpful for Thai text and printed numbers, but it often struggles with:

- fractions,
- radicals,
- superscripts/subscripts,
- geometry labels,
- Thai mathematical wording,
- mixed Thai/LaTeX-like notation.

Best practice:

1. Generate OCR text.
2. Pass both image and OCR text to the VLM.
3. Tell the VLM that OCR may contain mistakes.
4. Ask it to rely on the image when OCR and image disagree.

### 6. Image Tiling for Very Wide Inputs

For very wide samples, create auxiliary crops:

- full image,
- left half,
- center crop,
- right half,
- diagram-focused crop if detected,
- OCR crop if text is very small.

Then either:

- send multiple images to a VLM that supports multi-image input, or
- solve each crop separately and verify using the full image.

This is especially useful for wide textbook-style images where question text, diagram, and choices are spatially separated.

### 7. Candidate Generation and Verification

Use multiple candidates rather than a single generation:

1. Prompt variant A: direct solve.
2. Prompt variant B: read text first, then solve.
3. Prompt variant C: detect relevant diagram labels, then solve.
4. Optional OCR-assisted prompt.
5. Optional crop-assisted prompt.

Normalize and vote. If candidates disagree, ask a verifier VLM to choose one.

## Answer Normalization

The dataset contains diverse answer formats. Normalize enough to remove accidental formatting noise, but not so much that the answer changes meaning.

Recommended safe normalization:

- Strip leading/trailing whitespace.
- Collapse repeated spaces.
- Convert full-width digits to ASCII digits if produced.
- Normalize comma spacing in `x=70,y=24` style answers.
- Remove markdown code fences or surrounding quotes.
- Preserve Thai units.
- Preserve exact fractions, radicals, pi, and LaTeX when generated.

Risky normalization:

- Removing units.
- Converting exact radicals to decimals.
- Converting fractions to decimals.
- Removing `$...$` if the evaluator expects LaTeX-like strings.
- Translating Thai units to English.

## Validation Checks

Use the 280 training samples to test preprocessing choices:

- exact-match accuracy after normalization,
- accuracy by answer type: integer, decimal, Thai unit, LaTeX, text,
- OCR-only vs image-only vs image+OCR,
- full image vs crop/tiling,
- model self-consistency gain.

Track errors manually. Useful tags:

- `ocr_missed_text`
- `diagram_misread`
- `wrong_formula`
- `arithmetic_error`
- `unit_missing`
- `format_mismatch`
- `ambiguous_or_na`

## Technique Ranking for This Dataset

| Rank | Technique | Expected impact | Cost | Notes |
|---:|---|---|---|---|
| 1 | High-resolution image input | High | Low | Most important for small Thai/math text |
| 2 | Final-answer-only prompt | High | Low | Prevents explanation leakage into CSV |
| 3 | Self-consistency / multi-prompt voting | High | Medium | Test set is only 420 rows, so this is practical |
| 4 | OCR-assisted prompting | Medium-high | Medium | Useful for text-heavy images, risky for formulas |
| 5 | Crop/tiling for wide images | Medium | Medium | Helps if model downscales wide images too much |
| 6 | Mild contrast/sharpening | Medium | Low | Validate; do not overprocess |
| 7 | External symbolic solver | Medium | High | Useful only when OCR/parsing is reliable |
| 8 | Fine-tuning on local train only | Low-medium | Medium-high | 280 rows is too small for robust fine-tuning |

## Research Basis

- MathVista shows visual math requires fine-grained visual understanding and compositional reasoning: https://mathvista.github.io/
- Qwen2.5-VL emphasizes OCR, layout, chart, and diagram understanding as key capabilities: https://qwenlm.github.io/blog/qwen2.5-vl/
- TextVQA/OCR-VQA research motivates OCR-aware VQA pipelines for images where the answer depends on reading text:
  - TextVQA: https://textvqa.org/assets/paper/TextVQA.pdf
  - OCR-VQA: https://anandmishra22.github.io/files/mishra-OCR-VQA.pdf
- Program-of-thought and chain-of-thought work motivate separating reasoning from final answer formatting, though the submitted output should contain only the answer:
  - Program of Thoughts: https://arxiv.org/abs/2211.12588
  - Chain-of-Thought Prompting: https://arxiv.org/abs/2201.11903

