# Visual Math VQA Paper and Benchmark Ranking

Research snapshot: 2026-06-06

Task focus: papers and benchmark resources most relevant to a small Thai math VQA competition dataset with image-only problem statements and free-form answers.

## Ranking Logic

This ranking prioritizes usefulness for building a winning solution:

1. Direct relevance to visual mathematical reasoning.
2. Evidence of strong benchmark performance.
3. Practical technique transfer to a small dataset.
4. Availability of benchmark/task framing for evaluation design.

## Ranked Papers and Resources

| Rank | Paper / resource | Why it matters | Transfer to this project |
|---:|---|---|---|
| 1 | MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts | Core benchmark for visual math reasoning. It combines 6,141 examples from 28 existing multimodal math datasets plus 3 new datasets. | Use MathVista taxonomy to categorize errors: figure QA, geometry, math word problems, textbook QA, visual QA; arithmetic, algebra, geometry, logic, numeric, scientific, statistical reasoning. |
| 2 | MathVista leaderboard | Practical source for current model ranking on visual math reasoning. The page tracks testmini and private-test performance. | Use as the first model shortlist. As of the searched snapshot, DreamPRM (`o4-mini`) leads testmini, while older private-test results list InternVL variants at the top. |
| 3 | SoTA with Less / ThinkLite-VL | Shows that reinforcement fine-tuning on carefully selected hard visual-reasoning samples can produce strong MathVista results with far less data. Reports 75.1 MathVista for 7B and 79.7 for 72B. | Use the insight without full training: select hard examples, generate multiple reasoning attempts, and use verifier/reranker logic. If weights are available, test directly. |
| 4 | Qwen2.5-VL technical report and release | Strong open VLM foundation with explicit OCR, chart, layout, diagram, grounding, and structured-output capabilities. | Best open-source starting point for local inference and for any fine-tuning/reasoning adaptation. |
| 5 | Seed1.5-VL technical report | Reports broad VLM SOTA across many benchmarks and strong visual puzzle/reasoning performance. | Strong API/model candidate for ensemble or verification if accessible. |
| 6 | Open Vision Reasoner | Demonstrates that linguistic cold start plus multimodal RL can improve visual reasoning on MathVision and MathVerse. | Good method direction for smaller open models: train/reason with visual reflection and verifier rewards rather than naive SFT only. |
| 7 | MathVision / MATH-Vision dataset | Harder visual math benchmark. Useful because many models that look strong on MathVista still struggle on MATH-Vision. | Use for external validation or extra few-shot examples if allowed. Good source of geometry/counting style tasks. |
| 8 | We-Math | Decomposes visual math into knowledge concepts and sub-problems; evaluates whether models generalize or memorize. | Use its idea for prompting: identify required knowledge first, then solve; useful for manual error analysis. |
| 9 | MathVerse | Diagnoses whether models actually use the visual information by varying text/vision dependence. | Useful for deciding when OCR-only is insufficient and when image evidence must dominate. |
| 10 | TextVQA / OCR-VQA | Older but still important for reading text in images. | Supports OCR-aware preprocessing, especially for Thai text and printed problem statements. |
| 11 | Program-of-Thoughts and Chain-of-Thought prompting | Foundational prompting ideas for numerical reasoning. | Use reasoning internally, but strip final output to only the answer string for submission. |

## Key Takeaways

### 1. Math VQA is not just OCR

MathVista and MathVerse both emphasize that visual math needs actual image-grounded reasoning. OCR can recover text, but it cannot reliably reconstruct geometry, diagrams, charts, spatial relations, or visual patterns.

### 2. Model choice matters more than fine-tuning on 280 rows

The local labeled set is too small to train a competitive VLM from scratch. The strongest practical approach is inference-time use of SOTA VLMs with:

- high-resolution inputs,
- prompt variants,
- OCR as auxiliary context,
- answer normalization,
- verifier/reranking.

### 3. Reasoning-tuned VLMs are the current direction

ThinkLite-VL, Open Vision Reasoner, and related 2025 work point in the same direction: base VLMs improve substantially when trained or prompted for explicit visual reasoning, verification, and hard-sample selection.

### 4. Benchmarks disagree

A model strong on MathVista may not be the best on MathVision or MathVerse. For this Thai challenge, run a small validation bake-off on the 280 labeled rows rather than trusting any single leaderboard.

## Implementation Plan Derived from the Literature

1. Build an evaluation harness on `train.csv`.
2. Test 2-4 candidate models on the same 280 rows.
3. Use exact-match plus normalized exact-match.
4. Break scores down by answer type: integer, Thai unit, LaTeX, text, `<n/a>`.
5. Add OCR context and compare against image-only.
6. Add self-consistency and verifier selection.
7. Freeze the best prompt/model combination and generate `sample_submission.csv`-ordered predictions.

## Sources

- MathVista project and leaderboard: https://mathvista.github.io/
- MathVista paper: https://arxiv.org/abs/2310.02255
- ThinkLite-VL / SoTA with Less: https://arxiv.org/abs/2504.07934
- Qwen2.5-VL release: https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923
- Seed1.5-VL technical report: https://arxiv.org/abs/2505.07062
- Open Vision Reasoner: https://arxiv.org/abs/2507.05255
- We-Math: https://we-math.github.io/
- MathVision dataset: https://huggingface.co/datasets/MathLLMs/MathVision
- TextVQA paper: https://textvqa.org/assets/paper/TextVQA.pdf
- OCR-VQA paper: https://anandmishra22.github.io/files/mishra-OCR-VQA.pdf
- Program of Thoughts: https://arxiv.org/abs/2211.12588
- Chain-of-Thought Prompting: https://arxiv.org/abs/2201.11903

