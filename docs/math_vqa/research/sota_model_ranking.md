# Math VQA SOTA Model Ranking

Research snapshot: 2026-06-06

Task focus: Thai math visual question answering from cropped problem images. The target dataset has only 280 labeled training rows and 420 test rows, so practical SOTA for this project means strong inference-time visual reasoning, robust OCR/math notation handling, and answer-format control. Full fine-tuning is secondary unless extra public data is used.

## Ranking Caveat

There is no single universal SOTA model for "math VQA." Benchmarks differ:

- MathVista mixes figures, geometry, math word problems, textbook QA, and visual QA.
- MathVision/MATH-Vision focuses on harder visual math problems.
- MathVerse stresses whether models actually use the visual modality.
- We-Math decomposes visual math into knowledge concepts and sub-problems.

For this local Thai challenge, ranking should be interpreted as a deployment shortlist, not a universal truth. Thai OCR, exact answer formatting, API availability, GPU memory, and cost may change the practical winner.

## Short Recommendation

Use this order of attack:

1. If API access is allowed, start with `o3`/`o4-mini`-class or equivalent frontier visual reasoning models and run a self-consistency ensemble.
2. For open-source/local work, start with Qwen2.5-VL-72B-Instruct if hardware allows; otherwise Qwen2.5-VL-7B-Instruct plus a reasoning-tuned derivative.
3. Add OCR-assisted prompting only as auxiliary context; keep the original image in every prompt.
4. Use a verifier pass to choose among candidate answers and enforce final-answer-only formatting.

## Ranked Model Families

| Rank | Model family | Why it ranks here | Best use in this project | Risks |
|---:|---|---|---|---|
| 1 | Frontier proprietary visual reasoning models: OpenAI `o3` / `o4-mini` class | MathVista search results and official OpenAI visual reasoning material indicate very strong visual reasoning; MathVista currently lists DreamPRM using `o4-mini` at the top of testmini. | Highest-accuracy inference baseline and verifier. Use multiple prompt variants and exact-answer extraction. | API cost, rate limits, hidden model changes, and possible competition restrictions. |
| 2 | DreamPRM / PRM-guided reasoning with `o4-mini` | MathVista leaderboard snapshot shows DreamPRM (`o4-mini`) at 85.2 on MathVista testmini as of 2025-06-04. | If reproducible, use process-reward or verifier-style reranking over multiple solutions. | May not be publicly reproducible end to end; not a simple local model. |
| 3 | ThinkLite-VL-72B | The paper reports 79.7 MathVista and strong average gains through MCTS-guided sample selection plus reinforcement fine-tuning. | Best research direction for an open visual-reasoning model if weights are accessible. | Large 72B model; training recipe is expensive; availability must be verified. |
| 4 | ThinkLite-VL-7B | The paper reports 75.1 MathVista, surpassing larger models such as GPT-4o, o1, and Qwen2.5-VL-72B on that benchmark. | Strong local-ish model class when 72B is impractical; good target architecture for reproduction. | Still requires VLM inference resources; reported SOTA may depend on exact benchmark protocol. |
| 5 | Seed1.5-VL / Doubao thinking vision | Technical report claims SOTA on 38 of 60 public VLM benchmarks and strong reasoning/visual puzzle capability. MathVision pages reported Seed1.5-VL reaching 68.7 on MATH-Vision. | Strong API candidate where available; useful verifier for hard visual puzzles and geometry. | Access may be through Volcano Engine; exact version and availability matter. |
| 6 | Qwen2.5-VL-72B-Instruct | Official Qwen release emphasizes text, chart, layout, diagram, structured-output, localization, and OCR improvements. It is a strong open baseline and foundation for many reasoning-tuned models. | Best open foundation model to test first; supports exact-answer generation and OCR-heavy math images. | Heavy memory footprint; base model may underperform reasoning-tuned descendants on complex math. |
| 7 | Open Vision Reasoner (OVR) on Qwen2.5-VL-7B | NeurIPS 2025 work reports 51.8 on MathVision and 54.6 on MathVerse using linguistic cold-start plus multimodal RL. | Good open 7B reasoning direction for MathVision/MathVerse-style problems. | Its reported strength is MathVision/MathVerse, not necessarily Thai OCR or MathVista. |
| 8 | Qwen2.5-VL-7B-Instruct | Official release says the 7B model outperforms GPT-4o-mini on multiple tasks and has strong text/layout understanding. | Practical local baseline on a single high-memory GPU or quantized setup. | Lower raw reasoning than 72B/frontier models; may need self-consistency and OCR support. |
| 9 | InternVL2/InternVL2.5/InternVL3 family | InternVL variants appear near the top of the official MathVista test leaderboard and are common strong open VLM baselines. | Ensemble member; compare against Qwen for geometry and chart samples. | Exact best version changes quickly; Thai OCR quality must be tested. |
| 10 | General OCR + LLM pipeline | External OCR plus a strong text-only math model can solve OCR-dominant samples. | Fallback when image mostly contains text and formulas; useful for verifier context. | Loses diagram geometry and can corrupt math notation; should not replace image-conditioned reasoning. |

## Practical Prompt Template

Use a final-answer-only prompt for submission generation:

```text
You are solving a Thai math problem from an image.
Read all visible Thai text, numbers, labels, units, and math notation.
Solve carefully.
Return only the final answer exactly as it should appear in the CSV.
If the answer needs a Thai unit, include the unit.
If the exact answer is a fraction or radical, preserve exact math notation.
Do not include explanation.
```

For verification/reranking, allow short reasoning internally but require a final structured field:

```text
Given the image and candidate answers below, select the most likely correct final answer.
Return JSON only: {"answer": "...", "reason_code": "..."}.
Candidates: ...
```

## Source Notes

- MathVista project page: https://mathvista.github.io/
- OpenAI visual reasoning overview: https://openai.com/index/thinking-with-images/
- Qwen2.5-VL release: https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923
- ThinkLite-VL paper: https://arxiv.org/abs/2504.07934
- Seed1.5-VL technical report: https://arxiv.org/abs/2505.07062
- Open Vision Reasoner: https://arxiv.org/abs/2507.05255

