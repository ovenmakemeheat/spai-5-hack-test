# Chest X-ray Disease Detection: SOTA Model and Paper Ranking

Research snapshot: 2026-06-06

Task focus: multilabel chest X-ray disease detection, similar to the local notebook task where each image can have multiple positive findings and the competition metric is sample-average F1 after thresholding.

## Ranking Caveat

There is no single universal "best" chest X-ray model. Papers compare on different datasets, label sets, metrics, annotation noise levels, and train/test splits:

- CheXpert often reports AUROC on a small set of core findings.
- MIMIC-CXR and CXR-LT use noisier report-derived labels and more long-tailed diseases.
- Recent CXR-LT challenges emphasize mean average precision for rare, long-tailed multilabel disease detection.
- The current local task uses sample-average F1, so thresholding and label-set calibration matter more than AUROC alone.

The ranking below is therefore ordered by likely usefulness for a competitive chest disease classifier, not by a single leaderboard number.

## Short Recommendation

For this project, the strongest practical stack is:

1. Use a CXR-specific foundation encoder if accessible: CheXFound, Google CXR Foundation, BioViL-T, CheXagent vision encoder, or MedCLIP/BiomedCLIP.
2. Add a multilabel head trained with `BCEWithLogitsLoss`, asymmetric loss, or LDAM-style loss for imbalance.
3. Train several backbones at 224 and optionally 384/512 resolution.
4. Average probabilities across models.
5. Tune thresholds on validation for sample-average F1.
6. Force at least one positive label per image when the metric/data assumes every image has a finding.

## Ranked Models and Papers

| Rank | Model / paper | Why it ranks here | Best use in this project | Risks |
|---:|---|---|---|---|
| 1 | CheXFound + GLoRI | CXR-specific self-supervised foundation model pretrained on over 1M CXR images. The paper reports better performance than prior foundation models on CXR-LT 2024 40-finding classification and emphasizes global + local representations. | Best research-grade encoder/head idea for long-tailed multilabel disease detection. | More complex than a simple timm model. Need to verify public weights/code and integration cost. |
| 2 | Google CXR Foundation / ELIXR | Public Hugging Face model card describes CXR-specific embeddings, data-efficient classification, zero-shot classification, and strong CheXpert metrics. Good fit when labels are limited. | Extract embeddings, train a small multilabel classifier, ensemble with image backbones. | Gated terms of use, TensorFlow/JAX-oriented implementation, image pipeline differs from PyTorch/timm. |
| 3 | CheXagent / CheXagent vision encoder | Large CXR vision-language foundation model with disease classification, disease identification, report generation, grounding, and temporal tasks. Useful if the disease labels can be expressed as text prompts. | Use the CXR-adapted vision encoder or zero-shot disease classification as an auxiliary model. | Heavy model, research-only notice, prompt sensitivity, not necessarily optimized for this exact F1 metric. |
| 4 | BioViL-T | Domain-specific Microsoft CXR vision-language model trained with temporal multimodal pretraining over CXR images and reports. Strong for CXR representation learning and image-text tasks. | Use as an encoder or zero-shot/prompt model; ensemble with supervised heads. | Integration is more involved than timm. Temporal strength may not help if each sample has only one image. |
| 5 | Fine-tuned image-text encoders / UniCLIP-style methods | The 2024 Scientific Reports version of "Significantly improving zero-shot X-ray pathology classification" improves zero-shot CXR pathology classification by adapting contrastive image-text encoders to multilabel report structure. | If labels are scarce, use prompt-based scores or fine-tune a VLM using report/text-style positives and negatives. | Less direct for a closed 13-label supervised competition if enough labeled images exist. |
| 6 | CXR-LT challenge ensemble recipes | CXR-LT 2023/2024 solutions show practical patterns for long-tailed multilabel CXR: external data, class-wise ensembling, TTA, label text embeddings, noisy-label handling, and zero-shot strategies. | Copy the recipe: multiple backbones, class-wise thresholding, TTA, rare-class-aware loss, and probability ensembling. | Challenge metric is usually mAP, while this task uses sample-average F1. Must retune thresholds. |
| 7 | ConvNeXt-Large / modern timm supervised backbones with imbalance-aware loss | A 2026 CXR-LT report found ConvNeXt-Large best among tested CNN architectures and LDAM-DRW stronger than BCE/asymmetric losses for rare-class recognition on its benchmark. | Strong, practical supervised baseline using timm. Train with BCE first, then test ASL or LDAM-DRW. | Needs enough GPU memory. ImageNet pretraining is less domain-specific than CXR foundation models. |
| 8 | EVA/EVA02/Swin/ViT timm backbones | Modern transformer-style timm models can be strong when fine-tuned carefully. The local fastai+timm reference uses `eva02_base_patch14_224.mim_in22k`. | Add as a general-domain model in an ensemble with CXR-specific models. | ViTs can be data-hungry and may underperform domain-specific CXR encoders on small/noisy datasets. |
| 9 | TorchXRayVision DenseNet121 / ResNet CXR weights | Mature, easy-to-use CXR pretrained baselines with weights trained on NIH, PadChest, CheXpert, MIMIC, and combined datasets. | Fast baseline or ensemble member; good fallback when foundation models are hard to load. | DenseNet121 is older and usually not SOTA, but still reliable. Label spaces may not exactly match the competition labels. |
| 10 | MedCLIP / BiomedCLIP | The current notebook already uses this family. MedCLIP is attractive because it learns from unpaired medical images and text; BiomedCLIP is broader biomedical image-text pretraining. | Keep as the current baseline and ensemble it with timm/CXR Foundation/TorchXRayVision models. | May be behind newer CXR-specific foundation models; loading can be fragile. |

## Paper Notes

### 1. CheXFound: Chest X-ray Foundation Model with Global and Local Representations Integration

Source: https://arxiv.org/abs/2502.05142

Core idea:

- Self-supervised CXR foundation model.
- Pretrained on a curated CXR-1M dataset.
- Uses a GLoRI module to combine global image features and disease-specific local features.
- Reports strong performance on CXR-LT 2024 across 40 disease findings and better label efficiency.

Why it matters:

Chest disease classification is often driven by small local findings. A model that explicitly combines global context with disease-specific local regions is a better conceptual fit than a plain global-pooled classifier.

Project action:

- If weights are accessible, test frozen CheXFound features plus a multilabel head.
- If not, copy the idea: use attention pooling or query-based disease heads instead of only `global_pool="avg"`.

### 2. Google CXR Foundation / ELIXR

Source: https://huggingface.co/google/cxr-foundation

Core idea:

- Produces CXR embeddings for downstream classifiers.
- Supports data-efficient classification and zero-shot classification through image/text embedding variants.
- Model card reports CheXpert data-efficient classification mean AUC of `0.898` on five findings and zero-shot mean AUC of `0.846` across 13 findings.

Why it matters:

For a hackathon-scale dataset, a frozen CXR foundation embedding plus a calibrated classifier can be competitive and cheaper than full fine-tuning.

Project action:

- Extract embeddings for train/validation/test.
- Train logistic regression, LightGBM, or a small MLP per label.
- Blend probabilities with the current MedCLIP/timm model.

### 3. CheXagent

Sources:

- Paper: https://arxiv.org/abs/2401.12208
- Code: https://github.com/Stanford-AIMI/CheXagent

Core idea:

- Vision-language foundation model for CXR interpretation.
- Built for multiple task types, including disease classification, disease identification, view matching, grounding, and report generation.
- The repository exposes methods such as `binary_disease_classification` and `disease_identification`.

Why it matters:

It can turn disease names into prompts/tasks rather than requiring a fixed classifier head. This is useful for rare labels or label names that align well with radiology language.

Project action:

- Try zero-shot/prompt classification per label.
- Use its scores as ensemble features, not necessarily as the only model.

### 4. BioViL-T

Source: https://huggingface.co/microsoft/BiomedVLP-BioViL-T

Core idea:

- CXR-specific vision-language model.
- Uses temporal multimodal pretraining with CXR images and reports.
- Can be adapted to image/text classification, phrase grounding, and related CXR tasks.

Why it matters:

It is more domain-specific than generic BiomedCLIP and may provide better CXR representations, especially when disease label names can be encoded as text.

Project action:

- Use BioViL-T image embeddings with a multilabel head.
- Compare against MedCLIP and BiomedCLIP on the same validation split.

### 5. Fine-tuned Contrastive Image-Text Encoders for Zero-shot Pathology Classification

Source: https://arxiv.org/abs/2212.07050

Core idea:

- Improves zero-shot X-ray pathology classification by fine-tuning pretrained image-text encoders in a way that respects multilabel image-report pairs.
- Reports average macro AUROC gains across four CXR datasets and three pretrained models.

Why it matters:

Vanilla CLIP-style contrastive learning assumes one clean image-text pair, but CXR reports contain multiple findings. Loss design must reflect that multilabel structure.

Project action:

- Use positive and negative text prompts per disease.
- If enough compute exists, fine-tune an image-text encoder with multilabel-aware contrastive losses.

### 6. CXR-LT 2024 Challenge Overview

Source: https://arxiv.org/abs/2506.07984

Core idea:

- CXR-LT 2024 expands to 377,110 CXRs and 45 disease labels.
- Includes long-tailed classification on noisy and manually annotated test sets plus zero-shot generalization to unseen disease findings.
- Consolidates state-of-the-art solutions using multimodal models, noisy-label methods, generative strategies, and zero-shot learning.

Why it matters:

This is one of the closest benchmark settings to real multilabel CXR disease detection: many labels, rare diseases, noisy labels, and domain shift.

Project action:

- Use mAP ideas for ranking diagnostics, but optimize final thresholds for sample-F1.
- Add class-wise ensembling and TTA.
- Treat rare classes separately instead of relying on one global threshold.

### 7. Loss Design and Architecture Selection for Long-Tailed Multi-Label Chest X-Ray Classification

Source: https://arxiv.org/abs/2603.02294

Core idea:

- 2026 CXR-LT report comparing losses, CNN backbones, and post-training strategies.
- Reports LDAM with deferred re-weighting outperforming BCE and asymmetric losses for rare class recognition.
- Reports ConvNeXt-Large as the best single model in its experiments.

Why it matters:

This is directly relevant to class imbalance. The current notebook uses weighted BCE, which is a good start but likely not the endpoint for rare labels.

Project action:

- Baseline: weighted BCE.
- Next: asymmetric loss.
- Next: LDAM-DRW or logit-adjusted loss.
- Compare by validation sample-F1 and per-class F1, not training loss.

### 8. TorchXRayVision

Sources:

- Paper: https://arxiv.org/abs/2111.00595
- Code: https://github.com/mlmed/torchxrayvision

Core idea:

- Open-source CXR library with common preprocessing, datasets, and pretrained models.
- Provides DenseNet121 weights for NIH, PadChest, CheXpert, MIMIC-CXR, and combined training data.
- Also includes 512-resolution ResNet weights and official/baseline CheXpert models.

Why it matters:

It is not the newest method, but it is one of the easiest ways to get a domain-pretrained CXR model running quickly.

Project action:

- Use `densenet121-res224-all`, `densenet121-res224-chex`, and `densenet121-res224-mimic_ch` as ensemble members.
- Replace the final target mapping with this competition's 13 labels.

### 9. CXRBase

Source: https://arxiv.org/abs/2410.08861

Core idea:

- Self-supervised foundation model trained on 1.04M unlabeled CXR images.
- Then fine-tuned with labels for disease detection.
- Targets generalization across diverse clinical settings.

Why it matters:

It supports the same strategic direction as CheXFound: domain-specific self-supervised CXR pretraining is more promising than generic ImageNet pretraining alone.

Project action:

- Track for weights/code.
- If unavailable, prioritize CheXFound, Google CXR Foundation, or TorchXRayVision first.

## Implementation Ranking for This Repo

This is the order I would try in the local notebook/codebase:

| Priority | Experiment | Expected value | Effort |
|---:|---|---|---|
| 1 | Add TorchXRayVision DenseNet121 CXR-pretrained backbones | Fast domain-pretrained baseline | Low |
| 2 | Add timm ConvNeXt/EVA02/Swin backbones with current multilabel head | Strong supervised ensemble diversity | Low-medium |
| 3 | Add asymmetric loss and compare with weighted BCE | Better imbalance handling | Low |
| 4 | Tune per-class thresholds and compare to global threshold | Often improves multilabel F1 | Low |
| 5 | Add test-time augmentation and probability averaging | Better robustness | Low-medium |
| 6 | Use Google CXR Foundation embeddings with a small classifier | Strong label-efficient foundation baseline | Medium |
| 7 | Use BioViL-T embeddings or prompts | Strong CXR VLM signal | Medium |
| 8 | Use CheXagent disease scores as ensemble features | Useful zero-shot/semantic auxiliary model | Medium-high |
| 9 | Reproduce CheXFound/GLoRI-style disease query heads | Highest research upside | High |

## Suggested Model Ensemble

Start with a small but diverse ensemble:

1. `MedCLIP-ViT` from the current notebook.
2. `torchxrayvision` DenseNet121 trained on combined CXR datasets.
3. `timm` `convnext_base` or `convnext_large` with ImageNet-22K pretraining if available.
4. `timm` `eva02_base_patch14_224.mim_in22k`.
5. Google CXR Foundation embeddings plus a linear/MLP classifier if access works.

Blend:

```text
final_prob = weighted_average(model_probabilities)
```

Then tune:

- One global threshold for baseline comparability.
- Per-class thresholds for better class balance.
- Optional rule: if no labels are positive, set the top probability label to positive.

## Losses to Test

| Loss | Why test it |
|---|---|
| Weighted BCE | Current baseline; simple and stable. |
| Focal loss | Can focus learning on hard positives/negatives. |
| Asymmetric loss | Designed for multilabel imbalance with many easy negatives. |
| LDAM-DRW | Recent CXR-LT report found it strong for rare-class recognition. |
| Logit-adjusted BCE | Simple prior correction for imbalanced labels. |

Use validation sample-F1 and per-class F1 as the deciding metrics. AUROC can stay as a secondary diagnostic.

## Source List

- CheXFound, "Chest X-ray Foundation Model with Global and Local Representations Integration": https://arxiv.org/abs/2502.05142
- Google CXR Foundation / ELIXR model card: https://huggingface.co/google/cxr-foundation
- CheXagent paper: https://arxiv.org/abs/2401.12208
- CheXagent code: https://github.com/Stanford-AIMI/CheXagent
- BioViL-T model card: https://huggingface.co/microsoft/BiomedVLP-BioViL-T
- Fine-tuned image-text encoders for zero-shot CXR pathology classification: https://arxiv.org/abs/2212.07050
- CXR-LT 2024 challenge overview: https://arxiv.org/abs/2506.07984
- CXR-LT 2026 loss/backbone report: https://arxiv.org/abs/2603.02294
- TorchXRayVision paper: https://arxiv.org/abs/2111.00595
- TorchXRayVision code: https://github.com/mlmed/torchxrayvision
- CXRBase: https://arxiv.org/abs/2410.08861
