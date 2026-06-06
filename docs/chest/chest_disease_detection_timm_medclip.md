# Notebook Summary: Chest Disease Detection

Source notebooks:

- `notebook/chest/ref/gold_chest_disease_detection_baseline_row37.ipynb`
- `notebook/chest/ref/silver_house_recognition_fastai_timm_row31.ipynb`

## Objective

The chest notebook builds a multilabel chest X-ray disease detection pipeline. Each image can contain more than one finding, so the model predicts 13 independent binary labels instead of one mutually exclusive class.

The final output is a Kaggle-style `submission.csv` that preserves pre-filled rows in the official template and fills only rows whose label columns are empty.

The competition metric is sample-average F1:

```text
f1_score(y_true, y_pred_binary, average="samples")
```

This matters because the pipeline must optimize binary label sets per image, not just probability ranking or per-class AUC. The notebook tracks mean per-class AUC as a secondary diagnostic, but model selection and threshold tuning are driven by sample-average F1.

## Core Chest Disease Pipeline

The gold notebook follows this sequence:

1. Detect data location from Kaggle input, local zip extraction, or local folders.
2. Load `train.csv`, infer all disease label columns, and clip/fill label values to `0/1`.
3. Read the official `test_submission.csv` template and identify rows that need predictions.
4. Explore label imbalance, number of findings per image, label co-occurrence, and sample X-rays.
5. Build chest-specific image transforms.
6. Fine-tune a pretrained medical vision encoder with a multilabel classification head.
7. Tune a global probability threshold for sample-average F1.
8. Generate binary predictions and write `submission.csv`.

## Data Handling Techniques

The notebook is careful about competition file structure:

- It searches for `train.csv` in common Kaggle and local paths.
- It finds the actual image directory by scanning likely folders such as `images/images`, `images`, and nested directories.
- It filters training rows to filenames that exist on disk.
- It treats every column except `filename` as a disease label.
- It uses `test_submission.csv` as the source of truth for output column order and pre-filled rows.

The template handling is important. Some submission rows are already filled, so the notebook builds:

```text
PREDICT_MASK = rows where all label columns are NaN
```

Only those rows are predicted. Existing rows are copied through unchanged.

## Chest-Specific Preprocessing

Images are opened as RGB even though chest X-rays are grayscale. This keeps compatibility with pretrained 3-channel backbones.

For MedCLIP, the notebook uses grayscale CXR normalization repeated across three channels:

```text
mean = [0.5862785803043838] * 3
std  = [0.27950088968644304] * 3
```

Training transforms:

- Resize to `224 x 224`.
- Small random affine perturbation: rotation, translation, and scale.
- Light brightness and contrast jitter.
- Tensor conversion and normalization.

Evaluation transforms:

- Resize to `224 x 224`.
- Tensor conversion and normalization.

The notebook intentionally avoids horizontal flip because laterality can matter in chest radiographs. Flipping can move heart, aorta, or disease-side cues into anatomically implausible positions.

## Model Architecture

The gold notebook uses a pretrained image encoder plus a simple multilabel head:

```text
image -> pretrained encoder -> feature vector -> dropout -> linear(13)
```

The output layer has one logit per disease label. It does not use softmax because labels are not mutually exclusive. Probabilities are produced with sigmoid at validation and prediction time.

### Backbone Strategy

The notebook tries backbones in this order when `BACKBONE=auto`:

1. `medclip-vit`
2. `biomedclip`
3. `densenet121`

MedCLIP is the preferred option because it was pretrained on medical chest imaging data such as CheXpert and MIMIC-style radiographs. BiomedCLIP is a broader biomedical fallback. DenseNet121 is an ImageNet/timm fallback when medical CLIP weights are unavailable.

The fallback design is useful for hackathon work because it keeps the notebook runnable across different environments, even when a specialized checkpoint fails to download or load.

## Training Technique

The task is trained as multilabel classification with:

```text
BCEWithLogitsLoss(pos_weight=class_weights)
```

Positive class weights are computed from the training split:

```text
pos_weight = negative_count / positive_count
```

The weights are clipped to a maximum of `10.0`. This helps rare labels contribute to the loss without letting extremely sparse classes dominate training.

The optimizer uses discriminative learning rates:

| Parameters | Learning rate |
|---|---:|
| Pretrained encoder | `1e-5` |
| New classification head | `1e-4` |

This is a practical fine-tuning setup: the pretrained backbone changes slowly while the randomly initialized head learns faster.

Other training details:

- `AdamW` optimizer.
- Weight decay `1e-4`.
- Cosine annealing learning-rate schedule.
- CUDA automatic mixed precision when available.
- 90/10 train/validation split.
- Best checkpoint selected by validation sample-average F1, not validation loss.

## Threshold Tuning

Because the competition metric uses binary predictions, raw sigmoid probabilities are not enough. The notebook sweeps a global threshold from `0.05` to `0.60` in `0.01` increments and chooses the threshold that maximizes sample-average F1.

After thresholding, it applies an important fallback:

```text
if an image has no predicted positive labels:
    set the highest-probability label to 1
```

This matches the training assumption that every image has at least one finding and prevents all-zero predictions from hurting sample-level F1.

## Evaluation

The notebook evaluates with two complementary views:

- Mean per-class ROC-AUC for ranking quality.
- Sample-average F1 after thresholding for the actual competition metric.

It also computes per-class F1 at the tuned threshold. This helps identify labels that remain weak because of imbalance, visual ambiguity, or insufficient positive examples.

## timm Modeling Ideas From the Fastai Reference

The silver reference notebook is for house image recognition, not chest disease detection, but it contributes useful timm modeling patterns:

- Use `timm.list_models(pattern, pretrained=True)` to discover available pretrained architectures.
- Try modern transformer-style backbones such as EVA/EVA02.
- Use pretrained weights and fine-tune rather than training from scratch.
- Start with frozen backbone epochs, then unfreeze for full fine-tuning.
- Use callbacks for best-checkpoint saving and early stopping.
- Run inference through a test dataloader and convert model probabilities into submission labels.

The referenced model choice is:

```text
eva02_base_patch14_224.mim_in22k
```

This is a strong general-purpose image backbone. For chest disease detection, it should be adapted as a multilabel PyTorch/timm model rather than copied as a fastai single-label classifier.

## How to Adapt timm to the Chest Task

A timm version of the chest model can follow the same structure as the DenseNet fallback:

```python
import timm
import torch.nn as nn

encoder = timm.create_model(
    "eva02_base_patch14_224.mim_in22k",
    pretrained=True,
    num_classes=0,
    global_pool="avg",
)

model = nn.Sequential(
    encoder,
    nn.Dropout(0.2),
    nn.Linear(encoder.num_features, 13),
)
```

Key changes compared with the fastai reference:

- Use `ImageBlock + CategoryBlock` only for single-label tasks; chest disease needs multilabel targets.
- Use `BCEWithLogitsLoss`, not cross-entropy.
- Use sigmoid probabilities, not softmax.
- Optimize and tune a binary threshold for sample-average F1.
- Preserve official submission template rows instead of creating a fresh test-label file.

For timm backbones, start with ImageNet normalization unless the model's pretrained config provides a different mean/std. If using a medical pretrained model, prefer its domain-specific normalization.

## Practical Modeling Extensions

Strong next experiments for this task:

1. Try several timm encoders with the same multilabel head:
   - `densenet121`
   - `convnext_tiny`
   - `tf_efficientnetv2_s`
   - `swin_tiny_patch4_window7_224`
   - `eva02_base_patch14_224.mim_in22k`
2. Use per-class thresholds instead of one global threshold. Optimize each label's F1 on validation, then compare against global-threshold sample F1.
3. Increase input size to `384` for architectures with good pretrained weights at that size.
4. Add test-time augmentation with mild resize/brightness variants and average probabilities.
5. Use stratified multilabel splitting if label distribution drift appears in validation.
6. Ensemble medical-domain and timm general-domain models by averaging probabilities before threshold tuning.
7. Freeze the encoder for one warmup epoch, then unfreeze with a lower encoder learning rate.

## Main Takeaways

The strongest technique in the chest notebook is aligning every modeling choice with the actual multilabel F1 objective:

- Medical pretrained encoder when available.
- No softmax because findings can co-exist.
- Weighted BCE for class imbalance.
- Validation threshold sweep for sample-average F1.
- At least one positive prediction per image.
- Best checkpoint selected by F1.
- Submission template copied exactly, with only empty rows filled.

The timm reference adds a useful model-search and fine-tuning mindset. Its best ideas are pretrained modern backbones, frozen-to-unfrozen training, early stopping, and clean dataloader-based inference. To use those ideas for chest disease detection, keep the chest notebook's multilabel loss, sigmoid outputs, F1 threshold tuning, and template-preserving submission logic.
