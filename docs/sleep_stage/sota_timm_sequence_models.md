# SOTA Model Ideas: `timm` Pretraining and Better Sequence Heads

This note answers whether a pretrained `timm` model can replace the CNN part of the current PyTorch CNN-LSTM baseline, and what model should replace or improve the LSTM head.

Current baseline notebook:

```text
notebook/sleep_stage/sleep_stage.ipynb
```

Current data format:

```text
480 timesteps x 8 channels
BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI
```

Each sample is one 30-second wearable segment sampled at 16 Hz.

## Short Answer

Yes, `timm` pretrained models can be used, but not directly on `(480, 8)` time-series tensors. `timm` models are image backbones, so the signal must be converted into an image-like tensor.

Best practical approach:

```text
raw 480 x 8 signal
-> time-frequency image or channel-time image
-> pretrained timm encoder
-> segment embedding
-> Mamba / Transformer / TCN sequence head
-> sleep-stage prediction
```

For this competition, the strongest practical replacement for CNN-LSTM is:

```text
Dual encoder:
  1D raw-signal encoder
  + timm pretrained spectrogram encoder

Sequence head:
  Mamba or Transformer over consecutive segment embeddings
```

If time is limited, use:

```text
timm ConvNeXt-Tiny encoder + BiGRU/Transformer head
```

If large models are allowed, use:

```text
timm ConvNeXt-Base/Large or Swin-Base spectrogram encoder
+ raw-signal TCN encoder
+ Mamba/Transformer sequence head
+ ensemble across folds
```

## Why `timm` Is Not a Direct Drop-In

The baseline CNN uses `Conv1d` on tensors shaped:

```text
batch x time x channels
```

`timm` models expect image tensors shaped:

```text
batch x image_channels x height x width
```

So the competition signal must be transformed first. Good options:

### Option 1: Channel-Time Image

Create an image with:

```text
height = 8 sensor channels
width = 480 timesteps
```

This is simple and fast, but ImageNet pretraining is less aligned because the image has only 8 rows.

### Option 2: Time-Frequency Image

Create STFT, CWT, or log-FFT maps for each signal channel.

Example:

```text
BVP spectrogram
ACC magnitude spectrogram
HR/IBI/EDA/TEMP feature map
```

Then stack into 3 channels or adapt the first convolution to more channels.

This is the better `timm` path because pretrained image backbones work better on texture-like 2D patterns than on a thin `8 x 480` strip.

### Option 3: Hybrid Raw + Spectrogram

Use both:

```text
raw branch: 1D TCN / Conv1D
spectrogram branch: timm pretrained encoder
fusion: concatenate embeddings
sequence head: Mamba / Transformer / TCN
```

This is the highest-confidence architecture for this dataset.

## Recommended `timm` Backbones

### 1. ConvNeXt-Base or ConvNeXt-Large

Recommended first when large models are allowed.

Why:

- Strong modern CNN-style ImageNet backbone.
- Easier to fine-tune than pure ViT on small datasets.
- Works well when the input has local texture, such as spectrograms.
- More stable than pure large Transformers when only 83 training recordings are available.

Suggested models:

```python
convnext_tiny.fb_in22k_ft_in1k
convnext_base.fb_in22k_ft_in1k
convnext_large.fb_in22k_ft_in1k
```

Use `ConvNeXt-Base` as the main large baseline. Try `ConvNeXt-Large` if GPU memory is enough. Keep `ConvNeXt-Tiny` only as a fast debug model.

### 2. EfficientNetV2 / EfficientNet-B0-B3

Recommended for speed.

Why:

- Good accuracy/compute tradeoff.
- Useful for Kaggle notebooks with limited runtime.
- Often strong on small image-like classification tasks.

Suggested models:

```python
tf_efficientnetv2_s.in21k_ft_in1k
tf_efficientnet_b0.ns_jft_in1k
tf_efficientnet_b3.ns_jft_in1k
```

### 3. Swin Transformer Base / Large

Recommended if using larger spectrogram images and sufficient GPU memory.

Why:

- Windowed attention can capture local and mid-range patterns.
- Better suited to 2D spectrograms than raw channel-time strips.

Suggested model:

```python
swin_tiny_patch4_window7_224.ms_in22k_ft_in1k
swin_base_patch4_window7_224.ms_in22k_ft_in1k
swin_large_patch4_window7_224.ms_in22k_ft_in1k
```

Risk:

- More likely to overfit than ConvNeXt/EfficientNet if the dataset is small.
- More memory hungry than ConvNeXt.

### 4. MaxViT / CoAtNet / EVA-style ViT

Recommended as a later experiment.

Why:

- Hybrid CNN-attention architecture.
- Good general image backbone family.
- Large ViT-style models can be strong if the spectrogram representation is high quality and augmentation is used.

Risk:

- More compute.
- More tuning required.
- Pure ViT-style models are easier to overfit on small tabular-sensor competitions than ConvNeXt.

Suggested models to probe if available in the installed `timm` version:

```python
maxvit_base_tf_224.in21k_ft_in1k
coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k
eva02_base_patch14_224.mim_in22k
eva02_large_patch14_224.mim_in22k
```

## How to Use `timm`

The official `timm` API supports pretrained model creation with `pretrained=True`, changing the classifier with `num_classes`, and using feature extraction modes. Documentation:

- `timm` model API: <https://huggingface.co/docs/timm/en/reference/models>
- `timm` feature extraction: <https://huggingface.co/docs/timm/v1.0.8/feature_extraction>
- PyTorch Image Models GitHub: <https://github.com/huggingface/pytorch-image-models>

Simple classifier:

```python
import timm
from torch import nn

model = timm.create_model(
    "convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    in_chans=3,
    num_classes=5,
)
```

Feature extractor for sequence modeling:

```python
encoder = timm.create_model(
    "convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    in_chans=3,
    num_classes=0,
    global_pool="avg",
)

embedding = encoder(image_batch)
```

If using more than 3 input channels:

```python
encoder = timm.create_model(
    "convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    in_chans=8,
    num_classes=0,
    global_pool="avg",
)
```

For small datasets, even with a large model, start by freezing most of the encoder:

```python
for p in encoder.parameters():
    p.requires_grad = False
```

Then unfreeze the last blocks after the classifier stabilizes.

Large-model fine-tuning schedule:

```text
stage 1: freeze timm encoder, train classifier/sequence head only
stage 2: unfreeze final timm stage, low LR
stage 3: unfreeze all layers for 1 to 3 epochs, very low LR
stage 4: restore best validation weighted F1 checkpoint
```

Suggested learning rates:

```text
head LR: 1e-3
last-stage encoder LR: 1e-5 to 3e-5
full encoder LR: 1e-6 to 1e-5
```

## Better Replacement for LSTM

The LSTM in the baseline models the compressed sequence inside one 30-second segment. That is useful, but sleep staging usually benefits more from modeling transitions across neighboring 30-second segments.

The better sequence unit is:

```text
segment encoder -> sequence model over segment embeddings
```

Instead of:

```text
Conv1D -> LSTM inside one segment only
```

## Ranked Sequence Heads

### 1. Mamba / State-Space Model

Best SOTA direction for long sequence context.

Why:

- Recent sleep-staging papers apply Mamba to long-range temporal dependencies.
- A wearable non-EEG sleep-staging paper uses Mamba on multimodal wearable signals, including PPG, accelerometry, ECG, and temperature.
- Mamba scales better than standard attention for long overnight sequences.

Best architecture:

```text
per-segment encoder: raw 1D TCN + timm spectrogram encoder
sequence head: bidirectional Mamba or Mamba blocks
classifier: linear layer over each segment embedding
```

References:

- Mamba wearable sleep staging without EEG: <https://academic.oup.com/sleep/article/doi/10.1093/sleep/zsag022/8466336>
- Mamba-CAM-Sleep: <https://pubmed.ncbi.nlm.nih.gov/41336521/>
- BiT-MamSleep: <https://arxiv.org/abs/2411.01589>

Practical note:

- Use this after the baseline is stable. Mamba packages can be harder to install on Kaggle than standard PyTorch modules.

### 2. Small Transformer Encoder

Best easy SOTA-style replacement if Mamba setup is risky.

Why:

- Captures sleep-stage transitions across neighboring windows.
- Easy to implement in plain PyTorch with `nn.TransformerEncoder`.
- Can be used on top of frozen `timm` embeddings.

Best architecture:

```text
timm segment encoder
-> positional encoding
-> 2 to 4 Transformer encoder layers
-> per-segment classifier
```

References:

- SleepTransformer: <https://arxiv.org/abs/2105.11043>
- SleepViTransformer: <https://www.sciencedirect.com/science/article/pii/S1746809423006365>
- SleepGPT / time-frequency foundation model: <https://www.nature.com/articles/s41467-025-67970-4>

Practical note:

- Keep it small: `d_model=256`, `nhead=4`, `num_layers=2`.
- Use dropout and grouped validation by subject/file.

### 3. TCN Over Segment Embeddings

Best robust fallback.

Why:

- Easy to train.
- Good inductive bias for local temporal transitions.
- Less overfit-prone than a Transformer on small data.
- The reference notebook already uses TCN ideas.

Best architecture:

```text
timm/raw segment encoder
-> residual dilated Conv1D blocks over segment sequence
-> classifier
```

Use dilation rates:

```text
1, 2, 4, 8, 16
```

### 4. BiGRU / BiLSTM

Acceptable but no longer the strongest choice.

Why:

- Easy and reliable.
- Better than classifying every 30-second segment independently.

Why lower ranked:

- Mamba, TCN, and Transformer heads usually provide better long-context modeling.
- Recurrent models are slower over long sequences.

## Best Model Combinations

### Rank 1: Raw TCN + Large `timm` ConvNeXt Spectrogram + Mamba Head

Best expected accuracy if implementation time allows.

```text
raw 480 x 8
  -> 1D residual TCN encoder

spectrogram/log-FFT image
  -> timm ConvNeXt-Base/Large encoder

concat embeddings
  -> Mamba sequence head across subject segments
  -> classifier
```

Why it matches this dataset:

- Raw branch preserves morphology.
- Spectrogram branch uses ImageNet-pretrained 2D pattern recognition.
- Mamba models sleep-stage transitions over consecutive 30-second segments.

### Rank 2: Large `timm` ConvNeXt/Swin Spectrogram + Transformer Head

Best practical SOTA-style model.

```text
spectrogram/log-FFT image
  -> timm ConvNeXt-Base/Large or Swin-Base
  -> TransformerEncoder over neighboring segments
  -> classifier
```

Why:

- Easier than Mamba.
- Stronger than CNN-LSTM if sequence context is built correctly.
- Good Kaggle balance of implementation speed and performance.

Large version:

```text
d_model = 512 or 768
nhead = 8 or 12
num_layers = 4 to 6
dropout = 0.2 to 0.4
```

### Rank 3: Raw TCN + `timm` EfficientNetV2-L/ConvNeXt-Base + TCN Head

Best stable competition model.

```text
raw signal branch
  -> residual TCN

image branch
  -> timm EfficientNetV2-L or ConvNeXt-Base

fused embeddings
  -> residual TCN over segment sequence
  -> classifier
```

Why:

- Efficient.
- Stable.
- Less fragile than Transformer/Mamba.

### Rank 4: `timm` Swin-Base/Large Spectrogram + Transformer Head

High potential, higher risk.

```text
spectrogram image
  -> Swin-Base/Large
  -> Transformer head
```

Why:

- Strong attention-based image and sequence modeling.
- May overfit without careful validation, augmentation, and regularization.

## Suggested First Implementation

If large models are allowed, implement this first:

```text
ConvNeXt-Base spectrogram encoder + 4-layer Transformer head
```

Use:

- freeze `timm` encoder for 3 to 5 epochs;
- train the head first;
- unfreeze the last ConvNeXt stage;
- use weighted cross-entropy or focal loss;
- validate by grouped file split;
- predict segments in original subject order.

Input image recipe:

```text
1. For each 480 x 8 segment, normalize with train scaler.
2. Compute `np.log1p(np.abs(np.fft.rfft(x, axis=0)))`.
3. Result shape: frequency_bins x 8 channels.
4. Resize/interpolate to 224 x 224.
5. Convert to 3 channels:
   - channel 1: BVP spectrum
   - channel 2: accelerometer magnitude spectrum
   - channel 3: mean of TEMP, EDA, HR, IBI spectra
```

This is more aligned with ImageNet pretraining than an `8 x 480` channel-time strip.

## Large-Model Competition Plan

Use the following order:

### Model A: ConvNeXt-Base Spectrogram + Transformer

```text
log-FFT / STFT image
-> ConvNeXt-Base pretrained by timm
-> 4-layer Transformer over subject segment order
-> weighted CE or focal loss
```

This is the best first large model because ConvNeXt is strong and stable.

### Model B: ConvNeXt-Large Spectrogram + TCN Head

```text
log-FFT / STFT image
-> ConvNeXt-Large
-> residual temporal convolution head
```

This is useful if Transformer overfits or is too slow.

### Model C: Raw Signal TCN + Spectrogram ConvNeXt-Base Fusion

```text
raw 480 x 8
-> residual 1D TCN

spectrogram image
-> ConvNeXt-Base

concat
-> Transformer/TCN/Mamba head
```

This should beat a spectrogram-only model if raw BVP/ACC morphology matters.

### Model D: Swin-Base Spectrogram + Transformer

```text
spectrogram image
-> Swin-Base
-> Transformer head
```

Use as an ensemble member, not necessarily the primary model.

### Model E: Mamba Sequence Head

```text
segment embeddings from Model A or C
-> bidirectional Mamba/state-space sequence model
-> classifier
```

Use after the ConvNeXt + Transformer pipeline is working.

## Large-Model Ensemble

Final ensemble candidates:

```text
0.35 * ConvNeXt-Base + Transformer
0.25 * ConvNeXt-Large + TCN
0.20 * Raw TCN + ConvNeXt-Base fusion
0.10 * Swin-Base + Transformer
0.10 * baseline CNN-LSTM / residual TCN
```

Average probabilities before taking `argmax`. If only hard labels are available, majority vote is weaker but still usable.

For weighted F1, tune class-specific thresholds or class prior correction on validation folds:

```text
prediction_logit[class] += class_bias[class]
```

Search `class_bias` on validation data to improve minority-stage F1 without destroying dominant-class performance.

## Final Recommendation

For immediate competition improvement:

```text
1. Keep PyTorch baseline.
2. Add spectrogram image generation.
3. Train ConvNeXt-Base from timm as a segment encoder.
4. Add a 4-layer Transformer or residual TCN over neighboring segment embeddings.
5. Try Mamba only after the above is working.
```

The best research-aligned model is Mamba-based multimodal wearable sleep staging. With a large-model budget, the best practical Kaggle implementation is probably ConvNeXt-Base/Large spectrogram embeddings plus a Transformer/TCN sequence head, then a raw-signal fusion branch and ensemble.
