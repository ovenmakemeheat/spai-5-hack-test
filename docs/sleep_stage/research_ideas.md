# Research Ideas and State-of-the-Art Models for Sleep Stage Classification

This note summarizes research directions relevant to the notebook `notebook/sleep_stage/ref/final-hack-iot-tcnlstm-cnn-superai-ss5.ipynb`, which uses wearable-style multivariate signals, FFT features, CNN/LSTM experiments, and an improved residual TCN.

## Kaggle Competition Data Card

Competition: <https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-sleep-stage-classification/>

Data card information read with the Kaggle MCP:

- `train/`: 83 CSV files.
- Each training file contains continuous numerical values from multiple sensors.
- Signals are resampled to `16 Hz`.
- Each `30-second` segment has one sleep-stage label.
- `test_segment/`: test data is already split by subject ID into `30-second` intervals.
- Each segmented test file corresponds to one `id` in `sample_submission.csv`.
- Evaluation metric: weighted F1-score.
- Total files: 7,916 CSV files.
- Total data size: about 5.45 GB.

One downloaded test sample was inspected:

```text
dataset/kaggle_probe/test001_00000.csv
shape: (480, 8)
```

The `480` rows exactly match `30 seconds * 16 Hz`.

Columns in the downloaded test segment:

```text
BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI
```

First rows from the sample:

```text
BVP         ACC_X      ACC_Y      ACC_Z      TEMP      EDA       HR       IBI
-16.787332  -29.654620 -35.583667  44.486126 30.773655 0.769030 52.986465 0.911322
  2.375216  -30.549130 -35.593414  44.482602 30.773662 0.769137 52.986463 0.911322
 46.691694  -29.650487 -35.582574  44.487709 30.773652 0.768899 52.986385 0.911322
```

Training files include the same eight signal columns plus `Sleep_Stage`, based on the notebook preprocessing and the downloaded training-file header:

```text
BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI, Sleep_Stage
```

This means the problem is best treated as **wearable multimodal 30-second sleep-stage classification**, not EEG/PSG sleep staging. Papers that rely only on EEG are useful for architecture ideas, but they are a weaker dataset match.

## Ranked Research Match for This Competition

Ranking criteria:

- Dataset match: similarity to `BVP/PPG`, accelerometer, temperature, EDA, HR, and IBI wearable signals.
- Accuracy potential: whether the model family is known to work well for sleep-stage sequence modeling.
- Implementation fit: whether it can be adapted quickly to the current notebook and weighted F1 competition setup.

### 1. Google Wearable CNN Sleep Staging

Rank: highest dataset match.

Why it matches:

- Uses wearable-style signals rather than EEG-only PSG.
- Uses raw PPG-like data and 3D accelerometer signals, which closely match this competition's `BVP` and `ACC_X/Y/Z`.
- The competition also provides derived physiological channels, `HR` and `IBI`, plus `TEMP` and `EDA`, which can improve over a PPG+ACC-only setup.

Best idea to use:

- Train a deep 1D CNN or residual temporal CNN directly on raw 30-second windows.
- Add feature branches for BVP/PPG, accelerometer, and low-rate physiological features.
- Use neighboring windows for context if possible.

Expected competition fit:

- Very strong. This is the closest paper family to the actual signal format.

Paper page: <https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/>

### 2. Wearable Mamba / State-Space Sleep Staging Without EEG

Rank: strongest modern sequence-model direction for wearable data.

Why it matches:

- Targets wearable sleep staging without EEG.
- State-space sequence models are designed for long temporal sequences and are more efficient than full attention.
- This competition has subject-level sequences split into many 30-second files, so long-context modeling should help.

Best idea to use:

- Start with the current residual TCN.
- Add a compact Mamba/state-space block over consecutive 30-second segment embeddings.
- Predict labels as a sequence, then write each segment prediction to the submission file.

Expected competition fit:

- Very strong if implementation time allows sequence batching by subject.
- Riskier than CNN/TCN because Mamba dependencies and implementation details may be harder under hackathon constraints.

Paper: <https://academic.oup.com/sleep/article/49/4/zsag022/8466336>

### 3. U-Time / U-Sleep Fully Convolutional Sequence Segmentation

Rank: best proven sleep-stage sequence architecture.

Why it matches:

- Sleep staging is naturally sequence segmentation.
- The competition explicitly says continuous signals can be considered as continuous 30-second segments.
- U-Time and U-Sleep predict labels over long physiological sequences instead of isolated windows.

Best idea to use:

- Reconstruct each training file as a sequence of 480-row windows.
- Feed multiple consecutive windows to a U-Time-style encoder-decoder.
- Output one label per 30-second segment.

Expected competition fit:

- High. Dataset match is slightly weaker than wearable-specific CNN/Mamba papers because U-Sleep is mainly PSG-focused, but the modeling structure is very appropriate.

References:

- U-Time: <https://papers.nips.cc/paper/8692-u-time-a-fully-convolutional-network-for-time-series-segmentation-applied-to-sleep-staging>
- U-Sleep: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/>

### 4. Dual-Branch Residual TCN: Raw Signal + FFT

Rank: most practical upgrade from the current notebook.

Why it matches:

- The existing notebook already uses FFT magnitudes and an improved residual TCN.
- The downloaded test sample confirms each example is exactly one 480-step multichannel time series.
- A raw branch keeps morphology information that FFT-only preprocessing may discard.

Best idea to use:

- Branch A: raw normalized `480 x 8` signal.
- Branch B: `log1p(abs(rfft(signal)))` frequency features.
- Fuse both embeddings.
- Classify with residual TCN blocks and weighted/focal loss.

Expected competition fit:

- High and fast to implement.
- Probably the best next experiment if there is limited time.

### 5. SleepTransformer / Small Transformer After CNN Features

Rank: good accuracy potential, medium dataset match.

Why it matches:

- Attention can model transitions across multiple sleep windows.
- Works better when given a sequence of segment embeddings rather than one isolated 480-row window.

Why it is lower ranked:

- Full Transformers can overfit with only 83 training recordings.
- The dataset is wearable multimodal, while many sleep Transformer papers focus on EEG or spectrogram-style PSG inputs.

Best idea to use:

- Use CNN/TCN to encode each 30-second segment.
- Apply a small Transformer encoder over neighboring segment embeddings.

References:

- SleepTransformer: <https://arxiv.org/abs/2105.11043>
- SleepViTransformer: <https://www.sciencedirect.com/science/article/pii/S1746809423006365>

### 6. DeepSleepNet / TinySleepNet

Rank: useful baseline ideas, lower dataset match.

Why it matches:

- They combine local feature extraction with temporal context.
- They are classic references for automatic sleep staging.

Why they are lower ranked:

- They target EEG, not wearable BVP/ACC/TEMP/EDA/HR/IBI data.
- Direct transfer of EEG-specific assumptions is likely weaker than wearable-specific models.

Best idea to use:

- Borrow the CNN + sequence-context pattern, not the exact EEG-specific architecture.
- TinySleepNet's simplicity is useful if validation overfitting is severe.

References:

- DeepSleepNet: <https://arxiv.org/abs/1703.04046>
- TinySleepNet: <https://pubmed.ncbi.nlm.nih.gov/33018069/>

### 7. SleepFM / SleepGPT Foundation Models

Rank: research frontier, lowest immediate implementation fit.

Why it matters:

- Foundation models are the current state-of-the-art direction for large-scale sleep representation learning.
- They are especially useful when large unlabeled physiological datasets are available.

Why it is lower ranked for this competition:

- These models are not simple drop-in Kaggle baselines.
- They are typically PSG/foundation-model oriented, while this competition is wearable multimodal and time-limited.

Best idea to use:

- If unlabeled training/test recordings can be used legally, pretrain with masked reconstruction or contrastive learning on the 8-channel sensor streams.
- Fine-tune on weighted F1 classification.

References:

- SleepFM: <https://www.nature.com/articles/s41591-025-04133-4>
- SleepGPT: <https://www.nature.com/articles/s41467-025-67970-4>

## Recommended Direction

For this competition, the most practical next model is a **dual-branch residual temporal model**:

- Time-domain branch: raw normalized signals.
- Frequency-domain branch: FFT or log-FFT features.
- Temporal backbone: residual TCN, U-Time-style encoder-decoder, or lightweight Mamba/state-space block.
- Sequence smoothing/context: predict neighboring 30-second windows jointly instead of treating each `480 x 8` segment independently.
- Class imbalance handling: weighted cross-entropy, focal loss, or class-balanced sampling.

This direction fits the available notebook because it already uses 480-sample windows, FFT magnitudes, weighted F1, and residual TCN blocks.

## Why This Idea Fits the Notebook

The notebook currently converts each 480-row window into FFT magnitude features and trains an improved residual TCN. That is a reasonable competition baseline because physiological sleep signals contain both:

- local morphology, such as movement bursts, BVP/PPG changes, HR/IBI variability, and EDA fluctuations;
- longer temporal structure, such as transition patterns between wake, REM, N1, N2, and N3.

The main limitation is that each window is predicted independently. Sleep staging is usually a sequence-labeling task, so using neighboring windows should improve consistency and reduce impossible transitions.

## Key Papers

### DeepSleepNet

DeepSleepNet is an early influential CNN + RNN model for automatic sleep staging from raw single-channel EEG. The core idea is to learn features with CNN layers and then model sleep-stage transition patterns with recurrent layers.

Project relevance:

- Supports the notebook's CNN/LSTM baseline.
- Shows why temporal context matters for sleep staging.
- Useful as a conceptual baseline, even though the project signals are wearable/IoT rather than EEG-only.

Paper: <https://arxiv.org/abs/1703.04046>

### TinySleepNet

TinySleepNet is a smaller and more efficient model for raw single-channel EEG sleep staging. It was designed to reduce overengineering and improve generalization across multiple datasets.

Project relevance:

- Good reminder that smaller models may generalize better on limited, imbalanced data.
- Useful if the competition dataset is small or noisy.
- Supports using a compact TCN or compact CNN instead of a very large network.

Paper/code references:

- <https://pubmed.ncbi.nlm.nih.gov/33018069/>
- <https://github.com/akaraspt/tinysleepnet>

### U-Time

U-Time is a fully convolutional U-Net-style architecture for physiological time-series segmentation. It maps long input sequences to sleep-stage labels without recurrent layers.

Project relevance:

- Strong match for sleep staging as sequence segmentation.
- More robust and easier to train than many recurrent models.
- A useful upgrade from independent-window TCN classification: feed a longer sequence and output labels for multiple windows.

Paper: <https://papers.nips.cc/paper/8692-u-time-a-fully-convolutional-network-for-time-series-segmentation-applied-to-sleep-staging>

### U-Sleep

U-Sleep extends the U-Time idea into a ready-to-use fully convolutional sleep-staging system trained and evaluated at large scale. It accepts flexible EEG/EOG channel combinations and predicts sleep stages efficiently over long recordings.

Project relevance:

- Reinforces that fully convolutional sequence models are state-of-the-art practical systems.
- The architecture idea can be adapted to wearable signals: encoder, decoder, and segment classifier.
- Suggests predicting the full night or many consecutive windows in one forward pass.

Paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/>

### Google Wearable CNN Sleep Staging

Google Research reported a deep CNN for sleep staging from raw PPG and 3D accelerometer signals. The model used wearable-like signals and mapped stages to wake, light sleep, deep sleep, and REM.

Project relevance:

- Highly relevant because the notebook uses non-PSG wearable/IoT signals such as BVP/PPG-like data, accelerometer, temperature, EDA, HR, and IBI.
- Supports using raw or lightly processed wearable signals directly with deep CNNs.
- Shows that pretraining on larger PPG datasets can help.

Paper page: <https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/>

### SleepTransformer and Transformer Variants

Transformer-based sleep staging models use attention to capture long-range temporal context and can provide interpretability or uncertainty estimates. Newer variants also use spectrogram patches and multimodal attention.

Project relevance:

- Useful if there is enough data or if pretraining is possible.
- For small competition datasets, a full Transformer may overfit.
- A small Transformer encoder after CNN/TCN feature extraction is more practical than a large end-to-end Transformer.

References:

- SleepTransformer: <https://arxiv.org/abs/2105.11043>
- SleepViTransformer: <https://www.sciencedirect.com/science/article/pii/S1746809423006365>

### Mamba / State-Space Sleep Staging

Recent work applies Mamba/state-space sequence models to wearable sleep staging without EEG, using multimodal wearable signals such as ECG, accelerometry, temperature, and PPG.

Project relevance:

- Very relevant to IoT/wearable sleep staging.
- Mamba-style models are attractive for long sequences because they scale better than standard attention.
- A practical experiment would replace or augment residual TCN blocks with a small Mamba/state-space sequence block.

Paper: <https://academic.oup.com/sleep/article/49/4/zsag022/8466336>

### SleepFM and SleepGPT Foundation Models

The latest state-of-the-art direction is self-supervised pretraining on large-scale PSG corpora, then adapting learned representations to sleep staging and clinical prediction tasks.

Project relevance:

- Shows the field is moving from task-specific supervised models to foundation models.
- Directly training this from scratch is unrealistic for a small hackathon dataset.
- The practical takeaway is to use self-supervised pretraining or transfer learning if unlabeled recordings are available.

References:

- SleepFM: <https://www.nature.com/articles/s41591-025-04133-4>
- SleepGPT: <https://www.nature.com/articles/s41467-025-67970-4>

## State-of-the-Art Summary

There is no single universal state-of-the-art model for every sleep-staging setting because performance depends heavily on signal type:

- PSG with EEG/EOG/EMG: U-Sleep, Transformer variants, and foundation models are the strongest modern directions.
- Single-channel EEG: DeepSleepNet, TinySleepNet, IITNet, and modern compact CNN/attention models are common references.
- Wearable/IoT without EEG: deep CNNs, residual temporal models, multimodal fusion, and newer Mamba/state-space models are most relevant.
- Large unlabeled datasets: self-supervised foundation models such as SleepFM and SleepGPT are the current research frontier.

For this repository's notebook, the best state-of-the-art-inspired implementation path is:

1. Keep the residual TCN as the baseline.
2. Add raw-signal and FFT branches instead of using FFT only.
3. Train on sequences of windows, not isolated windows.
4. Add post-processing or sequence smoothing for sleep-stage transitions.
5. Try a U-Time-style encoder-decoder or compact Mamba block if implementation time allows.

## Concrete Model Proposal

### Model: Dual-Branch Residual Temporal Network

Input:

- `X_raw`: shape `(batch, windows, timesteps, channels)` or flattened as `(batch, long_timesteps, channels)`.
- `X_fft`: FFT magnitude or log-power features for the same windows.

Architecture:

1. Raw branch:
   - `Conv1D`
   - batch normalization
   - residual TCN blocks

2. Frequency branch:
   - `Conv1D`
   - batch normalization
   - residual TCN blocks

3. Fusion:
   - concatenate raw and FFT embeddings
   - dropout
   - temporal context block: TCN, BiLSTM, small Transformer, or Mamba

4. Output:
   - dense classifier
   - softmax over `W`, `R`, `N1`, `N2`, `N3`

Training:

- Use grouped train/validation split by subject/file to avoid leakage.
- Use weighted cross-entropy or focal loss.
- Monitor macro F1 or weighted F1.
- Reuse the training scaler for test inference.
- Save the best model by validation F1, not validation loss only.

## Implementation Notes for the Existing Notebook

- Reuse the training `StandardScaler` for test data. The current notebook fits a new scaler on test data, which can shift distributions.
- Add a validation strategy by file or subject, not random windows, if file identities correspond to subjects or sessions.
- Add confusion matrix and per-class F1, because N1 and REM are often harder than W/N2/N3.
- Try log-scaled FFT magnitude: `np.log1p(np.abs(np.fft.rfft(x)))`.
- Preserve temporal order when batching if using sequence-level context.
- Consider merging N1/N2 only if the competition label scheme allows it; otherwise keep all five classes.

## References

- DeepSleepNet: <https://arxiv.org/abs/1703.04046>
- TinySleepNet: <https://pubmed.ncbi.nlm.nih.gov/33018069/>
- U-Time: <https://papers.nips.cc/paper/8692-u-time-a-fully-convolutional-network-for-time-series-segmentation-applied-to-sleep-staging>
- U-Sleep: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/>
- Google wearable sleep staging CNN: <https://research.google/pubs/sleep-staging-classification-from-wearable-signals-using-deep-learning/>
- SleepTransformer: <https://arxiv.org/abs/2105.11043>
- SleepViTransformer: <https://www.sciencedirect.com/science/article/pii/S1746809423006365>
- Mamba wearable sleep staging without EEG: <https://academic.oup.com/sleep/article/49/4/zsag022/8466336>
- SleepFM: <https://www.nature.com/articles/s41591-025-04133-4>
- SleepGPT: <https://www.nature.com/articles/s41467-025-67970-4>
