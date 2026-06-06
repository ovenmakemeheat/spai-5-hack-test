# Notebook Summary: Sleep Stage Classification

Source notebook: `notebook/sleep_stage/ref/final-hack-iot-tcnlstm-cnn-superai-ss5.ipynb`

## Objective

The notebook builds a sleep-stage classification pipeline for multivariate IoT signal data. It trains a neural network to classify each 480-sample time window into one of five sleep stages:

- `W`
- `R`
- `N1`
- `N2`
- `N3`

The final output is a Kaggle-style submission file named `sub_residual.csv`.

## Data Pipeline

Training CSV files are loaded from:

```text
/kaggle/input/spai-signal-sleep-staging-classification/train/train
```

Each file contains continuous sensor signals and, for training data, a `Sleep_Stage` label column. The pipeline maps labels to integers:

```text
W -> 0
R -> 1
N1 -> 2
N2 -> 3
N3 -> 4
```

The data is split into fixed-size windows of 480 rows. For each window:

1. The raw signal values are reshaped into `(num_windows, 480, num_features)`.
2. Fast Fourier Transform is applied along the time axis.
3. The absolute FFT magnitude is used as the model input.
4. The window label is chosen by majority vote over the 480 row-level labels.

After preprocessing, features are normalized with `StandardScaler`.

## Signals Used

The plotting cell assumes eight input channels:

- `bvp`
- `acc_x`
- `acc_y`
- `acc_z`
- `temp`
- `eda`
- `hr`
- `ibi`

The notebook includes a visualization step that randomly selects one processed window and plots all eight signal channels.

## Metric

The notebook defines a custom weighted F1 metric for Keras. It:

1. Converts model probabilities to predicted class labels with `argmax`.
2. Computes precision, recall, and F1 per class.
3. Weights each class F1 by the number of true examples in that class.
4. Returns the weighted average.

This metric is useful because sleep-stage classes are often imbalanced.

## Model Ideas Used

### 1. FFT-Based Feature Representation

Instead of training directly on raw time-domain signals, the notebook transforms each signal window with FFT and uses frequency-domain magnitudes. This can help expose periodic patterns in physiological signals such as heart rate, movement, EDA, and BVP.

### 2. CNN + LSTM Baseline

The notebook defines a sequential CNN-LSTM model:

- `Conv1D` layers extract local temporal/frequency patterns.
- `MaxPooling1D` reduces sequence length.
- `Dropout` regularizes the model.
- Two `LSTM` layers model longer-range sequential dependencies.
- Dense layers map learned features to five sleep-stage classes.

This model is compiled with sparse categorical cross-entropy and the custom weighted F1 metric.

### 3. Basic TCN Experiment

A basic Temporal Convolutional Network idea is included as commented code. It uses:

- Dilated `Conv1D` layers with dilation rates `1`, `2`, and `4`.
- Batch normalization.
- Dropout.
- Global average pooling.

The idea is to capture temporal context with dilated convolutions instead of recurrent layers.

### 4. Improved Residual TCN

The main final model is an improved TCN with residual blocks. It uses:

- Initial `Conv1D` projection.
- Batch normalization.
- Dropout.
- Residual blocks with dilation rates `2`, `4`, and `8`.
- L2 regularization.
- Global average pooling.
- Softmax output for five classes.

Each residual block applies two dilated convolution layers and adds the original shortcut back to the transformed output. This helps deeper convolutional temporal models train more reliably.

### 5. Training Strategy

The improved TCN is trained with:

- `Adam` optimizer.
- Learning rate `0.01`.
- Sparse categorical cross-entropy.
- Validation split of `0.2`.
- Batch size `64`.
- Up to `200` epochs.
- Early stopping on `val_f1_weighted`.
- Learning-rate reduction on plateau.

The callbacks monitor weighted F1 in `max` mode, which matches the goal of improving the metric.

## Prediction and Submission

Test CSV files are loaded from:

```text
/kaggle/input/spai-signal-sleep-staging-classification/test_segment/test_segment
```

The same FFT window preprocessing is applied to test files. The trained model predicts class probabilities, then `argmax` converts probabilities to class IDs. These IDs are mapped back to sleep-stage labels and saved as:

```text
sub_residual.csv
```

The submission contains:

- `id`: test file basename without `.csv`
- `labels`: predicted sleep-stage label

## Important Notes

- The notebook fits a new `StandardScaler` on the test data instead of reusing the training scaler. For a cleaner inference pipeline, the scaler fitted on training data should be reused for test data.
- The CNN-LSTM model uses a high learning rate of `0.05`, which may be unstable.
- The basic TCN code is commented out and appears to be exploratory.
- The final active training path is the improved residual TCN.
- The notebook uses Kaggle-specific input paths, so paths must be adjusted before running locally.

