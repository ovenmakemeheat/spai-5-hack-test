# Notebook Summary: Heart Disease Prediction

Source notebooks:

- `notebook/heart_disease/ref/bronze_heart_disease_prediction_exploration_row25_drive.ipynb`
- `notebook/heart_disease/ref/silver_heart_disease_prediction_baseline_row25.ipynb`
- `notebook/heart_disease/ref/gold_heart_disease_f2_ensemble_row17.ipynb`

## Objective

The notebooks build a tabular binary classification pipeline for predicting whether a patient has a history of heart disease or heart attack. The final output is a Kaggle-style submission with:

```text
ID, History of HeartDisease or Attack
```

The competition metric is F2 score. Because F2 weights recall more heavily than precision, the notebooks focus on finding more positive heart disease cases instead of optimizing only accuracy or a default 0.5 probability threshold.

## Notebook Progression

The references evolve through three stages:

1. `bronze`: exploratory analysis, missing-value investigation, feature selection experiments, Optuna trials, AutoML experiments, and early XGBoost/LightGBM/H2O modeling.
2. `silver`: a cleaner baseline using manual feature engineering, 5-fold cross-validation, LightGBM, XGBoost, CatBoost, probability averaging, and F2 threshold tuning.
3. `gold`: the final F2-focused technique using structured encoding, missing-value flags, medical feature engineering, RobustScaler, BorderlineSMOTE, per-fold F2 threshold optimization, grid search, and top-3 soft voting.

## Data Preparation

The final notebook first renames the original long feature names into compact snake_case names. Important examples include:

- `History of HeartDisease or Attack` -> `target`
- `High Blood Pressure` -> `high_bp`
- `Told High Cholesterol` -> `high_chol`
- `Body Mass Index` -> `bmi`
- `General Health` -> `gen_health`
- `Vegetable or Fruit Intake (1+ per Day)` -> `fruit_veg`

Categorical text values are converted to numeric values before imputation:

- Binary `Yes`/`No` columns become `1`/`0`.
- `Sex` becomes `Male=1`, `Female=0`.
- `General Health` becomes an ordinal scale from `Excellent=1` to `Very Poor=5`.
- `Education Level` and `Income Level` become ordered numeric scales.
- Rows with missing target labels are dropped because labels are not imputed.

## Missing Data Handling

The final technique uses different imputation strategies by feature type:

| Feature type | Strategy | Reason |
|---|---|---|
| Binary features | Most frequent value | Preserves binary yes/no semantics |
| Ordinal features | Median | Robust for ordered categories |
| BMI | `KNNImputer(n_neighbors=5)` | Uses nearby observations for continuous BMI |
| Remaining numeric columns | Median | Simple fallback for any leftover missing values |

For columns with more than 1 percent missing values, the pipeline also adds a `{column}_was_missing` indicator. This keeps missingness as a potential signal, which can matter in health survey data.

## Feature Engineering

The notebooks rely heavily on domain-informed tabular features. The final feature set includes:

### BMI Features

BMI is transformed into clinical risk categories:

- Underweight: `bmi < 18.5`
- Normal: `18.5 <= bmi < 25`
- Overweight: `25 <= bmi < 30`
- Obese: `bmi >= 30`
- Severely obese: `bmi >= 35`

### Age Features

Age is converted into risk bands:

- Young: `< 35`
- Middle: `35-54`
- Senior: `55-69`
- Elderly: `>= 70`
- High risk: `>= 55`

The final notebook also adds continuous age transforms such as `age_log`, `age_sq`, and `bmi_age`.

### Clinical Risk Scores

The gold notebook builds composite scores that summarize known cardiovascular risk factors:

- `framingham_score`: simplified count using age, sex, high blood pressure, high cholesterol, smoking, and diabetes.
- `metabolic_syndrome`: positive when obesity, high blood pressure, high cholesterol, and diabetes meet a threshold.
- `lifestyle_risk_score`: smoking, heavy alcohol, lack of physical activity, and low fruit/vegetable intake.
- `cardiac_comorbidity_score`: stroke, diabetes, high blood pressure, and high cholesterol count.
- `cvd_burden_score`: combines Framingham risk, metabolic syndrome, lifestyle risk, stroke, walking difficulty, and poor general health.

### Interaction Features

The final model uses interaction terms to represent co-occurring risks:

- `hypertension_diabetes`
- `hypertension_high_chol`
- `smoker_diabetes`
- `obese_hypertension`
- `obese_diabetes`
- `stroke_age_risk`
- `male_senior`
- `triple_risk`
- `triple_metabolic`
- `elderly_obese`
- `htn_no_access`
- `age_x_framingham`

The silver baseline used a smaller version of the same idea with `risk_factor_count`, `high_bp_cholesterol`, `stroke_diabetes`, `lifestyle_score`, `health_access`, `age_bmi`, and `age_risk`.

## Imbalance Strategy

Heart disease positives are the minority class, so the notebooks use several imbalance techniques:

- `scale_pos_weight` in XGBoost.
- `is_unbalance=True` in LightGBM.
- Balanced class handling in CatBoost or logistic regression where applicable.
- `BorderlineSMOTE` in the final notebook.

The final workflow applies `BorderlineSMOTE(kind='borderline-1')` only inside each training fold, after scaling. Validation folds are not resampled, which avoids leaking synthetic data into validation.

## Cross-Validation

The final notebook evaluates models with 5-fold `StratifiedKFold`:

1. Split data into stratified train and validation folds.
2. Fit `RobustScaler` on the training fold.
3. Transform train and validation features.
4. Apply `BorderlineSMOTE` to the scaled training fold only.
5. Train each model on the resampled training data.
6. Predict validation probabilities.
7. Tune the classification threshold for F2 on that fold.

The main reported metrics are:

- F2 score, the primary target metric.
- ROC-AUC.
- PR-AUC.
- Mean and standard deviation across folds.
- Average optimal threshold.

## Models Used

The silver notebook trains three gradient boosting models:

- LightGBM
- XGBoost
- CatBoost

The gold notebook focuses on:

- Logistic regression with balanced class weights.
- XGBoost with `scale_pos_weight`.
- LightGBM with `is_unbalance=True`.

The bronze notebook also experiments with:

- Random forest.
- CatBoost.
- Optuna-based feature and model selection.
- AutoGluon tabular modeling.
- H2O AutoML.

## F2 Threshold Tuning

The most important modeling technique is threshold tuning for F2.

Instead of using:

```text
probability >= 0.5
```

the final notebook searches thresholds from `0.05` to `0.845` in `0.005` increments for each validation fold. The threshold that maximizes F2 is saved for each fold, and the average threshold is used later for inference.

This matters because F2 gives recall more weight than precision. In an imbalanced medical prediction task, the best F2 threshold is often lower than `0.5`.

## Hyperparameter Optimization

The final notebook performs manual grid search for XGBoost and LightGBM using 3-fold cross-validation. Each candidate is evaluated with the same F2-driven process:

- Robust scaling.
- BorderlineSMOTE on the training fold.
- Probability prediction on the validation fold.
- Threshold search.
- Mean F2 comparison.

The searched XGBoost grid includes:

- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`

The searched LightGBM grid includes:

- `num_leaves`
- `learning_rate`
- `min_child_samples`
- `subsample`

Best grid-search parameters are injected into the final ensemble models and retrained with more estimators.

## Ensemble Strategy

The final model uses soft voting:

1. Sort cross-validation results by mean F2.
2. Select the top three models.
3. Train those models on the full scaled and BorderlineSMOTE-resampled training data.
4. Average their predicted probabilities on the test set.
5. Convert probabilities to labels using the ensemble threshold, defined as the mean optimal threshold of the selected models.

Soft voting is used because averaging probabilities can reduce model variance and usually produces more stable predictions than a single model.

## Evaluation and Interpretation

The notebooks include several interpretation and diagnostic steps:

- Target distribution checks to confirm class imbalance.
- Missing-value summaries.
- Correlation heatmaps.
- Heart disease rate by age group and categorical risk factors.
- LightGBM feature importance in the silver notebook.
- Final model feature importances or logistic-regression coefficient magnitudes in the gold notebook.
- Confusion matrix and classification report for the silver ensemble.
- Probability histogram with the selected F2 threshold for the final test predictions.

## Final Submission Flow

The final inference process is:

1. Load `test.csv`.
2. Apply the same column mapping.
3. Encode categorical values.
4. Impute missing values.
5. Recreate engineered features.
6. Align test columns to the training feature columns.
7. Apply the fitted `RobustScaler`.
8. Predict probabilities from the top-3 trained models.
9. Average probabilities.
10. Apply the F2-optimized threshold.
11. Convert `1/0` predictions back to `Yes`/`No`.
12. Save `heart_disease_submission_esum.csv`.

## Key Takeaways

The strongest technique is not a single model. It is the full F2-optimized pipeline:

- Domain-informed cardiovascular feature engineering.
- Missingness indicators for health survey data.
- Fold-safe scaling and BorderlineSMOTE.
- Class-imbalance-aware tree models.
- Per-fold threshold tuning for F2.
- Grid search on the models most likely to perform well.
- Soft-voting ensemble of the top models by cross-validated F2.

This design directly matches the competition objective: maximize recall-sensitive F2 while keeping validation honest through stratified cross-validation and train-fold-only resampling.
