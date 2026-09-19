# Software and Model-Artifact Validation

## Scope

This document describes validation checks for web application version `1.1.2` using locked model version `1.0.0`. The checks evaluate implementation consistency and input-handling behavior. They do not constitute prospective clinical validation or external validation in an independent population.

## Locked artifacts

The application loads the following version-controlled artifacts:

- `best_sarcopenia_lr_model2.pkl`
- `best_features_model2.pkl`
- `best_threshold_model2.pkl`
- `model_metadata.pkl`

The expected model is a scikit-learn logistic-regression pipeline with nine ordered predictors:

```text
age, EQ5D, pa_aerobic, is_allownc, Chew_Diff,
Prot_Deficiency, N_EN, HE_wc, obe_4class
```

The locked screening threshold is `0.082884`. A model score greater than or equal to this value is classified as `Screen positive`.

## Automated checks

Run all checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The test suite checks that:

1. All locked model artifacts load successfully.
2. The stored algorithm, feature set, feature order, and threshold match the declared release.
3. The classification boundary uses `score >= threshold`.
4. Each screening classification receives the correct `recommended_action`.
5. Records younger than 65 years are rejected.
6. Age values above 80 are rejected with an instruction to code ages 80 years or older as 80.
7. Invalid binary and obesity-class codes are rejected.
8. Missing non-age predictors generate an imputation warning.
9. The synthetic example input reproduces the version-controlled expected output within the specified numerical tolerance.

## Synthetic validation example

`examples/sample_input.csv` contains synthetic records created only for software testing and interface demonstration. It contains no participant or clinical-study data.

`examples/expected_output.csv` stores the expected model scores, classifications, and recommended actions for those synthetic records. Prediction parity is evaluated with an absolute tolerance of `1e-10`.

## Temporal validation performance

The locked final model was evaluated using the 2024 KNHANES temporal validation dataset.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| ROC-AUC | 0.837 | 0.792-0.877 |
| PR-AUC | 0.324 | 0.251-0.425 |
| Accuracy | 0.767 | 0.742-0.792 |
| Balanced accuracy | 0.776 | 0.729-0.822 |
| Sensitivity | 0.786 | 0.702-0.869 |
| Specificity | 0.766 | 0.740-0.791 |
| PPV | 0.208 | 0.183-0.235 |
| NPV | 0.979 | 0.970-0.987 |
| F1 score | 0.329 | 0.292-0.369 |
| Brier score | 0.058 | 0.054-0.061 |

ROC-AUC, area under the receiver operating characteristic curve; PR-AUC, area under the precision-recall curve; PPV, positive predictive value; NPV, negative predictive value; CI, confidence interval.

## Input validation implemented in the application

The application checks required columns, numeric conversion, age eligibility (including the requirement to code ages 80 years or older as 80), binary codes, obesity-class codes, missing participant codes, duplicate participant codes, and missing predictor values. Missing non-age model inputs are handled by the imputation steps stored inside the locked preprocessing pipeline, and the application displays a warning before prediction.

## Reproducibility controls

- Python and runtime package versions are pinned.
- The model, preprocessing pipeline, feature order, and threshold are stored as locked artifacts.
- Synthetic expected outputs provide a regression check for future application changes.
- Model and application versions are reported separately in the interface.
- The illustrated PDF user manual is version controlled and is available from the **How to Use** tab.

## Validation record

On 4 September 2026, the automated suite completed successfully under Python `3.12.14`: 7 tests run, 7 passed, 0 failed. The Streamlit application compiled successfully, started locally, and returned `ok` from `/_stcore/health`. The batch-screening workflow was exercised with `examples/sample_input.csv`, including the updated age top-coding example, and the resulting interface was reviewed in a desktop viewport before the supplementary screenshots were retained. For web application version `1.1.2`, the **How to Use** tab displayed the updated field definitions and the user-manual download control, and the regenerated PDF was checked by text extraction and visual review of all pages.

## Limitations of this validation

- Automated tests establish software behavior, not clinical effectiveness.
- The 2024 evaluation is temporal validation within the KNHANES survey framework, not independent external validation.
- Pickle-based model files should be loaded only from this trusted repository release.
- Local calibration and workflow validation are required before operational implementation in a new setting.
