# Manuscript analysis code

`Supplementary_Code.py` is the executable source corresponding to the manuscript's Supplementary Code PDF. It runs the primary KNHANES 2022 model development and 2024 temporal validation, then the expanded-sample LR Model 2 sensitivity analysis. This directory contains code only; no participant records or analysis results are included.

The script reads two supplied analytic CSV files directly. It does not perform upstream cohort construction, variable recoding, data cleaning, imputation, or de-identification. Input variables and coding must match the manuscript and Supplementary Information. The primary CSV contains the 36 candidate predictors, outcome (`Sarcopenia`), outcome components (`MAX_grip`, `ASMI`), `ID`, and `year`. The expanded CSV contains the same outcome and identification columns, `sex`, and the nine fixed LR Model 2 predictors; laboratory predictors are not required in this file. Both files contain 2022 and 2024 records.

## Run

Use Python 3.12.14. From the repository root:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/Supplementary_Code.py --data /path/to/primary.csv --sensitivity-data /path/to/expanded.csv --output-dir /path/to/results
```

The primary analysis outputs are written to the selected result directory. Expanded-sample outputs are written to its `sensitivity_analysis/` subdirectory. These outputs can contain participant identifiers and predictions, so choose a local result directory appropriate for the input data. Result CSVs are not included in this repository.

## Analysis scope

The primary comparison uses the fixed 11-variable pool established in the full 2022 development cohort before outer cross-validation. It compares three measurement tiers and three classifiers. The within-fold selection analysis reselects seven of 32 low-cost variables inside training folds; it is a sensitivity analysis and does not reconstruct the original clinical decisions that formed the full candidate pool. Model selection and threshold locking use 2022 data. The fixed model and threshold are then applied to 2024 data.

Other analyses include all 36 paired candidate AUC comparisons with one Holm adjustment family, the age-only benchmark, the seven-variable model without waist circumference and obesity class, screening and referral burden, calibration, decision curves, outcome-component summaries, and SHAP summaries. The expanded-sample sensitivity analysis keeps the nine LR Model 2 predictors, refits on expanded 2022 data, locks a separate threshold, and evaluates both that threshold and the primary threshold on expanded 2024 data.

Bootstrap confidence intervals use 5,000 outcome-stratified resamples of fixed predictions. They are conditional on the fitted model and observed class counts; model selection and fitting are not repeated in bootstrap samples. The submitted code was checked with synthetic data, but manuscript results must be reproduced from the study CSVs. The screening application at the repository root is versioned separately from this analysis script.
