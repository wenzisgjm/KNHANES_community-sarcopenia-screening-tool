# Manuscript analysis code

`Supplementary_Code.py` is the executable manuscript analysis script. It runs the primary KNHANES 2022 model development and 2024 temporal validation, then the expanded-sample LR Model 2 sensitivity analysis. This directory contains code only; no participant records or analysis results are included.

The script reads two supplied analytic CSV files directly. It does not perform upstream cohort construction, variable recoding, data cleaning, imputation, or de-identification. It verifies that the supplied primary CSV satisfies the documented final complete-case conditions, outcome rule, and year-specific manuscript counts; this audit does not reconstruct exclusions from raw KNHANES records. Input variables and coding must match the manuscript and Supplementary Information. The primary CSV contains the 36 candidate predictors, outcome (`Sarcopenia`), outcome components (`MAX_grip`, `ASMI`), `ID`, and `year`. The expanded CSV contains the same outcome and identification columns, `sex`, and the nine fixed LR Model 2 predictors; laboratory predictors are not required in this file. Both files contain 2022 and 2024 records.

## Run

Use Python 3.13.15 for the manuscript analysis environment recorded in the saved primary-analysis notebook output. That output reports scikit-learn 1.6.1 and SHAP 0.52.0. The other pinned packages below are reproduction dependencies, not a captured package freeze of that notebook session. The web application's runtime and model-loading dependencies are maintained separately at the repository root.

From the repository root:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/Supplementary_Code.py --data /path/to/primary.csv --sensitivity-data /path/to/expanded.csv --output-dir /path/to/results
```

If downloading individual files, keep `Supplementary_Code.py` and `revision_figures.py` together in the same directory (and obtain `requirements.txt` for the dependencies). The existing manuscript-model check still stops the analysis if the 2022 selection is not LR Model 2; the age-specific figure is defined only for that locked model.

The primary analysis outputs are written to the selected result directory. Expanded-sample outputs are written to its `sensitivity_analysis/` subdirectory. These outputs can contain participant identifiers and predictions, so choose a local result directory appropriate for the input data. Result CSVs are not included in this repository.

## Analysis scope

The primary comparison uses the fixed 11-variable pool established in the full 2022 development cohort before outer cross-validation. It compares three measurement tiers and three classifiers. The within-fold selection analysis reselects seven of 32 low-cost variables inside training folds; it is a sensitivity analysis and does not reconstruct the original clinical decisions that formed the full candidate pool. Model selection and threshold locking use 2022 data. The fixed model and threshold are then applied to 2024 data.

Other analyses include Supplementary Table S15 age-adjusted pooled comparisons, all 36 paired candidate AUC comparisons with one Holm adjustment family, the age-only benchmark, the seven-variable model without waist circumference and obesity class, screening and referral burden, calibration, decision curves, outcome-component summaries, and SHAP summaries. The expanded-sample sensitivity analysis keeps the nine LR Model 2 predictors, refits on expanded 2022 data, locks a separate threshold, and evaluates both that threshold and the primary threshold on expanded 2024 data.

Bootstrap confidence intervals use 5,000 outcome-stratified resamples of fixed predictions. They are conditional on the fitted model and observed class counts; model selection and fitting are not repeated in bootstrap samples. The paired 2024 comparisons of LR Model 2 with the age-only benchmark and with the seven-variable model use the same resampling indices for every model in each replicate. Their ROC-AUC and PR-AUC differences and percentile intervals are written to `paired_temporal_validation_auc_comparisons.csv`; the pre-existing anthropometry-removal output is retained for backward compatibility. The analysis script has synthetic-data checks, and the revised figure helpers were also checked against the existing study fold results and locked validation predictions; reproducing the complete analysis requires the study CSVs. The screening application at the repository root is versioned separately from this analysis script.

## Revised manuscript figures

`revision_figures.py` reproduces the two final notebook layouts without fitting models, changing predictions, or selecting thresholds. `Supplementary_Code.py` calls it during the primary analysis. The fold plot uses the 45 original held-out fold results, reconciled against the corresponding OOF predictions, in the fixed order LR Models 1–3, RF Models 1–3, and GB Models 1–3. Its two panels show ROC-AUC and PR-AUC (average precision): five points per model, observed minimum–maximum ranges, and arithmetic means. The ranges are descriptive, not confidence intervals or standard-deviation error bars.

The compact age ROC plot uses only the 2024 validation participants and the same locked LR Model 2 and 2022 OOF threshold in both groups (65–74 and ≥75 years). It checks prediction alignment against the locked pipeline. Its 95% percentile intervals use 5,000 outcome-stratified bootstrap resamples per group, with one `numpy.random.default_rng(42)` stream: ages 65–74 first, then ≥75; cases are sampled before non-cases in each replicate. The model and threshold remain fixed. This subgroup seed differs from the existing overall validation seed; neither changes model fitting.

The following files are written under the selected output directory:

| Location | Output |
| --- | --- |
| `primary_complete_case_audit.csv` | Final analytic-input completeness, outcome-rule, and manuscript-count checks by year |
| `Table_S15_age_adjusted_comparisons.csv` | Age-adjusted EQ-5D mean difference and aerobic-activity odds ratio in the pooled primary sample |
| `reviewer_figures/candidate_model_fold_AUC_AP_distribution_2022.png` and `.pdf` | Two-panel fold figure; PNG at 600 dpi |
| `reviewer_figures/candidate_model_fold_AUC_AP_plot_data_2022.csv` | The 45 plotted fold rows with validation/event counts |
| `reviewer_figures/candidate_model_fold_AUC_AP_summary_2022.csv` | Means, sample SDs, minima, and maxima for both metrics |
| `candidate_model_performance.png` | Backward-compatible filename for the same revised fold figure |
| `reviewer_figures/ROC_age_groups_2024_LR_Model_2.png` and `.pdf` | Compact age-group ROC figure; PNG at 600 dpi |
| `reviewer_figures/ROC_age_groups_2024_LR_Model_2_metrics.csv` | Subgroup counts, AUC, sensitivity, specificity, bootstrap intervals, and fixed threshold |
| `reviewer_figures/ROC_age_groups_2024_LR_Model_2_coordinates.csv` | Empirical subgroup ROC coordinates and thresholds |

These outputs stay in the selected local result directory; they are not included in this code repository.

