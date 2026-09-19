#!/usr/bin/env python3
"""KNHANES sarcopenia screening: primary and sensitivity analyses.

Read the two supplied analytic CSVs directly; use their existing variable coding.
Primary CV is conditional on the fixed 11-variable pool determined before CV.
The within-fold analysis reselects seven of 32 low-cost variables only.
Develop and lock models/thresholds in 2022; evaluate fixed predictions in 2024.
Bootstrap intervals are conditional on fitted models and observed class counts.

Run: python Supplementary_Code.py --data primary.csv
     --sensitivity-data expanded.csv --output-dir results
"""

from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.stats import chi2_contingency, spearmanr, ttest_ind
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 5000
OUTCOME = "Sarcopenia"

QUESTIONNAIRE_FEATURES = [
    "age",
    "EQ5D",
    "pa_aerobic",
    "is_allownc",
    "Chew_Diff",
    "Prot_Deficiency",
    "N_EN",
]
ANTHROPOMETRIC_FEATURES = ["HE_wc", "obe_4class"]
LABORATORY_FEATURES = ["HE_HB", "HE_hsCRP"]

LOW_COST_CANDIDATES = [
    "age", "sex", "town_t", "EQ5D", "incm_3class", "edu_3class",
    "occp_type", "live_alone", "is_allownc", "Housing_Risk",
    "Self_Health", "HTN_Diag", "Dyslip_Diag", "DM_Diag",
    "Asthma_Diag", "Atopy_Diag", "Rhinitis_Diag", "Kidney_Diag",
    "Activity_Limit", "Ever_hospital", "Ever_drink", "PHQ9_depress",
    "GAD_anxious", "mh_stress", "pa_aerobic", "Resistance_Work",
    "Chew_Diff", "Comorbidity", "sedentary", "ever_smoke",
    "Prot_Deficiency", "N_EN",
]
ANTHROPOMETRIC_CANDIDATES = ["HE_wc", "obe_4class"]
LABORATORY_CANDIDATES = ["HE_HB", "HE_hsCRP"]
ALL_36_CANDIDATES = (
    LOW_COST_CANDIDATES + ANTHROPOMETRIC_CANDIDATES + LABORATORY_CANDIDATES
)

FEATURE_SETS = {
    "Model_1": QUESTIONNAIRE_FEATURES,
    "Model_2": QUESTIONNAIRE_FEATURES + ANTHROPOMETRIC_FEATURES,
    "Model_3": (
        QUESTIONNAIRE_FEATURES
        + ANTHROPOMETRIC_FEATURES
        + LABORATORY_FEATURES
    ),
}
CONTINUOUS_FEATURES = {"age", "EQ5D", "N_EN", "HE_wc", "HE_HB", "HE_hsCRP"}


def parse_arguments():
    parser = ArgumentParser(
        description="Develop the model in KNHANES 2022 and validate it in 2024."
    )
    parser.add_argument("--data", required=True, help="Final analytic CSV file.")
    parser.add_argument(
        "--sensitivity-data",
        required=True,
        help=(
            "Expanded CSV containing complete sarcopenia outcome components "
             "and the nine LR Model 2 predictors."
        ),
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory for tables and figures."
    )
    return parser.parse_args()


def descriptive_comparison(first, second, features, welch=False):
    """Raw group summaries; SMD direction is group B minus group A."""
    rows = []
    for feature in features:
        a, b = first[feature], second[feature]
        if feature in CONTINUOUS_FEATURES:
            denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows.append({
                "Variable": feature, "Level": "continuous",
                "A_n": len(a), "B_n": len(b), "A_mean": a.mean(), "B_mean": b.mean(),
                "A_SD": a.std(ddof=1), "B_SD": b.std(ddof=1),
                "SMD_B_minus_A": (b.mean() - a.mean()) / denominator if denominator else np.nan,
                "P": ttest_ind(a, b, equal_var=not welch).pvalue,
                "Test": "Welch t" if welch else "Pooled independent t",
            })
        else:
            levels = sorted(set(a) | set(b))
            counts = np.array([[a.eq(level).sum(), b.eq(level).sum()] for level in levels])
            p = chi2_contingency(counts, correction=welch)[1] if len(levels) > 1 else np.nan
            for level, (na, nb) in zip(levels, counts):
                pa, pb = na / len(a), nb / len(b)
                denominator = np.sqrt((pa * (1-pa) + pb * (1-pb)) / 2)
                # One non-reference SMD for binary variables; level-specific for multicategory.
                report_smd = len(levels) != 2 or level == (2 if feature == "sex" else 1)
                rows.append({
                    "Variable": feature, "Level": level, "A_n": len(a), "B_n": len(b),
                    "A_count": na, "B_count": nb, "A_proportion": pa, "B_proportion": pb,
                    "SMD_B_minus_A": (pb-pa) / denominator
                    if report_smd and denominator else np.nan,
                    "P": p, "Test": "Chi-square (Yates for 2x2)" if welch else "Pearson chi-square",
                })
    return pd.DataFrame(rows)


def descriptive_outputs(data, output_dir):
    baseline = descriptive_comparison(
        data.loc[data[OUTCOME].eq(0)], data.loc[data[OUTCOME].eq(1)],
        ["sex", *FEATURE_SETS["Model_3"]],
    )
    baseline.to_csv(output_dir / "baseline_by_outcome.csv", index=False)
    low_strength = data["MAX_grip"] < np.where(data["sex"].eq(1), 28, 18)
    low_mass = data["ASMI"] < np.where(data["sex"].eq(1), 7.0, 5.7)
    phenotypes = np.select(
        [low_strength & low_mass, low_strength & ~low_mass, ~low_strength & low_mass],
        ["Low_strength_and_mass", "Low_strength_only", "Low_mass_only"], default="Neither",
    )
    rows = []
    for year, mask in [("All", np.ones(len(data), dtype=bool)),
                       (2022, data["year"].eq(2022)), (2024, data["year"].eq(2024))]:
        for phenotype in ["Neither", "Low_strength_only", "Low_mass_only", "Low_strength_and_mass"]:
            n = int(np.sum((phenotypes == phenotype) & mask))
            rows.append({"Year": year, "Phenotype": phenotype, "n": n,
                         "Denominator": int(np.sum(mask)), "Proportion": n / np.sum(mask)})
    pd.DataFrame(rows).to_csv(output_dir / "outcome_component_phenotypes.csv", index=False)
    anemia = data["HE_HB"] < np.where(data["sex"].eq(1), 13, 12)
    anemia_frame = data.assign(Anemia=anemia.astype(int))
    descriptive_comparison(
        anemia_frame.loc[data[OUTCOME].eq(0)], anemia_frame.loc[data[OUTCOME].eq(1)], ["Anemia"]
    ).to_csv(output_dir / "descriptive_anemia_by_outcome.csv", index=False)


def fit_development_benchmark(development_data, features):
    y = development_data[OUTCOME].to_numpy(dtype=int)
    pipeline = make_pipeline(features, candidate_algorithms()["LogisticRegression"])
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    probability = cross_val_predict(
        pipeline, development_data[features], y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = select_youden_threshold(y, probability)
    return pipeline.fit(development_data[features], y), threshold


def screening_burden(y, probability, threshold, label):
    metrics = calculate_metrics(y, probability, threshold)
    n, tp, fp, fn = len(y), metrics["TP"], metrics["FP"], metrics["FN"]
    referrals = tp + fp
    return {
        "Model": label, "Threshold": threshold, "N": n, "Events": int(np.sum(y)),
        "TP": tp, "FP": fp, "TN": metrics["TN"], "FN": fn,
        "Referral_proportion": referrals / n,
        "False_positive_fraction_of_referrals": fp / referrals if referrals else np.nan,
        "True_positives_per_1000": tp / n * 1000,
        "False_positives_per_1000": fp / n * 1000,
        "Missed_cases_per_1000": fn / n * 1000,
        "NNS_model_detected": n / tp if tp else np.inf,
        "NNR": referrals / tp if tp else np.inf,
        "Inverse_prevalence": n / np.sum(y),
    }


def paired_validation_difference(y, primary_probability, reduced_probability):
    rows = []
    metrics = {"ROC_AUC": roc_auc_score, "PR_AUC": average_precision_score}
    distributions = {name: np.empty(N_BOOTSTRAP) for name in metrics}
    for iteration, index in enumerate(
        stratified_bootstrap_indices(y, N_BOOTSTRAP, RANDOM_STATE + 2024)
    ):
        for name, metric in metrics.items():
            distributions[name][iteration] = (
                metric(np.asarray(y)[index], primary_probability[index])
                - metric(np.asarray(y)[index], reduced_probability[index])
            )
    for name, metric in metrics.items():
        lower, upper = np.quantile(distributions[name], [0.025, 0.975])
        rows.append({
            "Metric": name,
            "Difference_LR_Model_2_minus_LR_Model_1": metric(y, primary_probability)
            - metric(y, reduced_probability),
            "CI_lower_95": lower, "CI_upper_95": upper,
        })
    return pd.DataFrame(rows)


def compare_sample_membership(primary, expanded, output_dir):
    keys = ["ID", "year"]
    primary_index = pd.MultiIndex.from_frame(primary[keys])
    expanded_index = pd.MultiIndex.from_frame(expanded[keys])
    is_primary = expanded_index.isin(primary_index)
    comparisons, counts = [], []
    for year in [2022, 2024]:
        year_mask = expanded["year"].eq(year)
        retained = expanded.loc[year_mask & is_primary]
        added = expanded.loc[year_mask & ~is_primary]
        for label, frame in [("Primary", retained), ("Newly_included", added),
                             ("Expanded", expanded.loc[year_mask])]:
            counts.append({"Year": year, "Sample": label, "N": len(frame),
                           "Events": int(frame[OUTCOME].sum()),
                           "Prevalence": frame[OUTCOME].mean()})
        if len(added):
            comparisons.append(descriptive_comparison(
                retained, added, ["sex", OUTCOME, *FEATURE_SETS["Model_2"]], welch=True
            ).assign(Year=year, Group_A="Primary", Group_B="Newly_included"))
    pd.DataFrame(counts).to_csv(output_dir / "annual_sample_counts.csv", index=False)
    if comparisons:
        pd.concat(comparisons).to_csv(output_dir / "primary_vs_newly_included.csv", index=False)


def make_preprocessor(features):
    continuous = [name for name in features if name in CONTINUOUS_FEATURES]
    categorical = [name for name in features if name not in CONTINUOUS_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_algorithms():
    return {
        "LogisticRegression": LogisticRegression(
             C=1.0, max_iter=5000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
             n_estimators=200,
             max_depth=5,
             random_state=RANDOM_STATE,
             n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
             n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def make_pipeline(features, estimator):
    return Pipeline(
        [("preprocess", make_preprocessor(features)), ("model", clone(estimator))]
    )


def aggregate_encoded_importance(preprocessor, original_columns, importances):
    """Aggregate one-hot encoded RF importances back to original variables."""
    continuous = [name for name in original_columns if name in CONTINUOUS_FEATURES]
    categorical = [name for name in original_columns if name not in CONTINUOUS_FEATURES]
    owners = list(continuous)
    if categorical:
        encoder = preprocessor.named_transformers_["categorical"]
        for name, categories in zip(categorical, encoder.categories_):
            owners.extend([name] * len(categories))
    if len(owners) != len(importances):
        raise RuntimeError("Encoded features do not match the RF importance vector.")
    return pd.Series(importances, index=owners).groupby(level=0).sum().sort_values(
        ascending=False, kind="mergesort"
    )


def rank_initial_candidates(development_data):
    """Rank the 36 candidates in the complete 2022 development cohort only."""
    preprocessor = make_preprocessor(ALL_36_CANDIDATES)
    transformed = preprocessor.fit_transform(development_data[ALL_36_CANDIDATES])
    selector = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(transformed, development_data[OUTCOME])
    ranking = aggregate_encoded_importance(
        preprocessor, ALL_36_CANDIDATES, selector.feature_importances_
    ).rename("RF_importance").rename_axis("feature").reset_index()
    ranking.insert(0, "RF_rank", np.arange(1, len(ranking) + 1))
    ranking["selected_in_fixed_11_pool"] = ranking["feature"].isin(
        FEATURE_SETS["Model_3"]
    )
    return ranking


def select_core_features(frame, outcome, seed):
    """Select seven low-cost predictors using training data available in a fold."""
    preprocessor = make_preprocessor(LOW_COST_CANDIDATES)
    transformed = preprocessor.fit_transform(frame[LOW_COST_CANDIDATES])
    selector = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
    ).fit(transformed, outcome)
    ranking = aggregate_encoded_importance(
        preprocessor, LOW_COST_CANDIDATES, selector.feature_importances_
    )
    return list(ranking.index[:7]), ranking


def build_cost_tier_sets(core_features):
    return {
        "Model_1": list(core_features),
        "Model_2": list(core_features) + ANTHROPOMETRIC_CANDIDATES,
        "Model_3": list(core_features)
        + ANTHROPOMETRIC_CANDIDATES
        + LABORATORY_CANDIDATES,
    }


def select_youden_threshold(y_true, probability):
    false_positive_rate, sensitivity, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    false_positive_rate = false_positive_rate[finite]
    sensitivity = sensitivity[finite]
    thresholds = thresholds[finite]
    youden = sensitivity - false_positive_rate
    tied = np.flatnonzero(np.isclose(youden, youden.max()))
    selected = tied[np.argmax(sensitivity[tied])]
    return float(thresholds[selected])


def calculate_metrics(y_true, probability, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "F1_score": f1_score(y_true, predicted, zero_division=0),
        "Brier_score": brier_score_loss(y_true, probability),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def nested_internal_validation(development_data):
    y = development_data[OUTCOME].astype(int).reset_index(drop=True)
    models = candidate_algorithms()
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(development_data, y))
    fold_rows = []
    oof_registry = {}

    for set_name, features in FEATURE_SETS.items():
        X = development_data[features].reset_index(drop=True)
        for model_name, estimator in models.items():
            oof_probability = np.full(len(y), np.nan)

            for fold, (fit_index, test_index) in enumerate(outer_splits, start=1):
                X_fit, X_test = X.iloc[fit_index], X.iloc[test_index]
                y_fit, y_test = y.iloc[fit_index], y.iloc[test_index]
                inner_cv = StratifiedKFold(
                    n_splits=N_SPLITS,
                    shuffle=True,
                    random_state=RANDOM_STATE + 1000 + fold,
                )
                inner_probability = cross_val_predict(
                    make_pipeline(features, estimator),
                    X_fit,
                    y_fit,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=-1,
                )[:, 1]
                fold_threshold = select_youden_threshold(y_fit, inner_probability)

                fitted = make_pipeline(features, estimator).fit(X_fit, y_fit)
                fold_probability = fitted.predict_proba(X_test)[:, 1]
                oof_probability[test_index] = fold_probability
                fold_rows.append(
                    {
                        "feature_set": set_name,
                        "algorithm": model_name,
                        "outer_fold": fold,
                        "threshold": fold_threshold,
                        **calculate_metrics(y_test, fold_probability, fold_threshold),
                    }
                )

            oof_registry[(set_name, model_name)] = oof_probability

    fold_results = pd.DataFrame(fold_rows)
    metric_columns = [
        "threshold",
        "ROC_AUC",
        "PR_AUC",
        "Accuracy",
        "Balanced_accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1_score",
        "Brier_score",
    ]
    summary = (
        fold_results.groupby(["feature_set", "algorithm"], sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(
        ["ROC_AUC_mean", "PR_AUC_mean"], ascending=False
    ).reset_index(drop=True)
    selected_key = (summary.loc[0, "feature_set"], summary.loc[0, "algorithm"])
    return fold_results, summary, oof_registry, selected_key


def strict_nested_feature_selection_sensitivity(development_data):
    """Repeat selection inside every inner-training and outer-training fold.

    This is a sensitivity analysis only. It does not overwrite the fixed feature
    sets, selected primary model, or primary locked threshold.
    """
    y = development_data[OUTCOME].astype(int).reset_index(drop=True)
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(development_data, y))
    models = candidate_algorithms()
    fold_rows, outer_selection_rows, inner_selection_rows = [], [], []
    selection_cache = {}

    for set_name in FEATURE_SETS:
        for model_name, estimator in models.items():
            for outer_fold, (fit_index, test_index) in enumerate(outer_splits, start=1):
                outer_train = development_data.iloc[fit_index].copy()
                outer_test = development_data.iloc[test_index].copy()
                y_fit, y_test = y.iloc[fit_index], y.iloc[test_index]
                inner_cv = StratifiedKFold(
                    n_splits=N_SPLITS,
                    shuffle=True,
                    random_state=RANDOM_STATE + 1000 + outer_fold,
                )
                inner_probability = np.full(len(y_fit), np.nan)

                for inner_fold, (inner_fit, inner_test) in enumerate(
                    inner_cv.split(outer_train, y_fit), start=1
                ):
                    inner_train = outer_train.iloc[inner_fit]
                    inner_valid = outer_train.iloc[inner_test]
                    inner_y = y_fit.iloc[inner_fit]
                    cache_key = (outer_fold, inner_fold)
                    if cache_key not in selection_cache:
                        selection_cache[cache_key] = select_core_features(
                            inner_train, inner_y,
                            RANDOM_STATE + 10000 * outer_fold + 100 * inner_fold,
                        )
                    core, _ = selection_cache[cache_key]
                    features = build_cost_tier_sets(core)[set_name]
                    for rank, feature in enumerate(core, start=1):
                        inner_selection_rows.append({
                            "feature_set": set_name, "algorithm": model_name,
                            "outer_fold": outer_fold, "inner_fold": inner_fold,
                            "rank": rank, "feature": feature,
                        })
                    fitted = make_pipeline(features, estimator).fit(
                        inner_train[features], inner_y
                    )
                    inner_probability[inner_test] = fitted.predict_proba(
                        inner_valid[features]
                    )[:, 1]

                fold_threshold = select_youden_threshold(y_fit, inner_probability)
                cache_key = (outer_fold, 0)
                if cache_key not in selection_cache:
                    selection_cache[cache_key] = select_core_features(
                        outer_train, y_fit, RANDOM_STATE + 100000 + outer_fold
                    )
                core, ranking = selection_cache[cache_key]
                features = build_cost_tier_sets(core)[set_name]
                for rank, feature in enumerate(core, start=1):
                    outer_selection_rows.append({
                        "feature_set": set_name, "algorithm": model_name,
                        "outer_fold": outer_fold, "rank": rank,
                        "feature": feature, "importance": float(ranking.loc[feature]),
                    })
                fitted = make_pipeline(features, estimator).fit(
                    outer_train[features], y_fit
                )
                probability = fitted.predict_proba(outer_test[features])[:, 1]
                fold_rows.append({
                    "feature_set": set_name, "algorithm": model_name,
                    "n_features": len(features), "outer_fold": outer_fold,
                    "threshold": fold_threshold,
                    **calculate_metrics(y_test, probability, fold_threshold),
                })

    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby(["feature_set", "algorithm"], as_index=False).agg(
        n_features=("n_features", "first"),
        ROC_AUC_mean=("ROC_AUC", "mean"), ROC_AUC_sd=("ROC_AUC", "std"),
        PR_AUC_mean=("PR_AUC", "mean"), PR_AUC_sd=("PR_AUC", "std"),
        Sensitivity_mean=("Sensitivity", "mean"),
        Specificity_mean=("Specificity", "mean"),
        Balanced_accuracy_mean=("Balanced_accuracy", "mean"),
    ).sort_values(["ROC_AUC_mean", "PR_AUC_mean"], ascending=False)
    return folds, summary, pd.DataFrame(outer_selection_rows), pd.DataFrame(inner_selection_rows)


def stratified_bootstrap_indices(y, iterations, seed):
    y = np.asarray(y, dtype=int)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        yield np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )


def compare_candidate_auc(y, oof_registry):
    """Paired resamples and one Holm family covering all 36 comparisons."""
    keys = list(oof_registry)
    points = {key: roc_auc_score(y, oof_registry[key]) for key in keys}
    distributions = {key: np.empty(N_BOOTSTRAP) for key in keys}
    for iteration, sampled in enumerate(
        stratified_bootstrap_indices(y, N_BOOTSTRAP, RANDOM_STATE + 2022)
    ):
        for key in keys:
            distributions[key][iteration] = roc_auc_score(
                np.asarray(y)[sampled], oof_registry[key][sampled]
            )
    auc_rows, comparison_rows = [], []
    for key in keys:
        lower, upper = np.quantile(distributions[key], [0.025, 0.975])
        auc_rows.append({
            "feature_set": key[0], "algorithm": key[1],
            "OOF_ROC_AUC": points[key],
            "CI_lower_95": lower, "CI_upper_95": upper,
        })
    for first, second in combinations(keys, 2):
        difference = distributions[first] - distributions[second]
        lower, upper = np.quantile(difference, [0.025, 0.975])
        left = (np.sum(difference <= 0) + 1) / (N_BOOTSTRAP + 1)
        right = (np.sum(difference >= 0) + 1) / (N_BOOTSTRAP + 1)
        comparison_rows.append({
            "model_A": " / ".join(first), "model_B": " / ".join(second),
            "ROC_AUC_A": points[first], "ROC_AUC_B": points[second],
            "difference_A_minus_B": points[first] - points[second],
            "difference_CI_lower_95": lower, "difference_CI_upper_95": upper,
            "P_raw": min(1.0, 2 * min(left, right)),
        })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["P_Holm"] = multipletests(comparisons["P_raw"], method="holm")[1]
    return pd.DataFrame(auc_rows), comparisons


def bootstrap_validation_metrics(y, probability, threshold):
    metric_names = [
        "ROC_AUC",
        "PR_AUC",
        "Accuracy",
        "Balanced_accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1_score",
        "Brier_score",
    ]
    point = calculate_metrics(y, probability, threshold)
    distributions = {name: np.empty(N_BOOTSTRAP) for name in metric_names}
    for iteration, sampled in enumerate(
        stratified_bootstrap_indices(y, N_BOOTSTRAP, RANDOM_STATE + 2024)
    ):
        sampled_metrics = calculate_metrics(
            np.asarray(y)[sampled], np.asarray(probability)[sampled], threshold
        )
        for name in metric_names:
            distributions[name][iteration] = sampled_metrics[name]

    rows = []
    for name in metric_names:
        values = distributions[name][np.isfinite(distributions[name])]
        lower, upper = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
        rows.append(
            {
                "Metric": name,
                "Estimate": point[name],
                "CI_lower_95": lower,
                "CI_upper_95": upper,
            }
        )
    return point, pd.DataFrame(rows)


def plot_candidate_models(summary, output_dir):
    plot_data = summary.sort_values("ROC_AUC_mean").copy()
    labels = plot_data["algorithm"].str.replace("LogisticRegression", "LR")
    labels = labels.str.replace("RandomForest", "RF")
    labels = labels.str.replace("GradientBoosting", "GB")
    labels = labels + " | " + plot_data["feature_set"].str.replace("_", " ")
    y_position = np.arange(len(plot_data))
    figure, axes = plt.subplots(1, 2, sharey=True)
    for axis, metric, title in [
        (axes[0], "ROC_AUC", "ROC-AUC"),
        (axes[1], "PR_AUC", "PR-AUC"),
    ]:
        for row, position in zip(plot_data.itertuples(), y_position):
            axis.errorbar(
                getattr(row, f"{metric}_mean"),
                position,
                xerr=getattr(row, f"{metric}_std"),
                fmt="o",
            )
        axis.set_xlabel(f"Mean {title}")
        axis.set_title(title)
    axes[0].set_yticks(y_position, labels)
    axes[0].set_ylabel("Candidate model")
    figure.suptitle("Nested Internal Validation in the 2022 Development Cohort")
    figure.tight_layout()
    figure.savefig(output_dir / "candidate_model_performance.png")
    plt.close(figure)


def plot_roc_pr(y, probability, threshold, validation_table, output_dir):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    point = calculate_metrics(y, probability, threshold)
    roc_row = validation_table.set_index("Metric").loc["ROC_AUC"]
    pr_row = validation_table.set_index("Metric").loc["PR_AUC"]
    figure, axes = plt.subplots(1, 2)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1])
    axes[0].scatter(1 - point["Specificity"], point["Sensitivity"])
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title(
        f"ROC curve\nAUC {roc_row.Estimate:.3f} "
        f"(95% CI {roc_row.CI_lower_95:.3f}-{roc_row.CI_upper_95:.3f})"
    )
    axes[1].plot(recall, precision)
    axes[1].axhline(np.mean(y))
    axes[1].scatter(point["Sensitivity"], point["PPV"])
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[1].set_title(
        f"Precision-recall curve\nAUC {pr_row.Estimate:.3f} "
        f"(95% CI {pr_row.CI_lower_95:.3f}-{pr_row.CI_upper_95:.3f})"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "ROC_PR_curves_2024.png")
    plt.close(figure)


def calibration_analysis(y, probability, output_dir):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    predicted_logit = np.log(clipped / (1 - clipped))
    fitted = sm.GLM(
        y, sm.add_constant(predicted_logit), family=sm.families.Binomial()
    ).fit()
    confidence_interval = np.asarray(fitted.conf_int())
    metrics = pd.DataFrame(
        [
            {
                "Metric": "Calibration_intercept",
                "Estimate": fitted.params[0],
                "CI_lower_95": confidence_interval[0, 0],
                "CI_upper_95": confidence_interval[0, 1],
            },
            {
                "Metric": "Calibration_slope",
                "Estimate": fitted.params[1],
                "CI_lower_95": confidence_interval[1, 0],
                "CI_upper_95": confidence_interval[1, 1],
            },
        ]
    )

    grouped = pd.DataFrame({"observed": y, "probability": probability})
    grouped["quintile"] = pd.qcut(grouped["probability"], 5, labels=False, duplicates="drop")
    grouped = grouped.groupby("quintile").agg(
        n=("observed", "size"),
        events=("observed", "sum"),
        predicted=("probability", "mean"),
        observed=("observed", "mean"),
    )
    intervals = [
        proportion_confint(row.events, row.n, method="wilson")
        for row in grouped.itertuples()
    ]
    grouped["lower"] = [interval[0] for interval in intervals]
    grouped["upper"] = [interval[1] for interval in intervals]

    figure, axis = plt.subplots()
    maximum = max(0.2, grouped["upper"].max()) * 1.05
    axis.plot([0, maximum], [0, maximum], label="Ideal")
    axis.errorbar(
        grouped["predicted"],
        grouped["observed"],
        yerr=np.maximum(0, [grouped["observed"] - grouped["lower"],
                            grouped["upper"] - grouped["observed"]]),
        marker="o",
        label="Observed (quintiles)",
    )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed proportion",
        xlim=(0, maximum),
        ylim=(0, maximum),
        title="Calibration in the 2024 Validation Cohort",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_plot_2024.png")
    plt.close(figure)
    return metrics, grouped.reset_index()
def decision_curve_analysis(validation_data, probability, age_probability, threshold, output_dir):
    y = validation_data[OUTCOME].to_numpy(dtype=int)
    thresholds = np.arange(0.01, 0.201, 0.001)

    def model_net_benefit(model_probability, probability_threshold):
        positive = model_probability >= probability_threshold
        true_positive = np.sum(positive & (y == 1))
        false_positive = np.sum(positive & (y == 0))
        odds = probability_threshold / (1 - probability_threshold)
        return true_positive / len(y) - false_positive / len(y) * odds

    prevalence = y.mean()
    results = pd.DataFrame({"threshold_probability": thresholds})
    results["LR_Model_2"] = [model_net_benefit(probability, value) for value in thresholds]
    results["Age_only"] = [model_net_benefit(age_probability, value) for value in thresholds]
    results["Refer_all"] = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    results["Refer_none"] = 0.0

    figure, axis = plt.subplots()
    for column in ["LR_Model_2", "Age_only", "Refer_all", "Refer_none"]:
        axis.plot(thresholds, results[column], label=column.replace("_", " "))
    axis.axvline(threshold, label=f"Locked threshold {threshold:.3f}")
    axis.set(
        xlabel="Threshold probability for referral",
        ylabel="Net benefit",
        title="Decision-Curve Analysis: 2024 Temporal Validation",
        xlim=(0.01, 0.20),
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "decision_curve_2024.png")
    plt.close(figure)

    locked_values = {
        "threshold_probability": threshold,
        "LR_Model_2": model_net_benefit(probability, threshold),
        "Age_only": model_net_benefit(age_probability, threshold),
        "Refer_all": prevalence - (1 - prevalence) * threshold / (1 - threshold),
        "Refer_none": 0.0,
    }
    return results, pd.DataFrame([locked_values])


def shap_importance(fitted_pipeline, development_data, validation_data, features, output_dir):
    """Explain the same fitted LR using one 2022 background, on the logit scale."""
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    transformed_train = pd.DataFrame(
        preprocessor.transform(development_data[features]), columns=names
    )
    background = transformed_train.sample(
        min(100, len(transformed_train)), random_state=RANDOM_STATE
    )
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(background))
    rows, direction_rows = [], []
    for year, raw in [(2022, development_data), (2024, validation_data)]:
        raw = raw.reset_index(drop=True)
        transformed = pd.DataFrame(preprocessor.transform(raw[features]), columns=names)
        explanation = explainer(transformed)
        components = pd.DataFrame(explanation.values, columns=names)
        grouped = pd.DataFrame({
            feature: components[[
                name for name in names if name == feature or name.startswith(feature + "_")
            ]].sum(axis=1) for feature in features
        })
        if not np.allclose(
            grouped.sum(axis=1) + explanation.base_values,
            model.decision_function(transformed.to_numpy()), atol=1e-7,
        ):
            raise ValueError("SHAP contributions do not reconstruct the fitted LR logit.")
        groups = {"Overall": np.ones(len(raw), dtype=bool)}
        if year == 2024:
            groups.update({"Male": raw["sex"].eq(1), "Female": raw["sex"].eq(2)})
        for group, mask in groups.items():
            importance = grouped.loc[mask].abs().mean().sort_values(ascending=False)
            for rank, (feature, value) in enumerate(importance.items(), start=1):
                rows.append({
                    "Year": year, "Group": group, "Rank": rank, "Feature": feature,
                    "Mean_absolute_SHAP": value,
                    "Mean_signed_SHAP": grouped.loc[mask, feature].mean(),
                })
        for feature in features:
            if feature in CONTINUOUS_FEATURES:
                direction_rows.append({
                    "Year": year, "Feature": feature, "Level": "continuous",
                    "Summary": "Spearman raw value versus signed SHAP",
                    "Estimate": spearmanr(raw[feature], grouped[feature]).statistic,
                })
            else:
                for level, index in raw.groupby(feature).groups.items():
                    direction_rows.append({
                        "Year": year, "Feature": feature, "Level": level,
                        "Summary": "Mean signed SHAP within category",
                        "Estimate": grouped.loc[index, feature].mean(),
                    })
    table = pd.DataFrame(rows)
    pd.DataFrame(direction_rows).to_csv(output_dir / "SHAP_directions.csv", index=False)
    ranks = table.query("Group == 'Overall'").pivot(
        index="Feature", columns="Year", values="Rank"
    )
    pd.DataFrame([{
        "Spearman_rank_correlation": spearmanr(ranks[2022], ranks[2024]).statistic,
        "n_predictors": len(features),
    }]).to_csv(output_dir / "SHAP_year_rank_comparison.csv", index=False)
    overall = table.query("Year == 2024 and Group == 'Overall'").sort_values(
        "Mean_absolute_SHAP"
    )
    figure, axis = plt.subplots()
    axis.barh(overall["Feature"], overall["Mean_absolute_SHAP"])
    axis.set(xlabel="Mean absolute SHAP value (logit)", title="Overall SHAP Importance")
    figure.tight_layout()
    figure.savefig(output_dir / "SHAP_importance_overall.png")
    plt.close(figure)
    sex_table = table.query("Year == 2024 and Group != 'Overall'").pivot(
        index="Feature", columns="Group", values="Mean_absolute_SHAP"
    ).loc[overall["Feature"]]
    sex_table.plot.barh()
    plt.xlabel("Mean absolute SHAP value (logit)")
    plt.tight_layout()
    plt.savefig(output_dir / "SHAP_importance_by_sex.png")
    plt.close()
    return table


def subgroup_performance(validation_data, probability, threshold):
    """Evaluate the locked model by sex, residence, and age group."""
    frame = validation_data.reset_index(drop=True).copy()
    frame["age_group"] = np.where(frame["age"] >= 75, ">=75", "65-74")
    rows = []
    for column in ["sex", "town_t", "age_group"]:
        for level, index in frame.groupby(column).groups.items():
            y = frame.loc[index, OUTCOME].to_numpy(dtype=int)
            if np.unique(y).size < 2:
                continue
            rows.append({
                "subgroup": column, "level": level, "n": len(y),
                "cases": int(y.sum()),
                **calculate_metrics(y, np.asarray(probability)[index], threshold),
            })
    return pd.DataFrame(rows)


def run_analysis(data_path, output_dir):
    data_path = Path(data_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analytic_data = pd.read_csv(data_path, dtype={"ID": str})
    development_data = analytic_data.loc[analytic_data["year"].eq(2022)].reset_index(drop=True)
    validation_data = analytic_data.loc[analytic_data["year"].eq(2024)].reset_index(drop=True)
    y_development = development_data[OUTCOME].to_numpy(dtype=int)
    y_validation = validation_data[OUTCOME].to_numpy(dtype=int)

    descriptive_outputs(analytic_data, output_dir)
    ranking = rank_initial_candidates(development_data)
    ranking.to_csv(
        output_dir / "candidate_variable_RF_ranking_and_fixed_pool_2022.csv",
        index=False,
    )

    fold_results, summary, oof_registry, selected_key = nested_internal_validation(development_data)
    if selected_key != ("Model_2", "LogisticRegression"):
        raise RuntimeError(
            f"The selected model was {selected_key}, not the manuscript model "
            "('Model_2', 'LogisticRegression'). Check the data and software versions."
        )
    fold_results.to_csv(output_dir / "candidate_model_fold_results_2022.csv", index=False)
    summary.to_csv(output_dir / "Table_S3_candidate_models.csv", index=False)
    plot_candidate_models(summary, output_dir)

    strict_folds, strict_summary, strict_outer, strict_inner = (
        strict_nested_feature_selection_sensitivity(development_data)
    )
    strict_folds.to_csv(
        output_dir / "strict_nested_feature_selection_fold_results_2022.csv",
        index=False,
    )
    strict_summary.to_csv(
        output_dir / "strict_nested_feature_selection_summary_2022.csv",
        index=False,
    )
    strict_outer.to_csv(
        output_dir / "strict_nested_outer_fold_selections_2022.csv", index=False
    )
    strict_inner.to_csv(
        output_dir / "strict_nested_inner_fold_selections_2022.csv", index=False
    )

    selections = strict_outer.loc[
        strict_outer["feature_set"].eq("Model_1")
        & strict_outer["algorithm"].eq("LogisticRegression")
    ]
    frequencies = selections.groupby("feature").size().reindex(LOW_COST_CANDIDATES, fill_value=0)
    frequencies.rename("outer_folds_selected").to_csv(output_dir / "selection_frequency.csv")
    overlap = selections.groupby("outer_fold")["feature"].apply(
        lambda values: len(set(values) & set(QUESTIONNAIRE_FEATURES)) / 7
    )
    overlap.rename("fixed_core_overlap").to_csv(output_dir / "selection_overlap.csv")

    auc_table, comparison_table = compare_candidate_auc(
        y_development, oof_registry
    )
    auc_table.to_csv(output_dir / "candidate_model_OOF_AUC_CI.csv", index=False)
    comparison_table.to_csv(output_dir / "paired_AUC_comparisons.csv", index=False)
    locked_features = FEATURE_SETS[selected_key[0]]
    locked_threshold = select_youden_threshold(
        y_development, oof_registry[selected_key]
    )
    locked_pipeline = make_pipeline(
        locked_features, candidate_algorithms()[selected_key[1]]
    ).fit(development_data[locked_features], y_development)

    pd.DataFrame({
        "ID": development_data["ID"], "observed": y_development,
        "oof_probability": oof_registry[selected_key],
        "locked_threshold": locked_threshold,
    }).to_csv(output_dir / "selected_model_OOF_predictions_2022.csv", index=False)
    age_model, age_threshold = fit_development_benchmark(development_data, ["age"])
    reduced_features = FEATURE_SETS["Model_1"]
    reduced_threshold = select_youden_threshold(
        y_development, oof_registry[("Model_1", "LogisticRegression")]
    )
    reduced_model = make_pipeline(
        reduced_features, candidate_algorithms()["LogisticRegression"]
    ).fit(development_data[reduced_features], y_development)

    validation_probability = locked_pipeline.predict_proba(
        validation_data[locked_features]
    )[:, 1]
    validation_point, validation_table = bootstrap_validation_metrics(
        y_validation, validation_probability, locked_threshold
    )
    validation_table.to_csv(output_dir / "Table_S4_temporal_validation.csv", index=False)
    age_probability = age_model.predict_proba(validation_data[["age"]])[:, 1]
    reduced_probability = reduced_model.predict_proba(validation_data[reduced_features])[:, 1]
    benchmark_tables, burden_rows = [], []
    for label, probability, threshold in [
        ("LR_Model_2", validation_probability, locked_threshold),
        ("Age_only", age_probability, age_threshold),
        ("LR_Model_1_no_anthropometry", reduced_probability, reduced_threshold),
    ]:
        if label == "LR_Model_2":
            table = validation_table.copy()
        else:
            _, table = bootstrap_validation_metrics(y_validation, probability, threshold)
        benchmark_tables.append(table.assign(Model=label, Threshold=threshold))
        burden_rows.append(screening_burden(y_validation, probability, threshold, label))
    pd.concat(benchmark_tables).to_csv(output_dir / "benchmark_performance.csv", index=False)
    pd.DataFrame(burden_rows).to_csv(output_dir / "screening_referral_burden.csv", index=False)
    paired_validation_difference(y_validation, validation_probability, reduced_probability).to_csv(
        output_dir / "anthropometry_removal_paired_comparison.csv", index=False
    )
    plot_roc_pr(
        y_validation,
        validation_probability,
        locked_threshold,
        validation_table,
        output_dir,
    )

    calibration_metrics, calibration_groups = calibration_analysis(
        y_validation, validation_probability, output_dir
    )
    calibration_metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    calibration_groups.to_csv(output_dir / "calibration_groups.csv", index=False)

    dca_results, dca_at_threshold = decision_curve_analysis(
        validation_data,
        validation_probability,
        age_probability,
        locked_threshold,
        output_dir,
    )
    dca_results.to_csv(output_dir / "decision_curve_data.csv", index=False)
    dca_at_threshold.to_csv(output_dir / "net_benefit_at_locked_threshold.csv", index=False)
    shap_table = shap_importance(
        locked_pipeline,
        development_data,
        validation_data,
        locked_features,
        output_dir,
    )
    shap_table.to_csv(output_dir / "SHAP_importance_overall_and_by_sex.csv", index=False)

    subgroup_performance(
        validation_data, validation_probability, locked_threshold
    ).to_csv(output_dir / "subgroup_performance_best_model.csv", index=False)
    prediction_output = validation_data[["ID", "year", OUTCOME]].copy()
    prediction_output["predicted_risk"] = validation_probability
    prediction_output["threshold"] = locked_threshold
    prediction_output["predicted_class"] = (
        validation_probability >= locked_threshold
    ).astype(int)
    prediction_output.to_csv(output_dir / "final_predictions_2024.csv", index=False)

    print("Selected model:", selected_key)
    print("Locked threshold:", f"{locked_threshold:.6f}")
    print("2022 sample:", len(y_development), "events:", int(y_development.sum()))
    print("2024 sample:", len(y_validation), "events:", int(y_validation.sum()))
    print("2024 ROC-AUC:", f"{validation_point['ROC_AUC']:.6f}")
    print("2024 PR-AUC:", f"{validation_point['PR_AUC']:.6f}")
    return analytic_data, locked_threshold, shap_table


def sensitivity_nested_validation(development_data):
    """Repeat nested five-fold validation for the prespecified LR Model 2 only."""
    features = FEATURE_SETS["Model_2"]
    X = development_data[features].reset_index(drop=True)
    y = development_data[OUTCOME].astype(int).reset_index(drop=True)
    estimator = candidate_algorithms()["LogisticRegression"]
    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(X, y))
    oof_probability = np.full(len(y), np.nan)
    fold_rows = []

    for fold, (fit_index, test_index) in enumerate(outer_splits, start=1):
        X_fit, X_test = X.iloc[fit_index], X.iloc[test_index]
        y_fit, y_test = y.iloc[fit_index], y.iloc[test_index]
        inner_cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + 1000 + fold,
        )
        inner_probability = cross_val_predict(
            make_pipeline(features, estimator),
            X_fit,
            y_fit,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        fold_threshold = select_youden_threshold(y_fit, inner_probability)
        fitted = make_pipeline(features, estimator).fit(X_fit, y_fit)
        fold_probability = fitted.predict_proba(X_test)[:, 1]
        oof_probability[test_index] = fold_probability
        fold_rows.append(
            {
                "outer_fold": fold,
                "threshold": fold_threshold,
                **calculate_metrics(y_test, fold_probability, fold_threshold),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("Sensitivity-analysis OOF probabilities are incomplete.")
    locked_threshold = select_youden_threshold(y, oof_probability)
    fitted_pipeline = make_pipeline(features, estimator).fit(X, y)
    return pd.DataFrame(fold_rows), oof_probability, locked_threshold, fitted_pipeline


def run_sensitivity_analysis(data_path, output_dir, primary_data, primary_threshold, primary_shap):
    """Develop LR Model 2 in expanded 2022 data and validate it in expanded 2024 data."""
    output_dir = Path(output_dir).expanduser().resolve() / "sensitivity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(Path(data_path).expanduser().resolve(), dtype={"ID": str})
    compare_sample_membership(primary_data, data, output_dir)
    development_data = data.loc[data["year"].eq(2022)].reset_index(drop=True)
    validation_data = data.loc[data["year"].eq(2024)].reset_index(drop=True)
    y_development = development_data[OUTCOME].to_numpy(dtype=int)
    y_validation = validation_data[OUTCOME].to_numpy(dtype=int)

    fold_results, oof_probability, locked_threshold, fitted_pipeline = (
        sensitivity_nested_validation(development_data)
    )
    fold_results.to_csv(
        output_dir / "sensitivity_nested_fold_results_2022.csv", index=False
    )
    pd.DataFrame(
        {
            "observed": y_development,
            "oof_probability": oof_probability,
            "locked_threshold": locked_threshold,
        }
    ).to_csv(output_dir / "sensitivity_OOF_predictions_2022.csv", index=False)

    age_model, _ = fit_development_benchmark(development_data, ["age"])
    features = FEATURE_SETS["Model_2"]
    validation_probability = fitted_pipeline.predict_proba(
        validation_data[features]
    )[:, 1]
    validation_point, validation_table = bootstrap_validation_metrics(
        y_validation, validation_probability, locked_threshold
    )
    validation_table.to_csv(
        output_dir / "Table_S5_sensitivity_temporal_validation.csv", index=False
    )

    # Supplemental threshold analysis: keep the expanded-sample model fixed and
    # evaluate it at the original primary-analysis threshold without refitting.
    original_point, original_table = bootstrap_validation_metrics(
        y_validation, validation_probability, primary_threshold
    )
    original_table.to_csv(
        output_dir / "sensitivity_model_at_original_threshold_with_95CI.csv",
        index=False,
    )
    pd.DataFrame([{
        "Threshold": primary_threshold,
        "N": len(y_validation), "Events": int(y_validation.sum()),
        "TP": original_point["TP"], "FP": original_point["FP"],
        "TN": original_point["TN"], "FN": original_point["FN"],
    }]).to_csv(
        output_dir / "sensitivity_model_at_original_threshold_confusion.csv",
        index=False,
    )
    plot_roc_pr(
        y_validation,
        validation_probability,
        locked_threshold,
        validation_table,
        output_dir,
    )

    calibration_metrics, calibration_groups = calibration_analysis(
        y_validation, validation_probability, output_dir
    )
    calibration_metrics.to_csv(
        output_dir / "sensitivity_calibration_metrics.csv", index=False
    )
    calibration_groups.to_csv(
        output_dir / "sensitivity_calibration_groups.csv", index=False
    )
    age_probability = age_model.predict_proba(validation_data[["age"]])[:, 1]
    shap_table = shap_importance(
        fitted_pipeline, development_data, validation_data, features, output_dir
    )
    shap_table.to_csv(output_dir / "SHAP_importance_expanded.csv", index=False)
    pd.concat([
        primary_shap.query("Year == 2024 and Group == 'Overall'").assign(Sample="Primary"),
        shap_table.query("Year == 2024 and Group == 'Overall'").assign(Sample="Expanded"),
    ]).to_csv(output_dir / "SHAP_primary_vs_expanded.csv", index=False)
    pd.DataFrame([
        screening_burden(y_validation, validation_probability, threshold, label)
        for label, threshold in [("Expanded_relocked", locked_threshold),
                                 ("Expanded_primary_threshold", primary_threshold)]
    ]).to_csv(output_dir / "screening_referral_burden.csv", index=False)
    predictions = validation_data[["ID", "year", OUTCOME]].copy()
    predictions["predicted_risk"] = validation_probability
    predictions["threshold"] = locked_threshold
    predictions["predicted_class"] = (validation_probability >= locked_threshold).astype(int)
    predictions.to_csv(output_dir / "final_predictions_2024.csv", index=False)
    dca_results, dca_at_threshold = decision_curve_analysis(
        validation_data,
        validation_probability,
        age_probability,
        locked_threshold,
        output_dir,
    )
    dca_results.to_csv(
        output_dir / "sensitivity_decision_curve_data.csv", index=False
    )
    dca_at_threshold.to_csv(
        output_dir / "sensitivity_net_benefit_at_locked_threshold.csv", index=False
    )
    print("Sensitivity analysis: LR Model 2")
    print("Locked threshold:", f"{locked_threshold:.6f}")
    print("Expanded 2022 sample:", len(y_development), "events:", int(y_development.sum()))
    print("Expanded 2024 sample:", len(y_validation), "events:", int(y_validation.sum()))
    print("2024 ROC-AUC:", f"{validation_point['ROC_AUC']:.6f}")
    print("2024 PR-AUC:", f"{validation_point['PR_AUC']:.6f}")


def main():
    args = parse_arguments()
    primary, threshold, primary_shap = run_analysis(args.data, args.output_dir)
    run_sensitivity_analysis(
        args.sensitivity_data, args.output_dir, primary, threshold, primary_shap
    )


if __name__ == "__main__":
    main()
