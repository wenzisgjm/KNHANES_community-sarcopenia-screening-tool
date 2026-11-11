"""Approved manuscript figures drawn from existing folds and locked predictions.

These helpers never fit a model, choose a threshold, or modify input frames.
The plotting layout and age bootstrap order reproduce the final analysis notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def plot_candidate_fold_discrimination(
    fold_results, output_dir, *, y_development, oof_registry, random_state=42
):
    """Plot all 45 actual fold results, after reconciling their OOF predictions.

    The public analysis uses ``algorithm``/``ROC_AUC`` column names; the final
    notebook used ``model``/``AUC``. Only these names are adapted here. Shared
    fold indices are reconstructed from the original split rule, without fitting.
    """
    y_train = np.asarray(y_development)
    if y_train.ndim != 1 or not np.isin(y_train, [0, 1]).all():
        raise ValueError("Development outcomes must be a one-dimensional binary array.")
    candidate_fold_results_2022 = fold_results.rename(
        columns={"algorithm": "model", "ROC_AUC": "AUC"}
    ).copy(deep=True)
    candidate_oof_registry = {
        key: {"oof_probability": value} for key, value in oof_registry.items()
    }
    outer_splits_2022 = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=random_state
    ).split(np.zeros(len(y_train)), y_train))
    # Preserve the scientific ordering, rather than sorting by performance.
    _fold_feature_sets = ["Model_1", "Model_2", "Model_3"]
    _fold_algorithms = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    _fold_algorithm_labels = {
        "LogisticRegression": "LR", "RandomForest": "RF", "GradientBoosting": "GB"
    }
    _fold_keys = [(fs, alg) for alg in _fold_algorithms for fs in _fold_feature_sets]
    _fold_order = {key: number for number, key in enumerate(_fold_keys)}
    fold_auc_plot_data_2022 = candidate_fold_results_2022.copy(deep=True)
    _fold_required_columns = {"feature_set", "model", "outer_fold", "AUC", "PR_AUC"}
    if not _fold_required_columns.issubset(fold_auc_plot_data_2022.columns):
        raise ValueError(f"Required columns: {sorted(_fold_required_columns)}")

    _fold_observed_keys = set(zip(
        fold_auc_plot_data_2022["feature_set"], fold_auc_plot_data_2022["model"]
    ))
    if _fold_observed_keys != set(_fold_keys) or len(fold_auc_plot_data_2022) != 45:
        raise ValueError("Expected exactly 9 candidate models, with 5 rows per model.")
    if fold_auc_plot_data_2022.duplicated(["feature_set", "model", "outer_fold"]).any():
        raise ValueError("Duplicate model/fold rows were found.")
    for _fold_metric in ("AUC", "PR_AUC"):
        if not np.isfinite(fold_auc_plot_data_2022[_fold_metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Every fold must have a finite {_fold_metric}.")
        if not fold_auc_plot_data_2022[_fold_metric].between(0, 1).all():
            raise ValueError(f"{_fold_metric} values must lie between 0 and 1.")
    if len(outer_splits_2022) != 5:
        raise ValueError("Exactly five shared outer splits are required.")

    _fold_y = np.asarray(y_train, dtype=int)
    _fold_validation_indices = [np.asarray(pair[1], dtype=int) for pair in outer_splits_2022]
    if not np.array_equal(
        np.sort(np.concatenate(_fold_validation_indices)), np.arange(len(_fold_y))
    ):
        raise ValueError("Outer validation folds must partition the 2022 sample exactly once.")

    # Reconcile each displayed value with its original OOF predictions on the
    # same shared outer validation indices. This does not rerun cross-validation.
    _fold_counts = {}
    for _fold_number, _fold_indices in enumerate(_fold_validation_indices, start=1):
        _fold_counts[_fold_number] = (len(_fold_indices), int(_fold_y[_fold_indices].sum()))
    for _fold_key in _fold_keys:
        _fold_group = fold_auc_plot_data_2022.loc[
            fold_auc_plot_data_2022["feature_set"].eq(_fold_key[0])
            & fold_auc_plot_data_2022["model"].eq(_fold_key[1])
        ].sort_values("outer_fold")
        if _fold_group["outer_fold"].tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"Missing or invalid outer-fold labels for {_fold_key}.")
        _fold_probabilities = np.asarray(
            candidate_oof_registry[_fold_key]["oof_probability"], dtype=float
        )
        if _fold_probabilities.shape != _fold_y.shape or not np.isfinite(_fold_probabilities).all():
            raise ValueError(f"Invalid OOF probability array for {_fold_key}.")
        for _fold_row in _fold_group.itertuples(index=False):
            _fold_indices = _fold_validation_indices[int(_fold_row.outer_fold) - 1]
            for _fold_metric, _fold_scorer in (
                ("AUC", roc_auc_score), ("PR_AUC", average_precision_score)
            ):
                _fold_recalculated = _fold_scorer(
                    _fold_y[_fold_indices], _fold_probabilities[_fold_indices]
                )
                if not np.isclose(
                    getattr(_fold_row, _fold_metric), _fold_recalculated, rtol=0, atol=1e-12
                ):
                    raise ValueError(
                        f"OOF/fold {_fold_metric} mismatch: {_fold_key}, fold {_fold_row.outer_fold}."
                    )

    fold_auc_plot_data_2022["n_validation"] = fold_auc_plot_data_2022["outer_fold"].map(
        lambda fold: _fold_counts[int(fold)][0]
    )
    fold_auc_plot_data_2022["n_events"] = fold_auc_plot_data_2022["outer_fold"].map(
        lambda fold: _fold_counts[int(fold)][1]
    )
    fold_auc_plot_data_2022["model_order"] = [
        _fold_order[(fs, alg)] for fs, alg in zip(
            fold_auc_plot_data_2022["feature_set"], fold_auc_plot_data_2022["model"]
        )
    ]
    fold_auc_plot_data_2022 = fold_auc_plot_data_2022.sort_values(
        ["model_order", "outer_fold"]
    ).reset_index(drop=True)

    fold_auc_summary_2022 = (
        fold_auc_plot_data_2022.groupby(["feature_set", "model"], sort=False)
        .agg(
            n_folds=("AUC", "count"),
            AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
            AUC_minimum=("AUC", "min"), AUC_maximum=("AUC", "max"),
            PR_AUC_mean=("PR_AUC", "mean"), PR_AUC_sd=("PR_AUC", "std"),
            PR_AUC_minimum=("PR_AUC", "min"), PR_AUC_maximum=("PR_AUC", "max"),
        )
        .reset_index()
    )

    # Horizontal minimum–maximum ranges with five individual fold points.
    # The central marker is the arithmetic mean, matching the manuscript.
    # Vertical offsets distinguish overlapping points; x values are unchanged.
    _fold_labels = [
        f"{_fold_algorithm_labels[alg]} | {fs.replace('_', ' ')}" for fs, alg in _fold_keys
    ]
    _fold_positions = np.arange(1, 10)
    _fold_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    _fold_markers = ["o", "s", "^", "D", "v"]
    _fold_offsets = np.linspace(-0.13, 0.13, 5)
    _fold_range_color = "#A4ADB7"
    _fold_mean_color = "#263746"

    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fold_auc_figure_2022, _fold_axes = plt.subplots(
            nrows=1, ncols=2, figsize=(14.2, 7.6), sharey=True
        )
        for _fold_ax, _fold_metric, _fold_panel_title in zip(
            _fold_axes,
            ("AUC", "PR_AUC"),
            ("A. ROC-AUC", "B. PR-AUC (average precision)"),
        ):
            _fold_means = fold_auc_summary_2022[f"{_fold_metric}_mean"].to_numpy()
            _fold_minima = fold_auc_summary_2022[f"{_fold_metric}_minimum"].to_numpy()
            _fold_maxima = fold_auc_summary_2022[f"{_fold_metric}_maximum"].to_numpy()
            _fold_ax.hlines(
                _fold_positions, _fold_minima, _fold_maxima,
                color=_fold_range_color, linewidth=2.0, zorder=1,
            )
            # Explicit end caps are placed at the observed extrema.
            _fold_ax.vlines(
                _fold_minima, _fold_positions - 0.09, _fold_positions + 0.09,
                color=_fold_range_color, linewidth=1.4, zorder=1,
            )
            _fold_ax.vlines(
                _fold_maxima, _fold_positions - 0.09, _fold_positions + 0.09,
                color=_fold_range_color, linewidth=1.4, zorder=1,
            )
            _fold_ax.vlines(
                _fold_means, _fold_positions - 0.22, _fold_positions + 0.22,
                color=_fold_mean_color, linewidth=2.5, zorder=2,
            )
            for _fold_i, (_fold_color, _fold_marker, _fold_offset) in enumerate(
                zip(_fold_colors, _fold_markers, _fold_offsets), start=1
            ):
                _fold_points = fold_auc_plot_data_2022.loc[
                    fold_auc_plot_data_2022["outer_fold"].eq(_fold_i)
                ].sort_values("model_order")
                _fold_ax.scatter(
                    _fold_points[_fold_metric], _fold_positions + _fold_offset,
                    s=45, marker=_fold_marker, color=_fold_color,
                    edgecolor="white", linewidth=0.7, zorder=4,
                )
            for _fold_separator in (3.5, 6.5):
                _fold_ax.axhline(_fold_separator, color="#D8DDE3", linewidth=0.8)

            _fold_ax.set_title(_fold_panel_title, fontsize=12.5, fontweight="bold", pad=12)
            _fold_ax.set_xlabel(
                "Outer validation-fold ROC-AUC" if _fold_metric == "AUC"
                else "Outer validation-fold PR-AUC (average precision)",
                fontsize=10.5, labelpad=10,
            )
            # The two metrics have separate, explicitly labeled numerical scales.
            _fold_upper = min(1.0, np.ceil((_fold_maxima.max() + 0.03) * 10) / 10)
            _fold_lower = (
                max(0.0, min(0.5, np.floor((_fold_minima.min() - 0.03) * 10) / 10))
                if _fold_metric == "AUC" else 0.0
            )
            _fold_ax.set_xlim(_fold_lower, _fold_upper)
            _fold_ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D8DDE3", alpha=0.7)
            _fold_ax.set_axisbelow(True)
            _fold_ax.tick_params(axis="y", length=0)
            _fold_ax.tick_params(axis="x", labelsize=10)
            for _fold_side in ("top", "right"):
                _fold_ax.spines[_fold_side].set_visible(False)
            for _fold_side in ("left", "bottom"):
                _fold_ax.spines[_fold_side].set_color("#888888")

        _fold_axes[0].set_yticks(_fold_positions)
        _fold_axes[0].set_yticklabels(_fold_labels, fontsize=10.5)
        _fold_axes[0].set_ylim(9.6, 0.4)
        _fold_axes[0].set_ylabel("Candidate model", fontsize=11, labelpad=9)
        fold_auc_figure_2022.suptitle(
            "Fold-level Discrimination of Nine Candidate Models",
            fontsize=16, fontweight="bold", y=0.97,
        )
        fold_auc_figure_2022.text(
            0.5, 0.922,
            "2022 development set | Five shared outer folds | Each point is one held-out fold",
            ha="center", va="center", fontsize=10, color="#555555",
        )
        _fold_legend_handles = [
            Line2D(
                [0], [0], marker=marker, linestyle="none", color=color,
                markersize=6.5, markeredgecolor="white", markeredgewidth=0.5,
                label=f"Fold {number}",
            )
            for number, (color, marker) in enumerate(zip(_fold_colors, _fold_markers), start=1)
        ]
        _fold_legend_handles.extend([
            Line2D([0], [0], color=_fold_range_color, linewidth=2, label="Min–max"),
            Line2D(
                [0], [0], color=_fold_mean_color, linestyle="none",
                marker="|", markersize=13, markeredgewidth=2.5, label="Mean",
            ),
        ])
        fold_auc_figure_2022.legend(
            handles=_fold_legend_handles, loc="center", bbox_to_anchor=(0.5, 0.118),
            ncol=7, frameon=False, fontsize=9.5, handletextpad=0.45, columnspacing=1.7,
        )
        fold_auc_figure_2022.text(
            0.5, 0.034,
            "Ranges show observed minimum–maximum values; dark vertical marks show means. PR-AUC is average precision.\n"
            "Fold summaries are descriptive, not confidence intervals; training samples overlap across folds.",
            ha="center", va="bottom", fontsize=9, color="#555555", linespacing=1.4,
        )
        fold_auc_figure_2022.subplots_adjust(
            left=0.13, right=0.98, top=0.835, bottom=0.245, wspace=0.16
        )

        _fold_output_directory = Path(output_dir) / "reviewer_figures"
        _fold_output_directory.mkdir(parents=True, exist_ok=True)
        for _fold_extension in ("png", "pdf"):
            fold_auc_figure_2022.savefig(
                _fold_output_directory / f"candidate_model_fold_AUC_AP_distribution_2022.{_fold_extension}",
                dpi=600, bbox_inches="tight", facecolor="white",
            )
        # Preserve the previous script filename as an alias of the revised plot.
        fold_auc_figure_2022.savefig(
            Path(output_dir) / "candidate_model_performance.png",
            dpi=600, bbox_inches="tight", facecolor="white",
        )
        plt.close(fold_auc_figure_2022)

    fold_auc_plot_data_2022.drop(columns="model_order").to_csv(
        _fold_output_directory / "candidate_model_fold_AUC_AP_plot_data_2022.csv",
        index=False, encoding="utf-8-sig",
    )
    fold_auc_summary_2022.to_csv(
        _fold_output_directory / "candidate_model_fold_AUC_AP_summary_2022.csv",
        index=False, encoding="utf-8-sig",
    )
    return fold_auc_plot_data_2022.drop(columns="model_order"), fold_auc_summary_2022


def plot_age_stratified_roc(
    validation_data, probability, threshold, output_dir, *, fitted_pipeline,
    features, selected_key, outcome="Sarcopenia", random_state=42
):
    """Plot 2024 age groups using the same locked LR Model 2 and threshold.

    A single RNG, seeded at 42 in the manuscript analysis, processes ages 65–74
    first and ages >=75 second. Each group uses 5,000 outcome-stratified resamples
    (cases then non-cases), without model fitting or threshold reselection.
    ``predict_proba`` below only checks the saved probabilities' row alignment.
    """
    valid_df = validation_data
    OUTCOME = outcome
    y_valid = valid_df[OUTCOME]
    valid_probability_2024 = probability
    locked_threshold = threshold
    best_pipe = fitted_pipeline
    best_features = features
    LOCKED_FEATURE_SET, LOCKED_MODEL = selected_key
    if LOCKED_FEATURE_SET != "Model_2" or LOCKED_MODEL != "LogisticRegression":
        raise ValueError("This figure is defined for the locked LR Model 2.")
    if not valid_df.index.is_unique:
        raise ValueError("valid_df must have a unique index for participant alignment.")
    if not isinstance(y_valid, pd.Series) or not y_valid.index.equals(valid_df.index):
        raise ValueError("y_valid must retain the same participant index/order as valid_df.")
    if not {"age", "year", OUTCOME}.issubset(valid_df.columns):
        raise ValueError("valid_df must contain age, year and the outcome column.")
    if not valid_df["year"].eq(2024).all():
        raise ValueError("Only the 2024 temporal-validation participants may enter this figure.")

    _age_roc_ages = pd.to_numeric(valid_df["age"], errors="coerce").to_numpy(dtype=float)
    _age_roc_y_raw = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float)
    _age_roc_prob = np.asarray(valid_probability_2024, dtype=float)
    _age_roc_threshold = float(locked_threshold)
    if _age_roc_prob.ndim != 1 or len(_age_roc_prob) != len(valid_df):
        raise ValueError("The 2024 predicted probabilities must align one-to-one with valid_df.")
    if not np.isfinite(_age_roc_ages).all() or np.any(_age_roc_ages < 65):
        raise ValueError("Age must be nonmissing and at least 65 years for every participant.")
    if not np.isfinite(_age_roc_y_raw).all() or not np.isin(_age_roc_y_raw, [0, 1]).all():
        raise ValueError("The observed outcome must contain only nonmissing 0/1 values.")
    _age_roc_y = _age_roc_y_raw.astype(int)
    if not np.array_equal(_age_roc_y, valid_df[OUTCOME].to_numpy()):
        raise ValueError("y_valid does not match the outcomes in valid_df.")
    if not np.isfinite(_age_roc_prob).all() or np.any((_age_roc_prob < 0) | (_age_roc_prob > 1)):
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    if not 0 < _age_roc_threshold < 1:
        raise ValueError("The 2022 locked threshold must be between 0 and 1.")

    # Verify that the saved probability vector still follows the current row order.
    # predict_proba is used only as an alignment check; the model is never refitted.
    _age_roc_positive = np.flatnonzero(np.asarray(best_pipe.classes_) == 1)
    if len(_age_roc_positive) != 1:
        raise ValueError("Cannot uniquely identify the positive class in the locked model.")
    _age_roc_check = best_pipe.predict_proba(valid_df[best_features])[:, _age_roc_positive[0]]
    if not np.allclose(_age_roc_prob, _age_roc_check, rtol=0, atol=1e-12):
        raise ValueError("Saved probabilities do not match the locked model/current participant order.")

    age_roc_bootstrap_iterations = 5000
    age_roc_random_state = int(random_state)
    _age_roc_rng = np.random.default_rng(age_roc_random_state)
    _age_roc_groups = [
        ("65–74 years", (_age_roc_ages >= 65) & (_age_roc_ages < 75), "#1F5A94", "-"),
        ("≥75 years", _age_roc_ages >= 75, "#D07A27", "--"),
    ]
    _age_roc_rows, _age_roc_curve_rows, _age_roc_plot_data = [], [], []

    for _age_roc_label, _age_roc_mask, _age_roc_color, _age_roc_linestyle in _age_roc_groups:
        _age_roc_group_y = _age_roc_y[_age_roc_mask]
        _age_roc_group_prob = _age_roc_prob[_age_roc_mask]
        _age_roc_pos_idx = np.flatnonzero(_age_roc_group_y == 1)
        _age_roc_neg_idx = np.flatnonzero(_age_roc_group_y == 0)
        if not len(_age_roc_pos_idx) or not len(_age_roc_neg_idx):
            raise ValueError(f"{_age_roc_label} must contain both cases and non-cases for a ROC curve.")

        _age_roc_pred = (_age_roc_group_prob >= _age_roc_threshold).astype(int)
        _age_roc_tn, _age_roc_fp, _age_roc_fn, _age_roc_tp = confusion_matrix(
            _age_roc_group_y, _age_roc_pred, labels=[0, 1]
        ).ravel()
        _age_roc_auc = roc_auc_score(_age_roc_group_y, _age_roc_group_prob)
        _age_roc_sensitivity = _age_roc_tp / (_age_roc_tp + _age_roc_fn)
        _age_roc_specificity = _age_roc_tn / (_age_roc_tn + _age_roc_fp)
        _age_roc_fpr, _age_roc_tpr, _age_roc_cutoffs = roc_curve(
            _age_roc_group_y, _age_roc_group_prob, drop_intermediate=False
        )

        # Within each age group, resample cases and non-cases separately.
        # These are subgroup-specific percentile CIs, conditional on the fitted model.
        _age_roc_boot = np.empty((age_roc_bootstrap_iterations, 3), dtype=float)
        for _age_roc_b in range(age_roc_bootstrap_iterations):
            _age_roc_sample_pos = _age_roc_rng.choice(
                _age_roc_pos_idx, size=len(_age_roc_pos_idx), replace=True
            )
            _age_roc_sample_neg = _age_roc_rng.choice(
                _age_roc_neg_idx, size=len(_age_roc_neg_idx), replace=True
            )
            _age_roc_sample = np.concatenate([_age_roc_sample_pos, _age_roc_sample_neg])
            _age_roc_boot[_age_roc_b] = [
                roc_auc_score(_age_roc_group_y[_age_roc_sample], _age_roc_group_prob[_age_roc_sample]),
                _age_roc_pred[_age_roc_sample_pos].mean(),
                1 - _age_roc_pred[_age_roc_sample_neg].mean(),
            ]
        _age_roc_ci_low, _age_roc_ci_high = np.percentile(_age_roc_boot, [2.5, 97.5], axis=0)
        _age_roc_row = {
            "age_group": _age_roc_label,
            "N": len(_age_roc_group_y),
            "Events": int(_age_roc_group_y.sum()),
            "Non_events": int((_age_roc_group_y == 0).sum()),
            "Threshold": _age_roc_threshold,
            "AUC": _age_roc_auc,
            "AUC_CI_lower_95": _age_roc_ci_low[0],
            "AUC_CI_upper_95": _age_roc_ci_high[0],
            "Sensitivity": _age_roc_sensitivity,
            "Sensitivity_CI_lower_95": _age_roc_ci_low[1],
            "Sensitivity_CI_upper_95": _age_roc_ci_high[1],
            "Specificity": _age_roc_specificity,
            "Specificity_CI_lower_95": _age_roc_ci_low[2],
            "Specificity_CI_upper_95": _age_roc_ci_high[2],
            "TN": int(_age_roc_tn), "FP": int(_age_roc_fp),
            "FN": int(_age_roc_fn), "TP": int(_age_roc_tp),
            "Bootstrap_iterations": age_roc_bootstrap_iterations,
            "Bootstrap_random_state": age_roc_random_state,
        }
        _age_roc_rows.append(_age_roc_row)
        _age_roc_curve_rows.extend(
            {"age_group": _age_roc_label, "FPR": fpr, "TPR": tpr, "threshold": threshold}
            for fpr, tpr, threshold in zip(_age_roc_fpr, _age_roc_tpr, _age_roc_cutoffs)
        )
        _age_roc_plot_data.append(
            (_age_roc_row, _age_roc_fpr, _age_roc_tpr, _age_roc_color, _age_roc_linestyle)
        )

    age_roc_metrics_2024 = pd.DataFrame(_age_roc_rows)
    age_roc_coordinates_2024 = pd.DataFrame(_age_roc_curve_rows)
    assert int(age_roc_metrics_2024["N"].sum()) == len(valid_df)
    assert int(age_roc_metrics_2024["Events"].sum()) == int(_age_roc_y.sum())

    _age_roc_dir = Path(output_dir) / "reviewer_figures"
    _age_roc_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        from matplotlib.patches import FancyBboxPatch as age_roc_FancyBboxPatch

        # Equal physical width/height and equal limits give a centered square ROC axis.
        _age_roc_fig = plt.figure(figsize=(7.4, 6.4))
        _age_roc_side = 4.7  # inches; square plotting area within the standard single-panel canvas
        _age_roc_left = (7.4 - _age_roc_side) / (2 * 7.4)
        _age_roc_bottom = 0.16
        _age_roc_width, _age_roc_height = _age_roc_side / 7.4, _age_roc_side / 6.4
        _age_roc_ax = _age_roc_fig.add_axes([
            _age_roc_left, _age_roc_bottom, _age_roc_width, _age_roc_height
        ])
        _age_roc_ax.set(
            xlabel="1 – Specificity", ylabel="Sensitivity",
            xlim=(-0.02, 1.02), ylim=(-0.02, 1.02)
        )
        _age_roc_ax.set_aspect("equal", adjustable="box")
        _age_roc_ax.set_xticks(np.linspace(0, 1, 6))
        _age_roc_ax.set_yticks(np.linspace(0, 1, 6))
        _age_roc_ax.set_axisbelow(True)
        _age_roc_ax.grid(True, color="#E5E5E5", linewidth=0.65, zorder=0)
        _age_roc_ax.tick_params(axis="both", labelsize=10, length=4, pad=5)
        _age_roc_ax.xaxis.label.set_size(11)
        _age_roc_ax.yaxis.label.set_size(11)
        _age_roc_ax.xaxis.labelpad = 10
        _age_roc_ax.yaxis.labelpad = 10
        for _age_roc_spine in _age_roc_ax.spines.values():
            _age_roc_spine.set_color("#666666")
            _age_roc_spine.set_linewidth(0.8)

        _age_roc_ax.plot(
            [0, 1], [0, 1], color="#8A8A8A", linestyle=":", linewidth=1.2, zorder=1
        )
        _age_roc_card_text_artists = []
        for _age_roc_i, (_age_roc_row, _age_roc_fpr, _age_roc_tpr,
                         _age_roc_color, _age_roc_linestyle) in enumerate(_age_roc_plot_data):
            _age_roc_ax.plot(
                _age_roc_fpr, _age_roc_tpr, color=_age_roc_color,
                linestyle=_age_roc_linestyle, linewidth=2.1, zorder=3
            )
            _age_roc_ax.scatter(
                1 - _age_roc_row["Specificity"], _age_roc_row["Sensitivity"],
                marker="D", s=48, facecolor=_age_roc_color,
                edgecolor="white", linewidth=1.0, zorder=5
            )

            # Both cards lie below y=x and below the empirical ROC curves.
            _age_roc_card_x = [0.34, 0.67][_age_roc_i]
            _age_roc_card_y, _age_roc_card_w, _age_roc_card_h = 0.025, 0.31, 0.29
            _age_roc_ax.add_patch(age_roc_FancyBboxPatch(
                (_age_roc_card_x, _age_roc_card_y), _age_roc_card_w, _age_roc_card_h,
                boxstyle="round,pad=0.006,rounding_size=0.008",
                transform=_age_roc_ax.transAxes, facecolor="#FFFFFF", alpha=0.96,
                edgecolor="#DDDDDD", linewidth=0.65, zorder=4
            ))
            _age_roc_text_x = _age_roc_card_x + 0.014
            _age_roc_header_y = _age_roc_card_y + _age_roc_card_h - 0.034
            _age_roc_ax.plot(
                [_age_roc_text_x, _age_roc_text_x + 0.026],
                [_age_roc_header_y, _age_roc_header_y],
                transform=_age_roc_ax.transAxes, color=_age_roc_color,
                linestyle=_age_roc_linestyle, linewidth=2.0, zorder=6
            )
            _age_roc_card_text_artists.append(_age_roc_ax.text(
                _age_roc_text_x + 0.038, _age_roc_header_y,
                _age_roc_row["age_group"], transform=_age_roc_ax.transAxes,
                ha="left", va="center", fontsize=9.0, weight="bold", zorder=6
            ))
            _age_roc_card_entries = [
                (f"n = {_age_roc_row['N']:,}; events = {_age_roc_row['Events']}", 7.8, "normal", "#666666"),
                (f"ROC-AUC = {_age_roc_row['AUC']:.3f}", 9.0, "bold", "#111111"),
                (f"95% CI: {_age_roc_row['AUC_CI_lower_95']:.3f}–{_age_roc_row['AUC_CI_upper_95']:.3f}", 7.9, "normal", "#333333"),
                (f"Sensitivity = {_age_roc_row['Sensitivity']:.3f}", 8.2, "normal", "#333333"),
                (f"Specificity = {_age_roc_row['Specificity']:.3f}", 8.2, "normal", "#333333"),
            ]
            for _age_roc_j, (_age_roc_text, _age_roc_size, _age_roc_weight,
                             _age_roc_text_color) in enumerate(_age_roc_card_entries):
                _age_roc_card_text_artists.append(_age_roc_ax.text(
                    _age_roc_text_x, _age_roc_header_y - 0.045 * (_age_roc_j + 1),
                    _age_roc_text, transform=_age_roc_ax.transAxes,
                    ha="left", va="center", fontsize=_age_roc_size,
                    weight=_age_roc_weight, color=_age_roc_text_color, zorder=6
                ))

        # The short key sits above the cards, in the same clear lower-right region.
        _age_roc_ax.legend(
            [Line2D([0], [0], color="#8A8A8A", linestyle=":", linewidth=1.2),
             Line2D([0], [0], color="none", marker="D", markersize=5.5,
                           markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=1)],
            ["Chance level", "Locked-threshold operating point"],
            loc="center left", bbox_to_anchor=(0.42, 0.37),
            frameon=False, fontsize=8.0, handlelength=2.2,
            handletextpad=0.65, labelspacing=0.6, borderaxespad=0
        )

        _age_roc_fig.suptitle("Age-Stratified ROC Curves", fontsize=16, weight="bold", y=0.985)
        _age_roc_fig.text(
            0.5, 0.931,
            "LR Model 2 | 2024 temporal validation | "
            f"2022 OOF threshold = {_age_roc_threshold:.6f}",
            ha="center", va="center", fontsize=8.5, color="#555555"
        )
        _age_roc_fig.text(
            _age_roc_left, 0.028,
            "95% CIs: 5,000 outcome-stratified bootstrap resamples per age group.\n"
            "Model and threshold fixed; subgroup estimates reflect limited event counts.",
            fontsize=8, color="#555555", ha="left", va="bottom", linespacing=1.5
        )
        _age_roc_fig.savefig(
            _age_roc_dir / "ROC_age_groups_2024_LR_Model_2.png", dpi=600,
            facecolor="white"
        )
        _age_roc_fig.savefig(
            _age_roc_dir / "ROC_age_groups_2024_LR_Model_2.pdf",
            facecolor="white"
        )
        plt.close(_age_roc_fig)

    age_roc_metrics_2024.to_csv(
        _age_roc_dir / "ROC_age_groups_2024_LR_Model_2_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    age_roc_coordinates_2024.to_csv(
        _age_roc_dir / "ROC_age_groups_2024_LR_Model_2_coordinates.csv",
        index=False, encoding="utf-8-sig"
    )
    return age_roc_metrics_2024, age_roc_coordinates_2024
