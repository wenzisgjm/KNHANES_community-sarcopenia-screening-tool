from pathlib import Path

import numpy as np
import pandas as pd


BINARY_FEATURES = ["pa_aerobic", "is_allownc", "Chew_Diff", "Prot_Deficiency"]

SCREEN_POSITIVE_ACTION = (
    "Arrange a standardised sarcopenia assessment according to the local protocol."
)
SCREEN_NEGATIVE_ACTION = (
    "No automatic referral based on this tool alone; assess further if symptoms or clinical concern are present."
)


def predict(model, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Generate model scores, screening classifications, and recommended actions."""
    probabilities = model.predict_proba(data)[:, 1]
    predicted_class = (probabilities >= threshold).astype(int)

    output = data.copy()
    output["model_score"] = probabilities
    output["predicted_class"] = predicted_class
    output["screening_result"] = np.where(
        predicted_class == 1, "Screen positive", "Screen negative"
    )
    output["recommended_action"] = np.where(
        predicted_class == 1,
        SCREEN_POSITIVE_ACTION,
        SCREEN_NEGATIVE_ACTION,
    )
    return output


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read a supported batch-screening file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or XLSX file.")


def validate_batch_data(data: pd.DataFrame, feature_columns: list[str]):
    """Validate model inputs while preserving non-model identification columns."""
    validated = data.copy()
    errors = []
    warnings = []

    if validated.empty:
        errors.append("The uploaded file contains no participant rows.")
        return validated, errors, warnings

    for column in feature_columns:
        original = validated[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            errors.append(f"{column} contains {invalid_count} non-numeric value(s).")
        validated[column] = converted

    if validated["age"].isna().any():
        errors.append("Age is required for every participant and cannot be missing.")
    if (validated["age"] < 65).any():
        count = int((validated["age"] < 65).sum())
        errors.append(
            f"{count} participant(s) are younger than 65 years. "
            "The model should only be used within its intended population."
        )
    if (validated["age"] > 80).any():
        count = int((validated["age"] > 80).sum())
        errors.append(
            f"{count} participant(s) have age values above 80. "
            "Ages 80 years or older must be coded as 80."
        )

    for column in BINARY_FEATURES:
        invalid_codes = sorted(
            validated.loc[
                validated[column].notna() & ~validated[column].isin([0, 1]), column
            ].unique().tolist()
        )
        if invalid_codes:
            errors.append(
                f"{column} contains invalid code(s): {invalid_codes}. Allowed codes are 0 and 1."
            )

    weight_codes = validated.loc[
        validated["obe_4class"].notna()
        & ~validated["obe_4class"].isin([0, 1, 2, 3]),
        "obe_4class",
    ].unique().tolist()
    if len(weight_codes):
        errors.append(
            f"obe_4class contains invalid code(s): {sorted(weight_codes)}. "
            "Allowed codes are 0, 1, 2, and 3."
        )

    missing_counts = validated[feature_columns].isna().sum()
    imputed_columns = [
        f"{column} ({int(count)})"
        for column, count in missing_counts.items()
        if count and column != "age"
    ]
    if imputed_columns:
        warnings.append(
            "Missing model inputs will be imputed by the locked preprocessing pipeline: "
            + ", ".join(imputed_columns)
            + "."
        )

    if validated["ID"].isna().any():
        warnings.append(
            "One or more participant identifiers are missing, which may make follow-up difficult."
        )
    if validated["ID"].duplicated().any():
        warnings.append(
            "Duplicate participant identifiers were detected. Confirm that each row represents "
            "the intended participant record."
        )

    return validated, errors, warnings
