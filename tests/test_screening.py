import io
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from screening_core import (
    SCREEN_NEGATIVE_ACTION,
    SCREEN_POSITIVE_ACTION,
    predict,
    read_uploaded_table,
    validate_batch_data,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "age",
    "EQ5D",
    "pa_aerobic",
    "is_allownc",
    "Chew_Diff",
    "Prot_Deficiency",
    "N_EN",
    "HE_wc",
    "obe_4class",
]


class NamedBytesIO(io.BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


class FixedProbabilityModel:
    def __init__(self, positive_probabilities):
        self.positive_probabilities = np.asarray(positive_probabilities, dtype=float)

    def predict_proba(self, data):
        negative_probabilities = 1.0 - self.positive_probabilities
        return np.column_stack([negative_probabilities, self.positive_probabilities])


class ScreeningValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(REPOSITORY_ROOT / "best_sarcopenia_lr_model2.pkl")
        cls.stored_features = joblib.load(REPOSITORY_ROOT / "best_features_model2.pkl")
        cls.threshold = float(joblib.load(REPOSITORY_ROOT / "best_threshold_model2.pkl"))
        cls.metadata = joblib.load(REPOSITORY_ROOT / "model_metadata.pkl")

    def test_locked_artifact_metadata(self):
        self.assertEqual(self.metadata["final_algorithm"], "LogisticRegression")
        self.assertEqual(self.metadata["final_feature_set"], "Model_2")
        self.assertEqual(list(self.stored_features), FEATURES)
        self.assertEqual(list(self.metadata["features"]), FEATURES)
        self.assertAlmostEqual(self.threshold, 0.082884, places=12)
        self.assertAlmostEqual(float(self.metadata["threshold"]), self.threshold, places=12)

    def test_threshold_boundary_and_recommended_actions(self):
        scores = [self.threshold - 1e-12, self.threshold, self.threshold + 1e-12]
        model = FixedProbabilityModel(scores)
        output = predict(model, pd.DataFrame({"test_value": [1, 2, 3]}), self.threshold)

        self.assertEqual(output["predicted_class"].tolist(), [0, 1, 1])
        self.assertEqual(
            output["screening_result"].tolist(),
            ["Screen negative", "Screen positive", "Screen positive"],
        )
        self.assertEqual(
            output["recommended_action"].tolist(),
            [SCREEN_NEGATIVE_ACTION, SCREEN_POSITIVE_ACTION, SCREEN_POSITIVE_ACTION],
        )

    def test_invalid_age_and_category_codes_are_rejected(self):
        data = pd.DataFrame(
            {
                "ID": ["SYN-INVALID"],
                "sex": ["F"],
                "age": [64],
                "EQ5D": [0.8],
                "pa_aerobic": [2],
                "is_allownc": [0],
                "Chew_Diff": [0],
                "Prot_Deficiency": [0],
                "N_EN": [1600],
                "HE_wc": [80],
                "obe_4class": [4],
            }
        )
        _, errors, _ = validate_batch_data(data, FEATURES)
        combined = " ".join(errors)

        self.assertIn("younger than 65", combined)
        self.assertIn("pa_aerobic contains invalid code", combined)
        self.assertIn("obe_4class contains invalid code", combined)

    def test_missing_non_age_predictor_generates_imputation_warning(self):
        data = pd.DataFrame(
            {
                "ID": ["SYN-MISSING"],
                "sex": ["M"],
                "age": [72],
                "EQ5D": [np.nan],
                "pa_aerobic": [0],
                "is_allownc": [0],
                "Chew_Diff": [0],
                "Prot_Deficiency": [0],
                "N_EN": [1700],
                "HE_wc": [86],
                "obe_4class": [2],
            }
        )
        _, errors, warnings = validate_batch_data(data, FEATURES)

        self.assertEqual(errors, [])
        self.assertTrue(any("EQ5D (1)" in warning for warning in warnings))

    def test_age_above_80_requires_top_coding(self):
        data = pd.DataFrame(
            {
                "ID": ["SYN-AGE"],
                "sex": ["F"],
                "age": [81],
                "EQ5D": [0.8],
                "pa_aerobic": [0],
                "is_allownc": [0],
                "Chew_Diff": [0],
                "Prot_Deficiency": [0],
                "N_EN": [1600],
                "HE_wc": [80],
                "obe_4class": [0],
            }
        )

        _, errors, _ = validate_batch_data(data, FEATURES)
        self.assertTrue(any("must be coded as 80" in error for error in errors))

        data.loc[0, "age"] = 80
        _, errors, _ = validate_batch_data(data, FEATURES)
        self.assertEqual(errors, [])

    def test_supported_batch_file_formats(self):
        source = pd.DataFrame({"ID": ["SYN-FILE"], "age": [70]})

        csv_bytes = source.to_csv(index=False).encode("utf-8")
        csv_loaded = read_uploaded_table(NamedBytesIO(csv_bytes, "sample.csv"))
        pd.testing.assert_frame_equal(csv_loaded, source)

        xlsx_buffer = io.BytesIO()
        source.to_excel(xlsx_buffer, index=False)
        xlsx_loaded = read_uploaded_table(
            NamedBytesIO(xlsx_buffer.getvalue(), "sample.xlsx")
        )
        pd.testing.assert_frame_equal(xlsx_loaded, source)

        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            read_uploaded_table(NamedBytesIO(b"", "legacy.xls"))

    def test_synthetic_prediction_parity(self):
        sample = pd.read_csv(REPOSITORY_ROOT / "examples" / "sample_input.csv")
        expected = pd.read_csv(REPOSITORY_ROOT / "examples" / "expected_output.csv")

        validated, errors, _ = validate_batch_data(sample, FEATURES)
        self.assertEqual(errors, [])

        output = predict(self.model, validated[FEATURES], self.threshold)
        np.testing.assert_allclose(
            output["model_score"].to_numpy(),
            expected["model_score"].to_numpy(),
            rtol=0.0,
            atol=1e-10,
        )
        self.assertEqual(
            output["screening_result"].tolist(),
            expected["screening_result"].tolist(),
        )
        self.assertEqual(
            output["recommended_action"].tolist(),
            expected["recommended_action"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
