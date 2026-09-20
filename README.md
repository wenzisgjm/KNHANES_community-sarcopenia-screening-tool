# Community-Based Sarcopenia Screening Tool for Older Adults

This repository contains the source code, locked model artifacts, synthetic examples, and software-validation checks for the web-based screening application accompanying the manuscript.

The application was developed with Streamlit and is deployed on Render. It provides preliminary sarcopenia risk screening for adults aged 65 years and older. It is intended for research and preliminary screening only. It is not a validated clinical decision-support system, does not provide a diagnosis, and must not be used as the sole basis for treatment or referral decisions.

## Versions

- Locked model version: `1.0.0`
- Web application version: `1.1.2`
- Python: `3.12.13`

The model and web application are versioned separately. Interface, documentation, or validation changes do not imply that the locked model was retrained.

## Screening mode

Users can upload batch data in CSV or Excel format. The current release does not provide manual single-participant data entry.

Supported file types:

- CSV (`.csv`)
- Excel Workbook (`.xlsx`)

Each row represents one participant. The required columns are:

```text
ID, sex, age, EQ5D, pa_aerobic, is_allownc, Chew_Diff,
Prot_Deficiency, N_EN, HE_wc, obe_4class
```

`ID` should be a non-identifying participant code. `ID` and `sex` are retained for result management and are not used as model predictors. Age must be numeric, with ages 80 years or older coded as 80. For `is_allownc`, current or previous National Basic Livelihood Security receipt is coded as 1 and no history as 0. Detailed definitions and coding rules are displayed in the application under **How to Use** and **Model Info**.

## Output

The downloadable results contain the original input columns and:

- `model_score`: the model-generated screening score.
- `screening_result`: `Screen positive` or `Screen negative`.
- `recommended_action`: the next-step guidance associated with the screening classification.

The score is a screening estimate, not a confirmed individual clinical probability. A screen-positive result requires standardised sarcopenia assessment according to the applicable local protocol.

## Repository contents

- `analysis/`: manuscript model-development and temporal-validation source code, run instructions, and pinned dependencies.
- `app.py`: Streamlit user interface.
- `screening_core.py`: prediction, file-reading, result-action, and input-validation logic.
- `best_sarcopenia_lr_model2.pkl`: locked logistic-regression pipeline.
- `best_features_model2.pkl`: ordered model feature list.
- `best_threshold_model2.pkl`: locked classification threshold.
- `model_metadata.pkl`: model metadata used by the application.
- `examples/`: synthetic input and expected output files.
- `tests/`: automated software-validation tests.
- `supplementary_material/`: manuscript-ready interface screenshots generated with synthetic data.
- `USER_GUIDE.md`: illustrated user instructions, input coding, result interpretation, and troubleshooting.
- `output/pdf/USER_GUIDE_v1.1.2.pdf`: current submission-ready illustrated user manual.
- `VALIDATION.md`: software-validation scope, procedures, and temporal-validation results.
- `requirements.txt`: pinned runtime dependencies.
- `.python-version`: pinned Python version for deployment.
- `render.yaml`: Render deployment configuration.

## Local installation

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local address displayed by Streamlit, usually `http://localhost:8501`.

## Run the validation checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests verify locked artifact metadata, threshold behavior, input validation, recommended actions, and prediction parity against the synthetic expected-output file. See `VALIDATION.md` for details.

## User documentation

See `USER_GUIDE.md` for the accessible web version of the illustrated user manual. A fixed-layout PDF version suitable for supplementary-material submission is available at `output/pdf/USER_GUIDE_v1.1.2.pdf`.

The same PDF can be downloaded directly from the web application under **How to Use > User Manual**.

## Important limitations

- The model is restricted to its intended population and setting.
- A screen-positive result is not a diagnosis.
- A screen-negative result does not rule out sarcopenia when symptoms, functional decline, or clinical concern are present.
- Model performance may change across populations, locations, and time periods.
- Local validation, calibration monitoring, privacy safeguards, and clinical governance are required before operational use.

## Code availability

This repository contains the manuscript analysis code (`analysis/`), the Streamlit web application source, locked model artifacts, and software-validation materials. Reviewers and readers can run the analysis script with separately obtained study CSVs or inspect and run the application with the locked model and synthetic examples. Participant-level research data and analysis result files are not included.

## License

This project is released under the MIT License. See `LICENSE`.
