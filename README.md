# Freight Rate Prediction Challenge

## 1. Project goal

Train a regression model on `data/train_test.csv`, predict the 12,000 rows in `data/validation.csv`, complete the supplied template, and create the fixed December 2025 rate chart.

## 2. Environment setup

Recommended: Python 3.11 or 3.12 in a virtual environment.

### Windows PowerShell

```powershell
cd freight_rate_prediction_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Alternatively, run `.venv\Scripts\python.exe` directly without activation.

### macOS/Linux

```bash
cd freight_rate_prediction_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Input files

- `data/train_test.csv`: labeled development data; target is `posted_rate`.
- `data/validation.csv`: 12,000 unlabeled loads.
- `data/validation_predictions_template.csv`: required IDs and empty prediction column.
- `data/december_chart_inputs.csv`: 31 fixed-route rows whose rates must be predicted.

## 4. Train, validate, and generate outputs

Run from the project root:

```bash
python train_model.py --data-dir data --output-dir outputs
```

The script:
1. Uses the latest month in the labeled data as a chronological holdout.
2. Trains a CatBoost regression model and uses early stopping based on holdout MAE.
3. Writes holdout metrics and holdout predictions.
4. Retrains a final model on all labeled rows using the selected iteration count.
5. Generates the validation submission and December predictions.
6. Saves the trained model.

Then run the supplied scorer:

```bash
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv --output-dir scorer_results
```

Expected outputs:
- `outputs/validation_predictions.csv`
- `outputs/december_chart_inputs.csv`
- `outputs/validation_metrics.json`
- `outputs/holdout_predictions.csv`
- `outputs/freight_rate_model.cbm`
- `scorer_results/candidate_december.png`

## 5. Model and validation approach

The target is `posted_rate`. The model uses CatBoost because it can model nonlinear relationships and categorical route/equipment features without manually one-hot encoding them. Features include origin/destination, route, equipment, distance, weight, market index, quote signal, and calendar parts derived from `date`. `load_id` is excluded from the model because it is an identifier rather than a load characteristic.

The validation split is time-ordered: the latest calendar month is held out and earlier months are used for training. This is more representative of predicting future loads than a random split. Missing numeric values are filled with medians computed from the development/training data; missing categorical values are represented as `Unknown`.

The report must state the measured metrics from `outputs/validation_metrics.json`. The local holdout metrics are not Spotter's final hidden-validation score.

## 6. Submission checklist

- [ ] `validation_predictions.csv` has exactly `load_id,predicted_rate`, in that order.
- [ ] Exactly 12,000 rows, one per required `TE-000001` through `TE-012000` ID.
- [ ] No missing, duplicate, non-numeric, or non-positive predicted rates.
- [ ] Scorer completes successfully and creates `candidate_december.png`.
- [ ] PDF/DOCX report explains data quality, split, model, metrics, and includes the December chart.
- [ ] Public/accessibly shared GitHub repository includes code, dependencies, and run instructions.
- [ ] 2–3 minute Loom video has camera on and covers findings, data quality, model choice, split, and code walkthrough.

## 7. GitHub delivery

Create a repository such as `freight-rate-prediction`. Copy the project files into it. Do not commit `.venv/`, temporary files, or secrets. Include generated deliverables if the assessment expects them; at minimum include `outputs/validation_predictions.csv`, the report, and chart.

Suggested commands:

```bash
git init
git add README.md requirements.txt train_model.py score.py data reports outputs scorer_results
git commit -m "Build freight rate prediction solution"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/freight-rate-prediction.git
git push -u origin main
```

Replace the URL with your actual repository URL. Confirm the repository is public or otherwise accessible to the assessment reviewers.

## 8. Loom outline (2–3 minutes)

- 0:00–0:25 — Objective and dataset sizes.
- 0:25–0:55 — Data-quality findings and treatment.
- 0:55–1:25 — Why CatBoost and the features used.
- 1:25–1:55 — Chronological holdout and measured metrics.
- 1:55–2:30 — Walk through `train_model.py`, generated CSV, and December chart.
- 2:30–2:45 — Show repository and final deliverables.
