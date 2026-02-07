# Build an ML Pipeline for Short-Term Rental Prices in NYC

This repository contains an end-to-end MLOps pipeline to train and evaluate a baseline regression model that predicts NYC short-term rental prices from tabular + text features. The pipeline is designed to be re-run whenever new weekly bulk data arrives.

The pipeline follows these high-level steps:

1. Fetch raw data
2. Basic cleaning and preprocessing
3. Data validation tests
4. Train/validation/test split
5. Train a Random Forest regression model
6. Select best model and test on the hold-out test set
7. Release the pipeline and re-run on a new sample (expected failure first, then fix and succeed)

---

## Project Links (for submission)

- GitHub Repo: https://github.com/sseib11/Project-Build-an-ML-Pipeline-Starter
- W&B Project: https://wandb.ai/sseib11-western-governors-university/nyc_airbnb

> Note: The rubric requires the W&B project to be public so the reviewer can access artifacts/runs.  
> If you cannot make the project public due to account/team restrictions, see **W&B Visibility / Access** below.

---

## Environment

I completed this project locally on **Windows + WSL (Ubuntu 24.04)** using **Conda** and **MLflow**.

### Required tools
- WSL Ubuntu (or Linux / macOS)
- Conda (Miniconda/Anaconda)
- Git
- Weights & Biases (W&B) account
- MLflow (installed via the project environment)

---

## Setup

### 1) Clone the repo
```bash
git clone git@github.com:sseib11/Project-Build-an-ML-Pipeline-Starter.git
cd Project-Build-an-ML-Pipeline-Starter
```

### 2) Create and activate the dev environment
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### 3) Login to Weights & Biases
Get your API key from:
https://wandb.ai/authorize

Then:
```bash
wandb login
```
Paste the key when prompted.

---

## Running the Pipeline

The pipeline is controlled by `main.py` and configuration in `config.yaml` (Hydra).

### Run the full pipeline
```bash
mlflow run .
```

### Run a single step (or multiple steps)
```bash
mlflow run . -P steps=download
mlflow run . -P steps=download,basic_cleaning
```

### Override config values via Hydra
Example:
```bash
mlflow run .   -P steps=download,basic_cleaning   -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

---

## Pipeline Steps (What each one does)

### `download`
Downloads a sample dataset (e.g., `sample1.csv` or `sample2.csv`) and logs it as a W&B artifact:
- `sample.csv` (type: `raw_data`)

### `basic_cleaning`
Cleans and filters the dataset and logs:
- `clean_sample.csv` (type: `clean_sample`)

### `data_check`
Runs data validation tests (PyTest), including:
- row count bounds
- price range bounds
- neighborhood names
- distribution drift check (KL)
- NYC boundary test

### `data_split`
Splits cleaned data into:
- `trainval_data.csv` (type: `trainval_data`)
- `test_data.csv` (type: `test_data`)

### `train_random_forest`
Trains a RandomForestRegressor with preprocessing + TF-IDF on listing name and logs:
- `random_forest_export` (type: `model_export`)
- metrics: MAE and R²
- feature importance visualization

### `test_regression_model`
Evaluates the `prod` model against the hold-out test dataset.

---

## Hyperparameter Optimization (Required)

Run a sweep using Hydra multi-run:
```bash
mlflow run .   -P steps=train_random_forest   -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

This produces multiple W&B runs with different hyperparameters.

---

## Model Selection and Production Tag (Required)

In W&B:
1. Go to **Runs** (Table view)
2. Show columns: `max_depth`, `n_estimators`, `mae`, `r2`
3. Sort by **mae** ascending
4. Choose the best run
5. Open its output artifact (model export)
6. Add alias **prod**

After adding alias `prod`, the pipeline can reference:
- `random_forest_export:prod`

---

## Test on the Test Set (Required)

```bash
mlflow run . -P steps=test_regression_model
```

The rubric expects the test-set performance to be comparable to validation (no major overfitting).

---

## Releases and Re-Running on New Data Sample (Required)

The rubric requires:

- A released version (example: `1.0.0`) that fails on `sample2.csv` due to out-of-bounds geolocation
- A follow-up released version (example: `1.0.1`) that fixes cleaning and succeeds

### Run a released version on sample2.csv
Replace `VERSION_HERE` with your release tag (e.g., `1.0.0`, `1.0.1`):

```bash
mlflow run https://github.com/sseib11/Project-Build-an-ML-Pipeline-Starter.git   -v VERSION_HERE   -P hydra_options="etl.sample='sample2.csv'"
```

### NYC boundary cleaning fix
The fix is applied in `src/basic_cleaning/run.py` and removes listings outside NYC boundaries:

```python
idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
df = df[idx].copy()
```

---

## W&B Visibility / Access (Important for Rubric)

The rubric states the W&B project should be **public** so the reviewer can access artifacts and verify steps.

If you cannot set the project to public (common causes: team/org restrictions or plan limitations), use one of these options:

### Option A (recommended): Create a public project under your personal entity
1. In W&B, create a new project under your *personal* account (not the team)
2. Set the new project visibility to **Public**
3. Re-run at least the core pipeline steps so artifacts appear in the public project

Tip: You can control W&B project/entity via environment variables before running MLflow:
```bash
export WANDB_PROJECT="nyc_airbnb"
export WANDB_ENTITY="YOUR_PERSONAL_ENTITY"
```

### Option B: Add the reviewer as a collaborator (if allowed)
If your W&B workspace is private but supports sharing:
- Invite the reviewer email as a viewer/collaborator to the project or workspace

### Option C: Include evidence in submission (last resort)
If none of the above is possible, include:
- Screenshots of W&B Artifacts page (showing `sample.csv`, `clean_sample.csv`, `trainval_data.csv`, `test_data.csv`, `random_forest_export`)
- Screenshots of run metrics (MAE / R²)
- Screenshot of pipeline lineage/graph view

---

## Troubleshooting

### Git push asks for password
GitHub does not support password authentication over HTTPS. Use SSH:

```bash
git remote set-url origin git@github.com:sseib11/Project-Build-an-ML-Pipeline-Starter.git
ssh -T git@github.com
git push
```

### Clean up MLflow conda environments (optional)
If MLflow created many environments and you need to clean them:
```bash
conda info --envs | grep mlflow | cut -f1 -d" "
for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y; done
```

---

## License
See `LICENSE.txt`.
