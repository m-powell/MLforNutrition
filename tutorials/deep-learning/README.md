# NHANES Glucose Prediction Pipeline

Standalone pipeline for predicting glucose levels from the NHANES dataset using a neural network. This directory is self-contained: copy it (with the CSV and paper) and run the pipeline without the parent Streamlit app.

## Quick start

1. **Setup** (one time):  
   `./setup.sh`  
   (Uses [uv](https://docs.astral.sh/uv/). Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh` if needed.)

2. **Run pipeline**:  
   `./run.sh`  
   or: `uv run python run_pipeline.py`

3. **Or run step-by-step in Jupyter**:  
   Open `Glucose_Prediction_Pipeline.ipynb` and run all cells.

Results are written to `outputs/`: `metrics.json`, `learning_curve.png`, `predicted_vs_actual.png`, `residuals.png`, `results_table.csv`, and `results_table.txt`.

## Configuration

Edit `config.yaml` to:

- **Data**: `data_path`, `target_col` (default `glucose`), optional `feature_cols`
- **Statistical significance filtering**: `use_statistical_significance_filter: true` to keep only features with **p < 0.1** (may dramatically reduce features); `feature_selection_p_value_threshold` (default 0.1)
- **Preprocessing**: `target_trim_lower_percentile` / `target_trim_upper_percentile` (default 0.05 / 0.95), `imputation` (median / mean / constant)
- **Splits**: `train_size`, `val_size`, `test_size`, `random_state`
- **Neural network**: all options under `nn` (epochs, batch_size, lr, etc.)
- **Hyperparameter optimization**: `use_hyperparameter_optimization: "optuna"` to run Optuna; options under `optuna` (n_trials, timeout). Default `"none"`.

## Pipeline steps

1. Upload (load CSV)  
2. Audit (missingness, dtypes, duplicates, constant/ID-like columns)  
3. First pre-processing (drop missing target, optional duplicates)  
4. Optional statistical significance filtering (p < 0.1)  
5. Second pre-processing (impute, StandardScaler, target trim, train/val/test split)  
6. Train neural network (optional Optuna)  
7. Performance assessment (metrics to `outputs/`)

## Paper and tutorial

The Word document in this directory includes a **Tutorial: Predicting Glucose Levels Using the Neural Network** section. To regenerate it after editing the script:  
`uv run python append_tutorial_to_paper.py`

## Requirements

- Python 3.10+
- uv (recommended) or pip

See `requirements.txt` for dependencies (torch, pandas, numpy, scikit-learn, scipy, pyyaml, optuna optional, python-docx for appending the tutorial).
