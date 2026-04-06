# Smart Phone Price Prediction

This project predicts mobile phone price range classes from device specifications using supervised machine learning.

Implemented models:
- K-Nearest Neighbors (KNN)
- Random Forest

The end-to-end pipeline is notebook-based: EDA -> preprocessing -> model training -> comparison.

## Project Goal

Classify each device into a `price_range` category using hardware and connectivity features such as `ram`, `battery_power`, `px_width`, and others.

## Current Repository Layout

```text
smart_phone_price_prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── processed_data.csv
├── models/
│   ├── knn_model.pkl
│   └── rf_model.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_KNN_Model.ipynb
│   ├── 04_RandomForest.ipynb
│   ├── 05_Comparison.ipynb
│   └── test.ipynb
└── results/
    ├── figures/
    │   ├── feature_distribution.png
    │   ├── final_features.png
    │   ├── KNN-values_vs_accuracies.png
    │   ├── ram_vs_pricerange.png
    │   └── rf_feature_importance.png
    └── metrics/
        ├── Heatmap.png
        └── knn_and_rf_confusion_matrices.png
```

## Data Summary

- `data/raw/train.csv`: 2000 rows (includes target `price_range`)
- `data/raw/test.csv`: 1000 rows (includes `id` + feature columns)

Main feature columns:
- `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`

Target:
- `price_range`

## Notebook Workflow

1. `notebooks/01_EDA.ipynb`
- Data inspection, class balance check, feature distributions, correlation analysis

2. `notebooks/02_Preprocessing.ipynb`
- Data cleaning (for example invalid zero values), transforms, train/test split, scaling, and saving processed arrays

3. `notebooks/03_KNN_Model.ipynb`
- KNN training/tuning, evaluation, confusion matrix, model export

4. `notebooks/04_RandomForest.ipynb`
- Random Forest hyperparameter search, final model training, feature importance, model export

5. `notebooks/05_Comparison.ipynb`
- Side-by-side model comparison and confusion matrix visualization
- Exports metrics and figures generated during comparison

## Latest Reported Performance

- KNN
  - Accuracy: `0.8375`
  - Macro F1: `~0.84`

- Random Forest
  - Accuracy: `0.9175`
  - Macro F1: `~0.92`
  - Best CV score (reported): `0.8998667711598747`
  - Reported best params: `bootstrap=True`, `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=5`, `n_estimators=100`

Random Forest currently performs better than KNN on the validation/test split used in notebooks.

## Environment Setup

1) Create and activate virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Launch Jupyter:

```bash
jupyter notebook
```

Run notebooks in this order:
1. `01_EDA.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_KNN_Model.ipynb`
4. `04_RandomForest.ipynb`
5. `05_Comparison.ipynb`

## Artifacts

Reusable artifacts:
- Processed arrays in `data/processed/`
- Trained models in `models/`

Comparison outputs are saved to:
- `results/metrics/`
- `results/figures/`

## Notes

- `data/processed/processed_data.csv` exists but may be empty depending on the executed preprocessing path.
- Most reproducible downstream inputs are the saved `.npy` arrays and `.pkl` model files.


