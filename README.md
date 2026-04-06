# Smart Phone Price Prediction

Notebook-based machine learning project to classify smartphone price categories (`price_range`) from device specifications.

## Highlights

- End-to-end pipeline: EDA -> preprocessing -> training -> comparison
- Two supervised models:
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Saved model artifacts for reuse
- Comparison notebook exports visual evaluation outputs

## Problem Statement

Given smartphone hardware and connectivity features (for example `ram`, `battery_power`, `px_width`), predict the price class of each device.

## Repository Structure

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

## Dataset

- `data/raw/train.csv`: 2000 rows, includes target `price_range`
- `data/raw/test.csv`: 1000 rows, includes `id` and feature columns

Main input features:
- `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`

Target:
- `price_range`

## Notebook Pipeline

1. `notebooks/01_EDA.ipynb`
  - Data overview, class distribution, feature distributions, correlation analysis

2. `notebooks/02_Preprocessing.ipynb`
  - Data cleaning, feature transforms, split, scaling, and processed array export

3. `notebooks/03_KNN_Model.ipynb`
  - KNN training/tuning, evaluation, confusion matrix, model save

4. `notebooks/04_RandomForest.ipynb`
  - Random Forest tuning/training, evaluation, feature importance, model save

5. `notebooks/05_Comparison.ipynb`
  - Model comparison and confusion matrix visualization

## Latest Reported Performance

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| KNN | 0.8375 | ~0.84 |
| Random Forest | 0.9175 | ~0.92 |

Random Forest currently outperforms KNN on the project test split.

Random Forest tuning notes from notebook output:
- Best CV score: `0.8998667711598747`
- Best params: `bootstrap=True`, `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=5`, `n_estimators=100`

## Quick Start

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Start Jupyter:

```bash
jupyter notebook
```

4) Run notebooks in order:

1. `01_EDA.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_KNN_Model.ipynb`
4. `04_RandomForest.ipynb`
5. `05_Comparison.ipynb`

## Artifacts

Primary reusable outputs:
- Processed arrays in `data/processed/`
- Trained models in `models/`

Evaluation outputs:
- Visuals in `results/figures/`
- Comparison exports in `results/metrics/`

## Notes

- `data/processed/processed_data.csv` may be empty depending on preprocessing path used.
- For reproducible downstream work, prefer `.npy` arrays and `.pkl` model files.


Updated after fixing Git email