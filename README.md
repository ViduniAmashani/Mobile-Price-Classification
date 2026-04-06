# Smart Phone Price Prediction

Predict mobile phone price range classes using supervised machine learning.

This project compares two classification models:
- K-Nearest Neighbors (KNN)
- Random Forest

The full workflow is implemented in notebooks, from exploratory analysis and preprocessing to model training, evaluation, and comparison.

## Project Overview

The objective is to classify smartphones into `price_range` categories using hardware and feature specifications.

Current pipeline includes:
- Exploratory Data Analysis (EDA)
- Data cleaning and feature preprocessing
- KNN training and evaluation
- Random Forest hyperparameter tuning and evaluation
- Side-by-side model comparison

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
		│   ├── Heatmap.png
		│   ├── KNN-values_vs_accuracies.png
		│   └── ram_vs_pricerange.png
		└── metrics/
```

## Dataset

### Raw Files
- `data/raw/train.csv`: 2000 rows, includes target `price_range`
- `data/raw/test.csv`: 1000 rows, includes `id` and feature columns

### Train Feature Columns
- `battery_power`
- `blue`
- `clock_speed`
- `dual_sim`
- `fc`
- `four_g`
- `int_memory`
- `m_dep`
- `mobile_wt`
- `n_cores`
- `pc`
- `px_height`
- `px_width`
- `ram`
- `sc_h`
- `sc_w`
- `talk_time`
- `three_g`
- `touch_screen`
- `wifi`

Target:
- `price_range`

## Workflow

### 1) EDA (`notebooks/01_EDA.ipynb`)
- Loads raw training data
- Checks shape, info, summary stats, missing values
- Reviews class distribution
- Creates feature distribution plots and correlation heatmap
- Notes observed data-quality issues used later in preprocessing

### 2) Preprocessing (`notebooks/02_Preprocessing.ipynb`)
- Cleans invalid or problematic values (for example, zero `px_height` rows)
- Handles `sc_w` zero values
- Applies log transformation to selected skewed features
- Drops low-correlation features
- Splits train/test (`80/20`, stratified)
- Applies `StandardScaler`
- Saves processed arrays to `data/processed/`

### 3) KNN Model (`notebooks/03_KNN_Model.ipynb`)
- Searches K values with cross-validation
- Plots K vs accuracy
- Trains final KNN model
- Evaluates with accuracy, classification report, confusion matrix
- Saves model to `models/knn_model.pkl`

### 4) Random Forest (`notebooks/04_RandomForest.ipynb`)
- Performs hyperparameter tuning using `GridSearchCV`
- Trains final Random Forest model using selected parameters
- Evaluates with accuracy, classification report, confusion matrix
- Saves model to `models/rf_model.pkl`

### 5) Model Comparison (`notebooks/05_Comparison.ipynb`)
- Compares KNN and Random Forest performance
- Summarizes metrics in table form
- Visualizes confusion matrices

## Reported Results

From notebook outputs:

- KNN
	- Accuracy: `0.8375`
	- Macro F1: `0.84`
	- Notebook comparison notes tuned setting around `K=36` with distance weighting

- Random Forest
	- Accuracy: `0.9175`
	- Macro F1: `0.92`
	- Best CV score (reported): `0.8998667711598747`
	- Reported best params:
		- `bootstrap=True`
		- `max_depth=20`
		- `min_samples_leaf=1`
		- `min_samples_split=5`
		- `n_estimators=100`

Random Forest is the stronger model in the current implementation.

## Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd smart_phone_price_prediction
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## How To Run

Launch Jupyter and run notebooks in order:

```bash
jupyter notebook
```

Recommended order:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Preprocessing.ipynb`
3. `notebooks/03_KNN_Model.ipynb`
4. `notebooks/04_RandomForest.ipynb`
5. `notebooks/05_Comparison.ipynb`

## Model Artifacts

Saved models:
- `models/knn_model.pkl`
- `models/rf_model.pkl`

Processed arrays:
- `data/processed/X_train_scaled.npy`
- `data/processed/X_test_scaled.npy`
- `data/processed/y_train.npy`
- `data/processed/y_test.npy`

## Notes

- `data/processed/processed_data.csv` currently exists but is empty.
- The primary reusable outputs are the `.npy` processed arrays and `.pkl` trained models.

## Future Improvements

- Add a standalone inference script (CLI/API) for production prediction
- Add automated metrics export to `results/metrics/`
- Add unit tests for preprocessing and model consistency
- Add model versioning and experiment tracking

