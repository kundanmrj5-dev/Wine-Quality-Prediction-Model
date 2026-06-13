# Wine Quality Prediction

Machine learning project to predict wine quality scores from physicochemical features using Python and Scikit-learn.

## Project Overview

This project covers the full ML workflow:

- Exploratory data analysis with Pandas, Matplotlib, and Seaborn
- Data preprocessing and train/test splitting
- Model training with Scikit-learn pipelines
- Regression model evaluation using MAE, RMSE, and R2 score
- Feature importance visualization
- Saved model artifact for future predictions

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Folder Structure

```text
wine-quality-prediction/
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── eda.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

Use the Wine Quality dataset from the UCI Machine Learning Repository.

Recommended file:

- `winequality-red.csv`

Place it here:

```text
data/winequality-red.csv
```

The CSV should contain a target column named `quality`.

Download the official red-wine dataset:

```bash
python src/download_data.py
```

If the dataset is not available, the project can generate a realistic sample dataset for demonstration:

```bash
python src/data_utils.py --create-sample
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run EDA

```bash
python src/eda.py
```

This creates plots in `outputs/`, including:

- target distribution
- correlation heatmap
- feature scatter plots

## Train Model

```bash
python src/train.py
```

This trains and compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is saved to:

```text
models/wine_quality_model.joblib
```

The model file is an active model bundle containing:

- trained Scikit-learn pipeline
- feature column order
- selected model name
- evaluation metrics
- dataset shape used during training

Evaluation results are saved to:

```text
outputs/metrics.csv
```

## Predict Wine Quality

After training, run:

```bash
python src/predict.py
```

Or pass custom values:

```bash
python src/predict.py ^
  --fixed_acidity 7.4 ^
  --volatile_acidity 0.7 ^
  --citric_acid 0.0 ^
  --residual_sugar 1.9 ^
  --chlorides 0.076 ^
  --free_sulfur_dioxide 11 ^
  --total_sulfur_dioxide 34 ^
  --density 0.9978 ^
  --pH 3.51 ^
  --sulphates 0.56 ^
  --alcohol 9.4
```

## Resume Description

Wine Quality Prediction - Machine Learning Project

- Developed a machine learning model to predict wine quality using Python and Scikit-learn.
- Performed EDA, data preprocessing, visualization, and model evaluation using Pandas, NumPy, Matplotlib, and Seaborn.
- Compared multiple regression algorithms and saved the best-performing model for future predictions.

Skills: Python, Pandas, NumPy, EDA, Data Preprocessing, Data Visualization, Matplotlib, Seaborn, Scikit-learn, Machine Learning, Model Evaluation.
