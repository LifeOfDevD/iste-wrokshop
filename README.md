# House Price Prediction — Supervised Machine Learning Pipeline

> **End-to-End Data Science & Regression Workflow**  
> *Data Cleaning • Exploratory Data Analysis • Feature Normalization • Scikit-Learn Linear Regression*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Data-Pandas%20%26%20NumPy-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## 📊 Overview

This project implements an end-to-end predictive machine learning pipeline for real estate valuation, developed for the **ISTE Thapar Chapter Machine Learning Workshop**. 

It handles dirty, uncurated raw tabular data (`house_price_dirty.csv`) with missing attributes and outliers, cleans and standardizes numeric distributions, and trains a multivariate linear regression model to predict residential house valuations.

---

## 🔬 Pipeline Architecture

```
Raw CSV Dataset (house_price_dirty.csv)
                    │
                    ▼
     [ Stage 1: Data Cleansing ]
     - Missing value imputation
     - Duplicate elimination
     - Type coercion & outlier filtering
                    │
                    ▼
     [ Stage 2: Feature Transformation ]
     - Numerical scaling via StandardScaler (Z-score normalization)
     - Correlation matrix evaluation
                    │
                    ▼
     [ Stage 3: Train / Test Split ]
     - 80/20 train-test partitioning
                    │
                    ▼
     [ Stage 4: Supervised Modeling ]
     - Multivariate Linear Regression (OLS)
                    │
                    ▼
     [ Stage 5: Validation & Metrics ]
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - Coefficient of Determination (R² Score)
```

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.9+
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (`LinearRegression`, `StandardScaler`, `train_test_split`)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/LifeOfDevD/iste-wrokshop.git
cd iste-wrokshop

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Execute pipeline
python projectone.py
```
