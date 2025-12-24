# 🌤️ Weather Regressor: Recursive Climate Prediction

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Library](https://img.shields.io/badge/Library-Scikit%20Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Library](https://img.shields.io/badge/Library-XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

An advanced **Machine Learning Time Series** project designed to forecast temperatures based on historical data. This system implements a **Recursive Multi-step Forecasting** strategy, utilizing feature engineering and model benchmarking to identify the most accurate regressor.

---

## 🚀 Features

### 📊 Advanced Feature Engineering (`limpieza.py`)
- **Cyclic Encoding:** Transforms time variables (Hour, Month, Day) into **Sine/Cosine** components to preserve temporal continuity.
- **Lag Features:** Creates historical reference points (1h, 3h, 6h, 12h, 1d, 3d, 1w, 1m, 1y) to capture trends.
- **Rolling Windows:** Calculates moving averages (3h, 24h) to smooth out noise.

### 🧠 Model Benchmarking (`modelos.py`)
- **Comprehensive Testing:** Compares performance across 9 different algorithms, including:
  - **Ensemble:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost.
  - **Linear/SVR:** Ridge, Lasso, Support Vector Regressor.
  - **Neural/Neighbors:** MLP Regressor, KNN.
- **Metrics:** Evaluates using MSE, RMSE, MAE, and $R^2$ Score.

### 🔮 Recursive Forecasting (`prediccion.py`)
- **GridSearchCV:** Automatically tunes hyperparameters for the best model (`HistGradientBoosting`).
- **Recursive Strategy:** Predicts one step ahead and feeds the prediction back into the dataset as a "lag" feature to predict the next step (multi-step forecasting).
- **Visualization:** Plots historical data vs. future predictions.

---

## 🛠️ Prerequisites

To run this project, you need **Python 3.x** and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
```

---

## 📂 Project Structure
- **data.py** - Initial ETL script. Extracts raw date/time and temperature data from source CSVs.

- **limpieza.py** - The core preprocessing script. Generates all Lag and Cyclic features.

- **modelos.py** - Sandbox for training and comparing multiple regression algorithms.

- **prediccion.py** - Main production script. Performs hyperparameter tuning and executes the recursive future prediction loop.

- **entrenamiento.py** - Basic training script for quick validation.

---

## ⚡ Usage Guide
### Phase 1: Data Preparation 🧹
Run the cleaning scripts to generate the dataset with engineered features.

```bash
python data.py
python limpieza.py
```

**Output:** Generates cleaned11.csv and 2clean11.csv.

### Phase 2: Model Selection (Optional) ⚖️
If you want to see how different models perform against each other:

``` bash
python modelos.py
```

**Output:** Console log comparing MAE/MSE/R2 across all implemented models.

### Phase 3: Recursive Prediction (Main) 🔮
Run the final prediction script. This will train the optimal model and forecast future temperatures.

``` bash
python prediccion.py
```
#### Workflow of prediccion.py:

- Loads and scales the data.

- Performs GridSearchCV to find best parameters.

- Trains HistGradientBoostingRegressor.

- Predicts future hours recursively (feeding outputs back as inputs).

**Output:** Displays a graph of the forecast and saves predicciones_futuras.csv.

## 👥 Authors
- González Martínez Silvia

- López Reyes José Roberto

---

**Disclaimer: This is an academic project (Práctica 8) designed to demonstrate Time Series Regression techniques using Scikit-Learn.**
