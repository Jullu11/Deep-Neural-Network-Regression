# Deep Neural Network (Regression)

**Author:** Jainesh Lad

Regression project that predicts **median house values** for California districts using a custom Deep Neural Network built with TensorFlow/Keras Model Subclassing API and the California Housing dataset.

---

## Overview

This project implements and evaluates a **DNN for regression** with:

- Exploratory data analysis (EDA) of the California Housing dataset
- A custom **`MLPRegress`** class using TensorFlow/Keras subclassing API
- A deep MLP with multiple hidden layers and ReLU activations
- Train/test splitting, scaling, and standard regression metrics

---

## Dataset

**California Housing** (1990 U.S. Census, via scikit-learn)

| Property | Value |
|----------|--------|
| **Samples** | 20,640 |
| **Features** | 8 numerical |
| **Target** | Median house value (hundreds of thousands of dollars) |

### Features

1. **MedInc** — Median income in block group  
2. **HouseAge** — Median house age (years)  
3. **AveRooms** — Average number of rooms per household  
4. **AveBedrms** — Average number of bedrooms per household  
5. **Population** — Block group population  
6. **AveOccup** — Average household size  
7. **Latitude** — Block group latitude  
8. **Longitude** — Block group longitude  

---

## Model

**MLPRegress** — Custom MLP for regression:

- **API:** TensorFlow/Keras Model Subclassing
- **Architecture:** Funnel-style (e.g. 128 → 64 → 32 units)
- **Activations:** ReLU in hidden layers
- **Output:** Single neuron (continuous prediction)

---

## Approach

1. Load and explore the California Housing dataset  
2. Perform EDA (distributions, correlation, scatter plots)  
3. Preprocess (train/test split, `StandardScaler`)  
4. Implement and instantiate `MLPRegress`  
5. Train and evaluate (MSE, MAE, R²)  
6. Visualize and interpret results  

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## Usage

1. Open `DNN_Regression.ipynb` in Jupyter Notebook or Google Colab.  
2. Run all cells in order (the dataset is loaded via `sklearn.datasets.fetch_california_housing`).

---

## Project Structure

```
.
├── DNN_Regression.ipynb   # Main notebook
└── README.md              # This file
```

---

## License

This project is for educational purposes.
