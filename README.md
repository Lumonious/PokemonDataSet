# Pokémon Legendary Classification (ML Pipeline Project)

This project aims to develop, compare, and optimize various machine learning models to classify whether a Pokémon is **legendary** based on its statistical attributes and type information.

-----

## 🔍 Objective

The main objective is to build a robust machine learning pipeline that can accurately predict the `Legendary` status of Pokémon. We will go through data loading, preprocessing, model training, evaluation, and hyperparameter optimization to identify the best-performing model.

-----

## 🔧 1. Setup and Libraries

To get started, make sure you have all the necessary Python libraries installed. You can typically install them using `pip`:

  * **NumPy**: For numerical operations.
  * **Pandas**: For data manipulation and analysis.
  * **Matplotlib**: For basic plotting.
  * **Seaborn**: For enhanced data visualizations.
  * **Scikit-learn**: For machine learning tools (preprocessing, models, evaluation).
  * **XGBoost**: Gradient boosting framework.
  * **LightGBM**: Gradient boosting framework.
  * **CatBoost**: Gradient boosting framework.

<!-- end list -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
```

-----

## 📥 2. Data Loading and Exploration

The dataset contains various Pokémon statistics. We'll start by loading it and performing an initial exploration to understand its structure.

  * Load the dataset from the provided URL:

    ```python
    url = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv"
    pokemon_df = pd.read_csv(url)
    ```

  * Explore the general structure of the dataset using `head()`, `info()`, and `describe()`.

  * Your target variable for classification is the **`Legendary`** column.

-----

## 🔍 3. Data Preprocessing

This crucial step prepares the raw data for machine learning models.

  * Check for and handle **missing values** using appropriate imputation strategies.
  * Convert **categorical variables** into numerical formats using `LabelEncoder` or `OneHotEncoder`.
  * Consider creating the following new features:
      * **`Total`**: Sum of all statistical attributes.
      * **`Is_Mono_Type`**: A boolean indicating if the Pokémon has only one type.
      * **`Attack_Defense_Ratio`**: The ratio of attack to defense.
  * Remove any unnecessary or highly correlated variables.
  * Apply `LabelEncoder` to the target variable (`Legendary`) to convert `True` to `1` and `False` to `0`.

-----

## 🔍 4. Dataset Splitting

Before training, we need to split the data into features (X) and the target variable (y), then further divide it into training and testing sets.

  * Separate features (X) and target variable (y).
  * Split the dataset into **80% training** and **20% testing** sets.

-----

## 🧠 5. Machine Learning Models

We will train and evaluate several common machine learning classification models. For each model:

  * Perform the **training** on the training set.
  * Make **predictions** on the test set.
  * Report the **accuracy, precision, recall, and F1-score** metrics.
  * **Visualize the confusion matrix**.

The models to be used are:

  * **Random Forest**
  * **Support Vector Machine (SVM)** (Use `StandardScaler` within a `Pipeline` for this and k-NN)
  * **k-Nearest Neighbors (k-NN)** (Use `StandardScaler` within a `Pipeline`)
  * **XGBoost**
  * **LightGBM**
  * **CatBoost**

-----

## 🏆 6. Determining the Best Performing Model

After evaluating all models, we'll identify the one that shows the most promising performance.

  * Compare the **accuracy scores** of all trained models.
  * Determine the model with the **best overall performance**.

-----

## 🔧 7. Hyperparameter Optimization (GridSearchCV)

Once the best model is identified, we'll fine-tune its performance using `GridSearchCV` for hyperparameter optimization.

  * Define an appropriate **parameter grid** for the chosen model.
  * Perform **cross-validation with `cv=5`**.
  * Print the **best score** and the **best parameters** found.
  * Make predictions on the test set again with the **optimized model** and compare the metrics.

-----

## 📊 8. Results Comparison and Reporting

Finally, we'll summarize and visualize the results to draw conclusions.

  * **Visualize the scores of all models** using a bar chart.
  * **Interpret why a particular model performed better** than others.
  * Extract and analyze **feature importance scores** (if applicable for the chosen model) to identify the most influential variables.

-----

## 🔁 Bonus Tasks

For an extra challenge, consider these extensions:

  * Address **class imbalance** (if any) using techniques like **SMOTE** and observe its impact on model performance.
  * Analyze model predictions based on `Type1`: Which Pokémon types are **most frequently misclassified**?

-----
