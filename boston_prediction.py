import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ---- CONFIG ----
DATA_PATH = "HousingData.csv"   # dataset file name (as requested)
TARGET = "MEDV"                 # Target column name (median house value)
RANDOM_STATE = 42
# ----------------

# ---- CHECK DATA FILE EXISTS ----
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# ---- LOAD & CLEAN DATA ----
with open(DATA_PATH, "r", newline='') as f:
    reader = csv.reader(f)
    header = next(reader)  # column names
    raw_data = []
    for row in reader:
        # convert empty strings and 'NA' -> np.nan for numeric conversion
        clean_row = [val if val.strip() != "" and val.upper() != "NA" else np.nan for val in row]
        raw_data.append(clean_row)

# Convert to float array (will be numeric features + target)
data = np.array(raw_data, dtype=np.float64)

# ---- TARGET & FEATURES ----
if TARGET not in header:
    raise ValueError(f"Target column '{TARGET}' not found in header: {header}")

target_idx = header.index(TARGET)

# Features X (all columns except target) and target y
X = np.delete(data, target_idx, axis=1)
y = data[:, target_idx]

# ---- DETECT FEATURE TYPES ----
categorical_features = []
for i, col_name in enumerate(header):
    if i == target_idx:
        continue
    col_index_in_X = i if i < target_idx else i - 1
    # ignore NaNs when checking unique values
    col_vals = X[:, col_index_in_X]
    unique_vals = np.unique(col_vals[~np.isnan(col_vals)])
    if len(unique_vals) <= 10:
        categorical_features.append(col_index_in_X)

numeric_features = [i for i in range(X.shape[1]) if i not in categorical_features]

print("Detected numeric feature indices:", numeric_features)
print("Detected categorical feature indices:", categorical_features)

# ---- TRAIN / TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---- PREPROCESSING PIPELINES ----
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ---- MODEL PIPELINE ----
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=RANDOM_STATE))
])

# ---- HYPERPARAMETER TUNING ----
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5]
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

# ---- PREDICTIONS ----
y_pred = grid.predict(X_test)

# ---- METRICS ----
mse = mean_squared_error(y_test, y_pred)   # compatible with all sklearn versions
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R²: {r2:.2f}")

# ---- SAVE MODEL ----
joblib.dump(grid.best_estimator_, "best_model.joblib")
print("Model saved as best_model.joblib")

# ---- PLOTS ----
sns.set_style("whitegrid")

# 1) Predicted vs Actual
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor="k")
plt.plot([np.nanmin(y_test), np.nanmax(y_test)],
         [np.nanmin(y_test), np.nanmax(y_test)],
         color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Predicted vs Actual Housing Prices")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=300)
plt.show()

# 2) Residuals vs Predicted
residuals = y_test - y_pred
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, edgecolor="k")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted MEDV")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig("residuals_vs_predicted.png", dpi=300)
plt.show()

# 3) Residuals Distribution
plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.tight_layout()
plt.savefig("residuals_distribution.png", dpi=300)
plt.show()

# 4) Feature Importances
best_pipeline = grid.best_estimator_
try:
    rf_model = best_pipeline.named_steps['model']
    importances = rf_model.feature_importances_
    # Attempt to get preprocessor output feature names (sklearn >= 1.0)
    try:
        feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    sorted_idx = importances.argsort()[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=300)
    plt.show()
except Exception:
    print("Feature importances not available for this model.")
