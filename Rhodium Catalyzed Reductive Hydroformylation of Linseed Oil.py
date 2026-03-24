# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:39:07 2025

@author: rachid
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 21:12:52 2025

@author: rachid
"""

# -*- coding: utf-8 -*-
"""
Full comparative regression pipeline with:
- Linear Regression
- SVR
- Random Forest
- XGBoost
- MLP

Features:
- Scaling of X and Y (important for SVR & MLP)
- Hyperparameter search (RandomizedSearchCV)
- Metrics per output + average
- Saving results to Excel and JSON
- Average R² bar plot
- Scatter plots (observed vs predicted) for XGBoost
- SHAP summary plots for the 4 outputs + 2x2 mosaic

Author: Rachid
"""

import subprocess
import sys

# ===============================================================
#       AUTO-INSTALL REQUIRED PACKAGES (IF MISSING)
# ===============================================================
packages = [
    "pandas", "numpy", "scikit-learn", "xgboost",
    "openpyxl", "matplotlib", "joblib", "scipy",
    "shap", "Pillow"
]

for pkg in packages:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ===============================================================
#                          IMPORTS
# ===============================================================
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from scipy.stats import loguniform, randint, uniform
import matplotlib.pyplot as plt

import shap
from PIL import Image

# ===============================================================
#                       GLOBAL PARAMETERS
# ===============================================================
excel_file = "walid_data_mol.xlsx"   
sheet_name = 0                       

cv_folds = 5
n_iter = 30                          
output_dir = Path("results_regression")
output_dir.mkdir(exist_ok=True)

# ===============================================================
#                         LOAD THE DATA
# ===============================================================
print("Loading Excel file...")
df = pd.read_excel(excel_file, sheet_name=sheet_name)


features = ['Rh(μmol)', 'C=C(μmol)', 'TEA(μmol)', 'T(°C)', 't(h)', 'CO(bar)', 'H2(bar)']
targets  = ['ALD(μmol)', 'ALC(μmol)', 'CONJ(μmol)', 'SAT(μmol)']

X = df[features].copy()
Y = df[targets].copy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

n_targets = len(targets)

# ===============================================================
#          SCALING OF X (FEATURES) AND Y (TARGETS)
# ===============================================================
# X is scaled inside pipelines that need it (Linear, SVR, MLP)
x_scaler = ColumnTransformer(
    transformers=[('num', StandardScaler(), list(range(X.shape[1])))],
    remainder='drop'
)


y_scaler = StandardScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_test_scaled = y_scaler.transform(Y_test)   

# ===============================================================
#                      EVALUATION FUNCTION
# ===============================================================
def evaluate(model_name, estimator, X_test, Y_test, target_names, y_scaler=None):
    """
    Compute R2, RMSE, MAE per target and the average over all targets.
    If y_scaler is provided, we assume the model predicts scaled Y
    and we inverse-transform the predictions before computing metrics.
    """
    Y_pred = estimator.predict(X_test)
    Y_pred = np.asarray(Y_pred)

    if y_scaler is not None:
 
        Y_pred = y_scaler.inverse_transform(Y_pred)

    rows = []
    for i, t in enumerate(target_names):
        y_true = Y_test[t].values
        y_pred = Y_pred[:, i]

        rows.append({
            "Model": model_name,
            "Output": t,
            "R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
        })


    rows.append({
        "Model": model_name,
        "Output": "AVG",
        "R2": float(np.mean([r["R2"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "MAE": float(np.mean([r["MAE"] for r in rows])),
    })
    return rows


def inverse_transform_single_target(y_scaled_1d, scaler, target_index, n_targets):
    """
    Helper to inverse-transform a single target that was scaled along with others.
    y_scaled_1d: array-like shape (n_samples,)
    scaler: fitted StandardScaler on all targets
    target_index: index of the current target
    n_targets: total number of targets
    """
    tmp = np.zeros((len(y_scaled_1d), n_targets))
    tmp[:, target_index] = y_scaled_1d
    inv = scaler.inverse_transform(tmp)
    return inv[:, target_index]

# ===============================================================
#                 TRAINING & HYPERPARAMETER SEARCH
# ===============================================================
results = []
best_params = {}

# ---------------------------------------------------------------
# 1) Linear Regression
# ---------------------------------------------------------------
print("Training Linear Regression ...")

lin = Pipeline([
    ('scale', x_scaler),
    ('reg', LinearRegression())
])

lin.fit(X_train, Y_train_scaled)
results += evaluate("Linear Regression", lin, X_test, Y_test, targets, y_scaler)
best_params["Linear Regression"] = {}

# ---------------------------------------------------------------
# 2) SVR
# ---------------------------------------------------------------
print("Training SVR (with hyperparameter search) ...")

svr = Pipeline([
    ('scale', x_scaler),
    ('reg', MultiOutputRegressor(SVR()))
])

svr_param_dist = {
    "reg__estimator__C": loguniform(1e0, 1e3),          # from 1 to 1000
    "reg__estimator__gamma": loguniform(1e-4, 1e0),     # from 1e-4 to 1
    "reg__estimator__epsilon": loguniform(1e-3, 1.0),   # from 0.001 to 1
    "reg__estimator__kernel": ["rbf", "poly"]
}

svr_search = RandomizedSearchCV(
    svr,
    svr_param_dist,
    n_iter=n_iter,
    cv=cv_folds,
    n_jobs=-1,
    random_state=42,
    scoring='r2',
    refit=True
)

svr_search.fit(X_train, Y_train_scaled)
results += evaluate("SVR", svr_search.best_estimator_, X_test, Y_test, targets, y_scaler)
best_params["SVR"] = svr_search.best_params_

# ---------------------------------------------------------------
# 3) Random Forest
# ---------------------------------------------------------------
print("Training Random Forest (with hyperparameter search) ...")

rf = Pipeline([
    ('reg', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

rf_param_dist = {
    "reg__estimator__n_estimators": randint(300, 800),
    "reg__estimator__max_depth": randint(5, 30),
    "reg__estimator__min_samples_split": randint(2, 12),
    "reg__estimator__min_samples_leaf": randint(1, 6)
}

rf_search = RandomizedSearchCV(
    rf,
    rf_param_dist,
    n_iter=n_iter,
    cv=cv_folds,
    n_jobs=-1,
    random_state=42,
    scoring='r2',
    refit=True
)

rf_search.fit(X_train, Y_train_scaled)
results += evaluate("Random Forest", rf_search.best_estimator_, X_test, Y_test, targets, y_scaler)
best_params["Random Forest"] = rf_search.best_params_

# ---------------------------------------------------------------
# 4) XGBoost
# ---------------------------------------------------------------
print("Training XGBoost (with hyperparameter search) ...")

xgb = Pipeline([
    ('reg', MultiOutputRegressor(
        XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42
        )
    ))
])

xgb_param_dist = {
    "reg__estimator__n_estimators": randint(300, 1000),
    "reg__estimator__max_depth": randint(2, 8),
    "reg__estimator__learning_rate": loguniform(0.01, 0.3),
    "reg__estimator__subsample": uniform(0.6, 0.4),
    "reg__estimator__colsample_bytree": uniform(0.6, 0.4),
}

xgb_search = RandomizedSearchCV(
    xgb,
    xgb_param_dist,
    n_iter=n_iter,
    cv=cv_folds,
    n_jobs=-1,
    random_state=42,
    scoring='r2',
    refit=True
)

xgb_search.fit(X_train, Y_train_scaled)
results += evaluate("XGBoost", xgb_search.best_estimator_, X_test, Y_test, targets, y_scaler)
best_params["XGBoost"] = xgb_search.best_params_

# ---------------------------------------------------------------
# 5) MLP
# ---------------------------------------------------------------
print("Training MLP (with hyperparameter search) ...")

mlp = Pipeline([
    ('scale', x_scaler),
    ('reg', MultiOutputRegressor(
        MLPRegressor(
            random_state=42,
            max_iter=4000,
            early_stopping=True
        )
    ))
])

mlp_param_dist = {
    "reg__estimator__hidden_layer_sizes": [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (100,),
        (200,)
    ],
    "reg__estimator__alpha": loguniform(1e-6, 1e-2),      # L2 regularization
    "reg__estimator__learning_rate_init": loguniform(1e-4, 1e-2),
    "reg__estimator__activation": ["relu", "tanh"],
}

mlp_search = RandomizedSearchCV(
    mlp,
    mlp_param_dist,
    n_iter=n_iter,
    cv=cv_folds,
    n_jobs=-1,
    random_state=42,
    scoring='r2',
    refit=True
)

mlp_search.fit(X_train, Y_train_scaled)
results += evaluate("MLP", mlp_search.best_estimator_, X_test, Y_test, targets, y_scaler)
best_params["MLP"] = mlp_search.best_params_

# ===============================================================
#                      SAVE NUMERICAL RESULTS
# ===============================================================
print("Saving metrics and best parameters...")

results_df = pd.DataFrame(results)
results_df.to_excel(output_dir / "model_comparison_detailed.xlsx", index=False)

with open(output_dir / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)


avg_df = (
    results_df[results_df["Output"] == "AVG"]
    [["Model", "R2"]]
    .set_index("Model")
    .sort_values("R2", ascending=False)
)

plt.figure(figsize=(8, 4))
avg_df.plot(kind="bar", legend=False)
plt.title("Model comparison (average R²)")
plt.ylabel("Average R²")
plt.tight_layout()
plt.savefig(output_dir / "avg_R2_per_model.png", dpi=200)
plt.close()

best_model_name = avg_df.index[0]
print("\nBest model by average R²:", best_model_name)

if best_model_name != "XGBoost":
    print("Warning: best model is not XGBoost, "
          "but scatter plots and SHAP will still be generated for XGBoost as requested.")

print("\nAnalysis completed! Results folder:", output_dir.resolve())
print("Saved files:")
print("- model_comparison_detailed.xlsx")
print("- best_params.json")
print("- avg_R2_per_model.png")

# ===============================================================
#       XGBoost: OBSERVED vs PREDICTED (SCATTER PLOTS)
# ===============================================================
print("\nCreating 'observed vs predicted' scatter plots for XGBoost ...")

best_xgb = xgb_search.best_estimator_
xgb_models = best_xgb.named_steps['reg'].estimators_   # list of 4 single-output models

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, t in enumerate(targets):
    model = xgb_models[i]


    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)


    y_train_pred = inverse_transform_single_target(
        y_train_pred_scaled, y_scaler, i, n_targets
    )
    y_test_pred = inverse_transform_single_target(
        y_test_pred_scaled, y_scaler, i, n_targets
    )

    y_train_true = Y_train[t].values
    y_test_true = Y_test[t].values


    r2 = r2_score(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)

    ax = axes[i]
    ax.scatter(y_train_true, y_train_pred, alpha=0.7, label="Train")
    ax.scatter(y_test_true, y_test_pred, alpha=0.8, label="Test")

    lo = min(Y[t].min(), y_train_pred.min(), y_test_pred.min())
    hi = max(Y[t].max(), y_train_pred.max(), y_test_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--")

    ax.set_xlabel(f"Real {t}")
    ax.set_ylabel(f"Predicted {t}")

    ax.set_title(
        f"{t} \nR²={r2:.3f} | RMSE={rmse:.2f} | MAE={mae:.2f}",
        fontsize=10
    )
    ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "scatters_all.png", dpi=300)
plt.close()
print("Saved: scatters_all.png")

# ===============================================================
#                           SHAP
# ===============================================================
print("\nComputing SHAP values for XGBoost models ...")

feature_names = X_train.columns.tolist()
individual_paths = []

for i, t in enumerate(targets):
    model = xgb_models[i]


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
        show=False,
        plot_size=(6, 4)
    )
    plt.title(f"{t}")
    plt.tight_layout()

    out_path = output_dir / f"shap_{t}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    individual_paths.append(out_path)

print("Individual SHAP plots saved:", ", ".join(p.name for p in individual_paths))

# ===============================================================
#                   2×2 SHAP MOSAIC IMAGE
# ===============================================================
print("Building 2×2 SHAP mosaic ...")

images = [Image.open(p).convert("RGB") for p in individual_paths]
w, h = images[0].size

target_width = 1400
scale = target_width / w
size = (target_width, int(h * scale))

images = [im.resize(size, Image.LANCZOS) for im in images]

cols, rows = 2, 2
W, H = size[0] * cols, size[1] * rows
mosaic = Image.new("RGB", (W, H), "white")

positions = [
    (0, 0),
    (size[0], 0),
    (0, size[1]),
    (size[0], size[1])
]

for im, (x, y) in zip(images, positions):
    mosaic.paste(im, (x, y))

mosaic_path = output_dir / "shap_all_summary.png"
mosaic.save(mosaic_path, format="PNG")

print("Saved SHAP mosaic as:", mosaic_path.name)
print("\nDone.")
