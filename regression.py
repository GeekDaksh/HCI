import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD PROCESSED DATA
# =========================================================

processed_path = "processed"

X_all = []
y_all = []
groups = []

for file in sorted(os.listdir(processed_path)):
    if file.endswith(".npz"):
        subject_name = file.replace("_features.npz", "")
        data = np.load(os.path.join(processed_path, file))

        X = data["X"]
        y = data["workload"]

        X_all.append(X)
        y_all.append(y)
        groups.extend([subject_name] * len(y))

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)
groups = np.array(groups)

print("Total samples:", len(y_all))
print("Feature dimension:", X_all.shape[1])
print("Total subjects:", len(np.unique(groups)))

# =========================================================
# EXPONENTIAL SMOOTHING (FOR REAL-TIME SIMULATION)
# =========================================================

def exponential_smoothing(predictions, alpha=0.3):
    smoothed = [predictions[0]]
    for t in range(1, len(predictions)):
        smoothed.append(alpha * predictions[t] + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

# =========================================================
# SUBJECT-WISE CROSS VALIDATION
# =========================================================

gkf = GroupKFold(n_splits=5)

r2_scores = []
mae_scores = []
rmse_scores = []

feature_importances = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups)):

    print(f"\n===== Fold {fold+1} =====")

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Standardization (train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)

    # Apply smoothing for evaluation realism
    y_pred_smooth = exponential_smoothing(y_pred, alpha=0.3)

    # Metrics
    r2 = r2_score(y_test, y_pred_smooth)
    mae = mean_absolute_error(y_test, y_pred_smooth)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_smooth))

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

    feature_importances.append(model.feature_importances_)

    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n===== FINAL RESULTS =====")
print(f"Mean R2   : {np.mean(r2_scores):.4f}")
print(f"Mean MAE  : {np.mean(mae_scores):.4f}")
print(f"Mean RMSE : {np.mean(rmse_scores):.4f}")

print("\nStandard Deviation:")
print(f"R2   : {np.std(r2_scores):.4f}")
print(f"MAE  : {np.std(mae_scores):.4f}")
print(f"RMSE : {np.std(rmse_scores):.4f}")

# =========================================================
# FEATURE IMPORTANCE ANALYSIS
# =========================================================

avg_importance = np.mean(feature_importances, axis=0)

print("\n===== FEATURE IMPORTANCE SUMMARY =====")

# Band grouping (based on feature order)
n_channels = 14

delta_idx = slice(0, n_channels)
theta_idx = slice(n_channels, 2*n_channels)
alpha_idx = slice(2*n_channels, 3*n_channels)
beta_idx  = slice(3*n_channels, 4*n_channels)

theta_alpha_idx = slice(4*n_channels, 5*n_channels)
beta_alpha_idx  = slice(5*n_channels, 6*n_channels)
entropy_idx     = slice(6*n_channels, 7*n_channels)

print("Delta mean importance :", np.mean(avg_importance[delta_idx]))
print("Theta mean importance :", np.mean(avg_importance[theta_idx]))
print("Alpha mean importance :", np.mean(avg_importance[alpha_idx]))
print("Beta mean importance  :", np.mean(avg_importance[beta_idx]))
print("Theta/Alpha importance:", np.mean(avg_importance[theta_alpha_idx]))
print("Beta/Alpha importance :", np.mean(avg_importance[beta_alpha_idx]))
print("Entropy importance    :", np.mean(avg_importance[entropy_idx]))

# Top 15 features
top_indices = np.argsort(avg_importance)[::-1][:15]

print("\nTop 15 Important Features:")
for idx in top_indices:
    print(f"Feature {idx}: {avg_importance[idx]:.6f}")
