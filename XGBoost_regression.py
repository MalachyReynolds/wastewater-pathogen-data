import numpy as np
import pandas as pd
import joblib
import inspect
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "filtered_data.csv"
DATE_COL = "date"
TARGET_COL = "uk_influenza_like_ed__influenza-like_syndromic_emergencyDepartment_countsByDay"

# Forecast horizon in days
HORIZON = 14

# Only use predictors that could plausibly lead ED influenza counts
BASE_FEATURES = [
    "netherlands_rna",
    "scotland_rna",
    "swiss_sars_cov_2",
    "swiss_influenza_a",
    "swiss_influenza_b",
    "swiss_rsv",
    "uk_ari_calls__acute-respiratory-infection_syndromic_NHS111triagedcalls_countsByDay",
    "uk_ari_syndromic_calls__acute-respiratory-infection_syndromic_NHS111triagedcalls_countsByDay",
    "uk_lrti_gp_hours__lower-respiratory-tract-infection_syndromic_GPInHours_rateByDay",
    "uk_urti_gp_hours__upper-respiratory-tract-infection_syndromic_GPInHours_rateByDay",
    "Temp",
    "WeekOfYear",
    "SinWeek",
    "CosWeek",
]

LAGS = [1, 3, 7, 14]
ROLL_WINDOWS = [7, 14]

TEST_FRACTION = 0.2
VALID_FRACTION = 0.1
RANDOM_STATE = 42

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(INPUT_FILE)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COL}")

# Keep only columns that actually exist in the file
BASE_FEATURES = [c for c in BASE_FEATURES if c in df.columns]

if len(BASE_FEATURES) == 0:
    raise ValueError("None of the selected base features were found in the CSV.")

# ============================================================
# CLEAN UP
# ============================================================
# Drop columns that are completely empty or constant among selected predictors
clean_features = []
for c in BASE_FEATURES:
    s = df[c]
    if s.notna().sum() == 0:
        continue
    if s.nunique(dropna=True) <= 1:
        continue
    clean_features.append(c)

BASE_FEATURES = clean_features

if len(BASE_FEATURES) == 0:
    raise ValueError("No usable feature columns remained after cleaning.")

# ============================================================
# CREATE FORECAST TARGET
# Predict ED influenza counts HORIZON days ahead
# ============================================================
df["y_raw"] = df[TARGET_COL].shift(-HORIZON)

# Log transform target to stabilize variance
df["y"] = np.log1p(df["y_raw"])

# ============================================================
# FEATURE ENGINEERING
# Use only past information for predictors
# ============================================================
# Build new feature columns in a dict and concat once to avoid
# DataFrame fragmentation/performance warnings.
new_cols = {}
for col in BASE_FEATURES:
    # Current-day feature
    new_cols[f"{col}_t"] = df[col]

    # Lag features
    for lag in LAGS:
        new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling means on past values only
    for w in ROLL_WINDOWS:
        new_cols[f"{col}_roll{w}"] = df[col].shift(1).rolling(w).mean()

if new_cols:
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Drop rows where target is unavailable
model_df = df.dropna(subset=["y"]).copy()

feature_cols = [c for c in model_df.columns if c not in [DATE_COL, TARGET_COL, "y_raw", "y"]]

# Keep numeric features only
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(model_df[c])]

X = model_df[feature_cols]
y = model_df["y"]

# ============================================================
# TIME SPLIT
# Chronological split only, no shuffle
# ============================================================
n = len(model_df)
test_size = max(1, int(n * TEST_FRACTION))
valid_size = max(1, int(n * VALID_FRACTION))
train_size = n - test_size - valid_size

if train_size <= 0:
    raise ValueError("Not enough data for train/validation/test split.")

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_valid = X.iloc[train_size:train_size + valid_size]
y_valid = y.iloc[train_size:train_size + valid_size]

X_test = X.iloc[train_size + valid_size:]
y_test = y.iloc[train_size + valid_size:]

# ============================================================
# MODEL
# Early stopping helps reduce overfitting on this small dataset
# ============================================================
model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    gamma=0.0,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    tree_method="hist",
)

# Build fit kwargs only with parameters accepted by the current xgboost
fit_kwargs = {"X": X_train, "y": y_train}
fit_sig = inspect.signature(model.fit)
if "eval_set" in fit_sig.parameters:
    fit_kwargs["eval_set"] = [(X_valid, y_valid)]
if "verbose" in fit_sig.parameters:
    fit_kwargs["verbose"] = 100
if "early_stopping_rounds" in fit_sig.parameters:
    fit_kwargs["early_stopping_rounds"] = 100
elif "callbacks" in fit_sig.parameters:
    # use callback early stopping if available
    try:
        fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=100)]
    except Exception:
        # fallback: no early stopping
        pass

model.fit(**fit_kwargs)

# ============================================================
# EVALUATION ON ORIGINAL SCALE
# ============================================================
pred_log = model.predict(X_test)

# Convert back from log1p
pred = np.expm1(pred_log)
actual = np.expm1(y_test.values)

pred = np.clip(pred, 0, None)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)

print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2:   {r2:.4f}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================
results = model_df.iloc[train_size + valid_size:][[DATE_COL]].copy()
results["actual"] = actual
results["predicted"] = pred
results.to_csv("xgb_predictions_new.csv", index=False)

print("Saved xgb_predictions_new.csv")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv("xgb_feature_importance.csv", index=False)
print("Saved xgb_feature_importance.csv")

print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))

# ============================================================
# SAVE MODEL
# ============================================================
joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "horizon_days": HORIZON,
        "base_features": BASE_FEATURES,
        "lags": LAGS,
        "roll_windows": ROLL_WINDOWS,
    },
    "xgb_ed_influenza_model.joblib",
)

print("Saved xgb_ed_influenza_model.joblib")