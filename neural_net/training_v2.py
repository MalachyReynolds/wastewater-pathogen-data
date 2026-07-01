import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
CSV_PATH = "filtered_data.csv"
DATE_COL = "date"
TARGET_COL = "uk_influenza_like_ed__influenza-like_syndromic_emergencyDepartment_countsByDay"

LOOKBACK_DAYS = 14     # previous 2 weeks as input
HORIZON_DAYS = 14      # predict 2 weeks ahead
BATCH_SIZE = 32
EPOCHS = 10000
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COL}")

# --------------------------------------------------
# CHOOSE INPUT FEATURES
# Drop target-related ED columns to avoid leakage
# Keep raw signals, seasonality, temperature, GP, NHS111, wastewater
# --------------------------------------------------
exclude_keywords = [
    "emergencyDepartment",   # do not use ED-derived predictors when forecasting ED
    "baseline",              # often derived from historical smoothing
    "averageRolling7Day",    # smoothed versions, usually redundant
]

feature_cols = []
for c in df.columns:
    if c in [DATE_COL, TARGET_COL]:
        continue
    if not pd.api.types.is_numeric_dtype(df[c]):
        continue
    if any(k in c for k in exclude_keywords):
        continue
    feature_cols.append(c)

if len(feature_cols) == 0:
    raise ValueError("No usable feature columns were found.")

print("Using features:")
for c in feature_cols:
    print(" -", c)

# --------------------------------------------------
# SIMPLE MISSING VALUE HANDLING
# Forward-fill only, so we do not use future information
# --------------------------------------------------
df[feature_cols] = df[feature_cols].ffill()

# Still drop any rows that remain missing in features or target
df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

# --------------------------------------------------
# SCALE FEATURES
# Fit scaler on training period only to avoid leakage
# --------------------------------------------------
raw_feature_matrix = df[feature_cols].values.astype(np.float32)
target_values = df[TARGET_COL].values.astype(np.float32)

n_rows = len(df)
train_row_end = int(n_rows * 0.70)
valid_row_end = int(n_rows * 0.85)

scaler = StandardScaler()
scaler.fit(raw_feature_matrix[:train_row_end])

scaled_features = scaler.transform(raw_feature_matrix).astype(np.float32)

# --------------------------------------------------
# BUILD SLIDING WINDOW DATASET
# Each sample:
#   X = previous LOOKBACK_DAYS of all features
#   y = ED count HORIZON_DAYS ahead
# --------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, dates, lookback_days, horizon_days, start_idx, end_idx):
        self.X = X
        self.y = y
        self.dates = dates
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.samples = []

        # The label index is the future day we want to predict
        # We split by label index so training/validation/test stay chronological
        for end_idx_window in range(lookback_days - 1, len(X) - horizon_days):
            label_idx = end_idx_window + horizon_days
            if start_idx <= label_idx < end_idx:
                self.samples.append((end_idx_window, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        end_idx_window, label_idx = self.samples[idx]

        start_idx_window = end_idx_window - self.lookback_days + 1
        x_seq = self.X[start_idx_window:end_idx_window + 1]
        y_val = self.y[label_idx]

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(1),
        )

# --------------------------------------------------
# SPLITS
# Train uses early part of the timeline
# Validation follows train
# Test is the final block
# --------------------------------------------------
train_ds = TimeSeriesDataset(
    X=scaled_features,
    y=target_values,
    dates=df[DATE_COL].values,
    lookback_days=LOOKBACK_DAYS,
    horizon_days=HORIZON_DAYS,
    start_idx=0,
    end_idx=train_row_end,
)

val_ds = TimeSeriesDataset(
    X=scaled_features,
    y=target_values,
    dates=df[DATE_COL].values,
    lookback_days=LOOKBACK_DAYS,
    horizon_days=HORIZON_DAYS,
    start_idx=train_row_end,
    end_idx=valid_row_end,
)

test_ds = TimeSeriesDataset(
    X=scaled_features,
    y=target_values,
    dates=df[DATE_COL].values,
    lookback_days=LOOKBACK_DAYS,
    horizon_days=HORIZON_DAYS,
    start_idx=valid_row_end,
    end_idx=len(df),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")
print(f"Test samples:  {len(test_ds)}")

# --------------------------------------------------
# MODEL
# LSTM reads the time window and outputs one number
# --------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: [batch, time, features]
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)

model = LSTMRegressor(input_size=len(feature_cols)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
best_val_loss = float("inf")
best_state = None
patience = float("inf")  # No early stopping by default
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= max(1, len(train_loader.dataset))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= max(1, len(val_loader.dataset))

    if epoch % 1000 == 0 or epoch == EPOCHS - 1:

        print(f"Epoch {epoch+1:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
model.eval()
all_preds = []
all_actual = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().reshape(-1)
        all_preds.extend(preds.tolist())
        all_actual.extend(yb.numpy().reshape(-1).tolist())

all_preds = np.array(all_preds)
all_actual = np.array(all_actual)

mae = mean_absolute_error(all_actual, all_preds)
rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
r2 = r2_score(all_actual, all_preds)

print("\nTest performance")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------
test_dates = []
for _, label_idx in test_ds.samples:
    test_dates.append(df[DATE_COL].iloc[label_idx])

pred_df = pd.DataFrame({
    "date": test_dates,
    "actual": all_actual,
    "predicted": all_preds
})

pred_df.to_csv("lstm_ed_predictions.csv", index=False)
print("Saved lstm_ed_predictions.csv")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "feature_cols": feature_cols,
    "lookback_days": LOOKBACK_DAYS,
    "horizon_days": HORIZON_DAYS,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
}, "lstm_ed_model.pt")

print("Saved lstm_ed_model.pt")

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(pred_df["date"], pred_df["actual"], label="Actual")

plt.plot(pred_df["date"], pred_df["predicted"], label="Predicted")

plt.xlabel("Date")

plt.ylabel("ED Influenza Count")

plt.title("Actual vs Predicted Emergency Department Influenza Counts")

plt.legend()

plt.tight_layout()

plt.savefig("lstm_ed_actual_vs_predicted.png", dpi=300)

plt.figure(figsize=(6,6))

plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.6)

m = max(pred_df["actual"].max(), pred_df["predicted"].max())

plt.plot([0,m],[0,m],'r--')

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Predicted vs Actual")

plt.show()