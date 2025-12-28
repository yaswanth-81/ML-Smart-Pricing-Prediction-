import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
submission_path = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\outputs\submission.csv"
actual_csv_path = r"Y:\ML_CHALLENGE_2025\ml-pricing-challenge\dataset\sample_test_out.csv"

# ============================================================
# LOAD CSV FILES
# ============================================================
if not os.path.exists(submission_path):
    raise FileNotFoundError(f"❌ Submission CSV not found at {submission_path}")
if not os.path.exists(actual_csv_path):
    raise FileNotFoundError(f"❌ Actual prices CSV not found at {actual_csv_path}")

submission_df = pd.read_csv(submission_path)
actual_df = pd.read_csv(actual_csv_path)

# ============================================================
# RENAME COLUMN TO MATCH
# ============================================================
actual_df = actual_df.rename(columns={"sample_id": "id"})

# ============================================================
# ALIGN IDS
# ============================================================
# Ensure the actual prices match the order of submission IDs
actual_df = actual_df.set_index('id').loc[submission_df['id']].reset_index()
y_true = actual_df['price'].values
y_pred = submission_df['price'].values

# ============================================================
# CALCULATE SMAPE
# ============================================================
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1, denominator)  # avoid division by zero
    return np.mean(numerator / denominator) * 100

score = smape(y_true, y_pred)
print(f"📊 SMAPE Score: {score:.2f}%")
