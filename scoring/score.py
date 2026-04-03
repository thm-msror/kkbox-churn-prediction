"""
scoring/score.py
KKBox Churn Prediction - Deployment Scoring Script

Usage:
    python scoring/score.py --input new_users.csv --output scored_users.csv

What it does:
    1. Loads the trained model pipeline from models/churn_model.pkl
    2. Loads the optimal decision threshold from models/threshold.json
    3. Reads new user data from the input CSV
    4. Scores every user and outputs churn_probability (0-1) and churn_prediction (0 or 1)
    5. Saves the scored data to the output CSV

The pipeline handles everything internally during predict_proba():
    - The Tomek Links sampling step is skipped automatically (only active during .fit())
    - No manual preprocessing is needed on new data
    - Feature scaling is also handled inside the LightGBM model

Input CSV requirements:
    - Must contain the same 23 feature columns used during training
    - Column names must match exactly (see FEATURE_COLS below)
    - Missing columns will raise a clear error message

Output columns added:
    - churn_probability: float 0 to 1 (model confidence that user will churn)
    - churn_prediction: 0 or 1 (1 = predicted to churn at the tuned threshold)
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths (relative to the repo root, where this script should be run from)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
MODEL_PATH   = REPO_ROOT / 'models' / 'churn_model.pkl'
FEATURES_PATH = REPO_ROOT / 'models' / 'feature_cols.json'
THRESHOLD_PATH = REPO_ROOT / 'models' / 'threshold.json'

# ──────────────────────────────────────────────────────────────────────────────
# Feature columns (fallback if feature_cols.json is missing)
# These must match exactly what was used in training
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FEATURE_COLS = [
    'city', 'registered_via', 'registration_init_time',
    'total_transactions', 'total_cancels', 'avg_plan_days',
    'avg_list_price', 'avg_amount_paid', 'auto_renew_rate',
    'last_payment_method', 'avg_num_25', 'avg_num_50', 'avg_num_75',
    'avg_num_985', 'avg_num_100', 'avg_num_unq', 'avg_total_secs',
    'total_days_active', 'listen_completion_ratio', 'discount_rate',
    'cancel_rate', 'gender_encoded', 'cluster'
]

DEFAULT_THRESHOLD = 0.8526


def load_model():
    """Load the trained pipeline from disk."""
    if not MODEL_PATH.exists():
        print(f"Error: model file not found at {MODEL_PATH}")
        print("Run 03_modeling.ipynb first to train and save the model.")
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Model type: {type(model.named_steps['model']).__name__}")
    return model


def load_feature_cols():
    """Load the list of feature columns from the saved JSON file."""
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, 'r') as f:
            cols = json.load(f)
        print(f"Feature list loaded: {len(cols)} features")
        return cols
    else:
        print(f"Warning: feature_cols.json not found, using default feature list")
        return DEFAULT_FEATURE_COLS


def load_threshold():
    """Load the optimal decision threshold from the saved JSON file."""
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, 'r') as f:
            data = json.load(f)
        threshold = data.get('best_threshold', DEFAULT_THRESHOLD)
        print(f"Threshold loaded: {threshold:.4f} (from threshold.json)")
        return threshold
    else:
        print(f"Warning: threshold.json not found, using default threshold {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD


def validate_columns(df, feature_cols):
    """Check that all required columns are present in the input data."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"\nError: the following required columns are missing from the input file:")
        for col in missing:
            print(f"  - {col}")
        print(f"\nInput file has {len(df.columns)} columns: {list(df.columns)}")
        sys.exit(1)


def score(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load new user data, score with the trained model, and save output.

    Parameters
    ----------
    input_path  : path to input CSV with new user data
    output_path : path to save the scored output CSV

    Returns
    -------
    DataFrame with churn_probability and churn_prediction columns added
    """
    # Load model, features, and threshold
    model        = load_model()
    feature_cols = load_feature_cols()
    threshold    = load_threshold()

    # Load input data
    if not os.path.exists(input_path):
        print(f"Error: input file not found at {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"\nInput loaded: {len(df):,} users from {input_path}")

    # Validate columns
    validate_columns(df, feature_cols)

    # Select features
    X = df[feature_cols]

    # Score
    # predict_proba returns [[prob_retained, prob_churned], ...]
    # We take column 1 (churn probability)
    # The ImbPipeline sampling step is skipped automatically during predict_proba
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    df['churn_prediction']  = (df['churn_probability'] >= threshold).astype(int)

    # Summary statistics
    n_churned   = df['churn_prediction'].sum()
    churn_pct   = n_churned / len(df) * 100
    avg_prob    = df['churn_probability'].mean()

    print(f"\nScoring complete:")
    print(f"  Total users scored:     {len(df):,}")
    print(f"  Predicted churners:     {n_churned:,} ({churn_pct:.1f}%)")
    print(f"  Average churn probability: {avg_prob:.4f}")
    print(f"  Decision threshold used:   {threshold:.4f}")

    # Save output
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved: {output_path}")
    print(f"Columns in output: {list(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='KKBox Churn Prediction - Score new users',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score the holdout set (deployment demo)
  python scoring/score.py \\
      --input  data/merged_datasets/holdout_new_users.csv \\
      --output data/merged_datasets/holdout_scored.csv

  # Score a new batch of users
  python scoring/score.py \\
      --input  data/new_users_april.csv \\
      --output data/new_users_april_scored.csv
        """
    )
    parser.add_argument('--input',  required=True,  help='Path to input CSV with new user data')
    parser.add_argument('--output', required=True,  help='Path to save the scored output CSV')
    args = parser.parse_args()

    score(args.input, args.output)


if __name__ == '__main__':
    main()