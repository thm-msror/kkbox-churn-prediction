"""
scoring/prepare_data.py
KKBox Churn Prediction - Data Preparation Script

This script is Stage 1 of the deployment pipeline.
It takes raw KKBox data files (same format as training data) and produces
a clean, feature-engineered CSV ready for score.py.

Usage:
    python scoring/prepare_data.py \\
        --members      data/new_members.csv \\
        --transactions data/new_transactions.csv \\
        --user_logs    data/new_user_logs.csv \\
        --train        data/new_train.csv \\
        --output       data/new_users_prepared.csv

Then run:
    python scoring/score.py \\
        --input  data/new_users_prepared.csv \\
        --output data/new_users_scored.csv

Then refresh Power BI by connecting to new_users_scored.csv.

Why two separate scripts?
    Stage 1 (this script): handles aggregation, cleaning, feature engineering.
    Aggregation requires collecting ALL events for a user before computing averages.
    This cannot run inside a sklearn pipeline because sklearn transforms one row
    at a time, not groups of rows.

    Stage 2 (score.py): handles sklearn ColumnTransformer scaling and LightGBM
    prediction. This runs in seconds on already-prepared data.

    This two-stage architecture is standard in production ML systems.
    The data pipeline (Stage 1) and the ML inference pipeline (Stage 2) are
    always separate because they have different compute requirements.
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent
MODELS_DIR = REPO_ROOT / 'models'

# The 9 columns used for clustering (must match what notebook 2 used)
CLUSTER_COLS = [
    'avg_total_secs', 'avg_num_100', 'listen_completion_ratio',
    'total_days_active', 'auto_renew_rate', 'cancel_rate',
    'discount_rate', 'avg_plan_days', 'total_transactions'
]

# IQR capping columns (must match notebook 1)
CAP_COLS = [
    'avg_list_price', 'avg_amount_paid', 'avg_num_25',
    'avg_num_50', 'avg_num_75', 'avg_num_985', 'avg_num_100',
    'avg_num_unq', 'avg_total_secs', 'total_days_active'
]


def aggregate_user_logs(user_logs_path: str) -> pd.DataFrame:
    """
    Aggregate user_logs from one row per day to one row per user.
    This mirrors what notebook 1 does.
    """
    print("Loading user logs (this may take 1-2 minutes for large files)...")
    logs = pd.read_csv(user_logs_path)
    print(f"  Raw user_logs: {len(logs):,} rows")

    logs_agg = logs.groupby('msno').agg(
        avg_num_25         = ('num_25',     'mean'),
        avg_num_50         = ('num_50',     'mean'),
        avg_num_75         = ('num_75',     'mean'),
        avg_num_985        = ('num_985',    'mean'),
        avg_num_100        = ('num_100',    'mean'),
        avg_num_unq        = ('num_unq',    'mean'),
        avg_total_secs     = ('total_secs', 'mean'),
        total_days_active  = ('date',       'count'),
    ).reset_index()

    print(f"  After aggregation: {len(logs_agg):,} users")
    return logs_agg


def aggregate_transactions(transactions_path: str) -> pd.DataFrame:
    """
    Aggregate transactions from one row per payment to one row per user.
    This mirrors what notebook 1 does.
    """
    print("Loading and aggregating transactions...")
    trans = pd.read_csv(transactions_path)
    print(f"  Raw transactions: {len(trans):,} rows")

    trans_agg = trans.groupby('msno').agg(
        total_transactions  = ('payment_method_id',  'count'),
        total_cancels       = ('is_cancel',           'sum'),
        avg_plan_days       = ('payment_plan_days',   'mean'),
        avg_list_price      = ('plan_list_price',     'mean'),
        avg_amount_paid     = ('actual_amount_paid',  'mean'),
        auto_renew_rate     = ('is_auto_renew',       'mean'),
        last_payment_method = ('payment_method_id',   'last'),
    ).reset_index()

    print(f"  After aggregation: {len(trans_agg):,} users")
    return trans_agg


def merge_all(members_path, transactions_path, user_logs_path, train_path):
    """
    Load all 4 raw files, aggregate, and merge on msno.
    Output: one row per user, 22 columns.
    """
    print("\nStep 1: Loading and aggregating raw files")
    members = pd.read_csv(members_path)
    train   = pd.read_csv(train_path)
    print(f"  members: {len(members):,} rows")
    print(f"  train:   {len(train):,} rows")

    trans_agg = aggregate_transactions(transactions_path)
    logs_agg  = aggregate_user_logs(user_logs_path)

    print("\nMerging all tables on msno...")
    df = train.merge(members,    on='msno', how='left')
    df = df.merge(trans_agg,     on='msno', how='left')
    df = df.merge(logs_agg,      on='msno', how='left')
    print(f"  Merged shape: {df.shape}")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning steps as notebook 1.
    - Drop age (bd) column
    - Fill gender nulls with 'unknown'
    - Median imputation for numeric nulls
    - IQR capping on behavioral columns (skip zero-variance)
    """
    print("\nStep 2: Cleaning data")

    # Drop age
    for col in ['bd', 'age']:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped: {col}")

    # Fill gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('unknown')
        print("  Gender: nulls filled with 'unknown'")

    # Median imputation for numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    nulls_filled = 0
    for col in num_cols:
        n = df[col].isnull().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].median())
            nulls_filled += n
    print(f"  Median imputation: {nulls_filled:,} nulls filled")

    # IQR capping (skip zero-variance columns - same as notebook 1)
    skipped = []
    capped  = []
    for col in CAP_COLS:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            skipped.append(col)
            continue
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
        capped.append(col)
    print(f"  IQR capping applied to: {capped}")
    print(f"  Skipped (IQR=0):        {skipped}")

    remaining_nulls = df.isnull().sum().sum()
    print(f"  Remaining nulls: {remaining_nulls}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as notebook 2.
    Creates: listen_completion_ratio, discount_rate, cancel_rate, gender_encoded
    """
    print("\nStep 3: Engineering features")

    # listen_completion_ratio
    total_songs = (df['avg_num_25'] + df['avg_num_50'] + df['avg_num_75'] +
                   df['avg_num_985'] + df['avg_num_100'])
    df['listen_completion_ratio'] = df['avg_num_100'] / (total_songs + 1e-6)
    df['listen_completion_ratio'] = df['listen_completion_ratio'].clip(0, 1)

    # discount_rate
    df['discount_rate'] = ((df['avg_list_price'] - df['avg_amount_paid']) /
                           (df['avg_list_price'] + 1e-6))
    df['discount_rate'] = df['discount_rate'].clip(0, 1)

    # cancel_rate
    df['cancel_rate'] = df['total_cancels'] / (df['total_transactions'] + 1e-6)
    df['cancel_rate'] = df['cancel_rate'].clip(0, 1)

    # gender encoding (must match notebook 2: female=0, male=1, unknown=2)
    gender_map = {'female': 0, 'male': 1, 'unknown': 2}
    df['gender_encoded'] = df['gender'].map(gender_map).fillna(2).astype(int)

    print(f"  Created: listen_completion_ratio, discount_rate, cancel_rate, gender_encoded")
    return df


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign cluster labels using the KMeans model and scaler saved from notebook 2.
    New users get the same cluster labels as training users.

    Requires (saved by notebook 2):
        models/kmeans_model.pkl    - fitted KMeans(n_clusters=6)
        models/cluster_scaler.pkl  - fitted StandardScaler
        models/cluster_names.json  - {cluster_number: cluster_name}
    """
    print("\nStep 4: Assigning clusters")

    kmeans_path = MODELS_DIR / 'kmeans_model.pkl'
    scaler_path = MODELS_DIR / 'cluster_scaler.pkl'
    names_path  = MODELS_DIR / 'cluster_names.json'

    if not all(p.exists() for p in [kmeans_path, scaler_path, names_path]):
        print("  Warning: clustering model files not found. Adding cluster=0 as placeholder.")
        print("  Run the save cell at the end of 02_features.ipynb to fix this.")
        df['cluster']      = 0
        df['cluster_name'] = 'unknown'
        return df

    with open(kmeans_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler_cluster = pickle.load(f)
    with open(names_path, 'r') as f:
        cluster_names = json.load(f)

    # Scale the same 9 columns used during training
    X_cluster = df[CLUSTER_COLS].copy()
    X_scaled  = scaler_cluster.transform(X_cluster)

    df['cluster']      = kmeans.predict(X_scaled)
    df['cluster_name'] = df['cluster'].astype(str).map(cluster_names).fillna('unknown')

    print(f"  Cluster distribution:")
    for name, count in df['cluster_name'].value_counts().items():
        print(f"    {name}: {count:,}")

    return df


def prepare(members_path, transactions_path, user_logs_path, train_path, output_path):
    """
    Full preparation pipeline: raw files -> model-ready CSV.
    """
    print("="*60)
    print("KKBox Data Preparation Pipeline")
    print("="*60)

    df = merge_all(members_path, transactions_path, user_logs_path, train_path)
    df = clean(df)
    df = engineer_features(df)
    df = assign_clusters(df)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Preparation complete")
    print(f"  Output: {output_path}")
    print(f"  Shape:  {df.shape}")
    print(f"\nNext step:")
    print(f"  python scoring/score.py --input {output_path} --output scored_output.csv")
    print("="*60)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='KKBox Churn Prediction - Prepare raw data for scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (using the training data as a test):
  python scoring/prepare_data.py \\
      --members      data/members_v3.csv \\
      --transactions data/transactions_v2.csv \\
      --user_logs    data/user_logs_v2.csv \\
      --train        data/train_v2.csv \\
      --output       data/merged_datasets/new_batch_prepared.csv

Then score it:
  python scoring/score.py \\
      --input  data/merged_datasets/new_batch_prepared.csv \\
      --output data/merged_datasets/new_batch_scored.csv
        """
    )
    parser.add_argument('--members',      required=True, help='Path to members CSV')
    parser.add_argument('--transactions', required=True, help='Path to transactions CSV')
    parser.add_argument('--user_logs',    required=True, help='Path to user_logs CSV')
    parser.add_argument('--train',        required=True, help='Path to train CSV (is_churn labels)')
    parser.add_argument('--output',       required=True, help='Path to save prepared CSV')
    args = parser.parse_args()

    prepare(args.members, args.transactions, args.user_logs, args.train, args.output)


if __name__ == '__main__':
    main()