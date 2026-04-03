# KKBox Churn Prediction

End-to-end machine learning pipeline to predict subscriber churn for KKBox, Asia's largest music streaming platform. Built for DSAI 4103 Business Analytics at the University of Doha for Science and Technology.

- **Student:** Tehreem Masroor | 60302531
- **Course:** DSAI 4103 — Business Analytics
- **University:** University of Doha for Science and Technology
- **Due:** April 4, 2026

---

## Business problem

KKBox operates a monthly subscription model across Taiwan, Hong Kong, Singapore, Malaysia, and Japan. Each month, some subscribers choose not to renew — this is called churn. Acquiring a new customer costs significantly more than retaining an existing one.

This project predicts which users will not renew within 30 days of their subscription expiring, so the retention team can send targeted discount offers before they leave.

---

## Results

| Metric | Value |
|---|---|
| Best model | LightGBM + Tomek Links |
| AutoML combinations tested | 60 (10 models x 6 sampling techniques) |
| AUC-ROC | 0.9768 |
| F1 score (tuned threshold 0.8526) | 0.8062 |
| F1 score (default threshold 0.5) | 0.6257 |
| Precision | 0.8382 |
| Recall | 0.7767 |
| Holdout f1 (500 never-seen users) | 0.9176 |
| Holdout AUC | 0.9971 |

---

## Dataset

**Source:** https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

Join the competition on Kaggle (free) to access the files. Download the 4 files below and place them in the `data/` folder. Data files are not included in this repo — the raw files total ~1.8 GB and exceed GitHub's 100 MB limit.

| File | Rows | Description |
|---|---|---|
| `members_v3.csv` | 6.7M | User demographics — city, gender, registration info |
| `transactions_v2.csv` | 1.4M | Subscription and payment history |
| `user_logs_v2.csv` | 18.4M | Daily listening behavior — songs played, seconds listened |
| `train_v2.csv` | 970K | Churn labels — is_churn = 1 means the user did not renew |

**How to download:**

1. Go to https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data
2. Join the competition (free) to unlock the download
3. Download and extract the `.7z` files for the 4 files listed above
4. Place the `.csv` files directly inside the `data/` folder (not in subfolders)

---

## Repo structure

```
kkbox-churn-prediction/
│
├── notebooks/
│   ├── 01_EDA.ipynb                     ← data loading, aggregation, merging, cleaning, 7 eda charts
│   ├── 02_features.ipynb                ← feature engineering, k=6 clustering, cluster profiles
│   └── 03_modeling.ipynb                ← automl (60 combinations), threshold tuning, shap, bias, mlflow
│
├── models/
│   ├── churn_model.pkl                  ← trained imb pipeline: tomek links + lightgbm
│   ├── feature_cols.json                ← list of 23 feature column names used in training
│   ├── threshold.json                   ← optimal decision threshold (0.8526)
│   ├── kmeans_model.pkl                 ← fitted kmeans(k=6) for assigning clusters to new users
│   ├── cluster_scaler.pkl               ← fitted standardscaler used for clustering
│   └── cluster_names.json               ← cluster number to cluster name mapping
│
├── scoring/
│   ├── prepare_data.py                  ← stage 1: aggregate raw csvs, clean, engineer features, assign clusters
│   └── score.py                         ← stage 2: load pkl, score prepared data, output predictions
│
├── dashboard/
│   └── kkbox_dashboard.pbix             ← power bi dashboard connected to model_ready_with_predictions.csv
│
├── Visualizations/
│   ├── plot_age_boxplot_before.png
│   ├── plot_age_boxplot_after.png
│   ├── plot_gender_before.png
│   ├── plot_gender_after.png
│   ├── plot_churn_overview.png
│   ├── plot_churn_by_city.png
│   ├── plot_churn_by_gender.png
│   ├── plot_listening_vs_churn.png
│   ├── plot_cancels_vs_churn.png
│   ├── plot_autorenew_vs_churn.png
│   ├── plot_correlation_heatmap.png
│   ├── plot_correlation_heatmap_v2.png
│   ├── plot_kmeans_elbow.png
│   ├── plot_cluster_churn.png
│   ├── plot_cluster_sizes.png
│   ├── model_comparison_final.png
│   ├── plot_confusion_matrix.png
│   ├── plot_confusion_matrix_comparison.png
│   ├── plot_threshold_tuning.png
│   ├── plot_auc_roc_curve.png
│   ├── plot_shap_summary.png
│   ├── plot_shap_waterfall.png
│   ├── plot_bias_gender.png
│   └── plot_bias_city.png
│
├── data/                                ← not tracked by git (too large, see .gitignore)
│   ├── members_v3.csv                   ← download from kaggle
│   ├── transactions_v2.csv              ← download from kaggle
│   ├── user_logs_v2.csv                 ← download from kaggle
│   ├── train_v2.csv                     ← download from kaggle
│   └── merged_datasets/
│       ├── merged_df.csv                ← output of 01_EDA.ipynb
│       ├── model_ready.csv              ← output of 02_features.ipynb
│       ├── model_ready_with_predictions.csv  ← output of 03_modeling.ipynb, connect this to power bi
│       ├── holdout_new_users.csv        ← 500 users set aside for deployment testing
│       └── holdout_scored.csv           ← scored holdout output
│
├── mlruns/                              ← not tracked by git (mlflow experiment logs)
├── venv/                                ← not tracked by git (virtual environment)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
# 1. clone the repo
git clone https://github.com/thm-msror/kkbox-churn-prediction
cd kkbox-churn-prediction

# 2. create virtual environment
python -m venv venv

# windows
venv\Scripts\activate

# mac / linux
source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. register the jupyter kernel
python -m ipykernel install --user --name kkbox-env --display-name "KKBox Project"

# 5. download data from kaggle (see dataset section above) and place in data/ folder
```

Open Jupyter, select kernel: KKBox Project, then run the notebooks in order.

---

## Run notebooks in order

```bash
# notebook 1: loads raw files, aggregates, merges, cleans, saves merged_df.csv
notebooks/01_EDA.ipynb

# notebook 2: feature engineering, k=6 clustering, saves model_ready.csv
notebooks/02_features.ipynb

# notebook 3: automl 60 combinations, threshold tuning, shap, bias, saves model + predictions
notebooks/03_modeling.ipynb
```

Each notebook saves its output to `data/merged_datasets/`. Run them top to bottom in order.

---

## Score new data (deployment)

The deployment pipeline has two stages. Stage 1 aggregates raw event data to one row per user. Stage 2 runs the sklearn pipeline to output churn predictions.

**Stage 1 — prepare raw data** (5-10 minutes, handles aggregation and cleaning):

```bash
python scoring/prepare_data.py \
    --members      data/members_v3.csv \
    --transactions data/transactions_v2.csv \
    --user_logs    data/user_logs_v2.csv \
    --train        data/train_v2.csv \
    --output       data/merged_datasets/new_batch_prepared.csv
```

**Stage 2 — score prepared data** (under 30 seconds):

```bash
python scoring/score.py \
    --input  data/merged_datasets/new_batch_prepared.csv \
    --output data/merged_datasets/new_batch_scored.csv
```

**Test on the holdout set** (500 users never seen during training):

```bash
python scoring/score.py \
    --input  data/merged_datasets/holdout_new_users.csv \
    --output data/merged_datasets/holdout_scored.csv

# Expected: f1=0.9176  AUC=0.9971  predicted churners: 40/500
```

**Refresh power bi:** replace `model_ready_with_predictions.csv` with the new scored file, then click Home → Refresh in Power BI.

---

## Why two stages

Sklearn's `ColumnTransformer` operates row by row — it scales, encodes, and imputes one user at a time. It cannot aggregate 18 million daily events by user id to compute averages. That requires seeing all events for a user before any result can be produced.

`prepare_data.py` handles the aggregation step. `score.py` handles the sklearn inference step. This two-stage architecture is standard in production ml systems — the data engineering pipeline and the ml inference pipeline are always separate. We learned about this standard separation after completing the main notebooks. If we had known this earlier, we would have held out some raw data rows (e.g., 20 rows from members, transactions, user_logs) with the same `msno` to fully test `prepare_data.py` end-to-end. Currently, the holdout test uses already-aggregated data.

---

## Technical Explanations

**Threshold Tuning:** 
By default, models use a 0.5 threshold to classify churn vs. retain. Because our data is imbalanced (9% churn) and the business cost of a false negative (losing a customer) is worse than a false positive (wasted discount), we optimized the threshold. Tuning it to 0.8526 significantly improved our F1 score from 0.63 to 0.81, balancing precision and recall to be highly effective for a marketing budget.

**Tomek Links (Undersampling):**
With only ~9% churners, a model could simply predict "retain" every time and be 91% "accurate." Tomek links combats this by finding pairs of nearest neighbors from opposite classes and removing the instance from the majority class. This cleans the decision boundary without generating "fake" synthetic data. Inside our `ImbPipeline`, it is only applied during training and is automatically skipped during predictions.

**SHAP & Bias Analysis:**
Explainability proves the model is learning real patterns. The SHAP summary plot confirms the most important features make business sense. Most importantly, **SHAP waterfall plots are actionable**: a customer success agent can pull up a churning user, see exactly *why* they are high risk (e.g., auto-renew was disabled), and tailor the intervention message. Bias analysis was conducted to ensure the model behaves fairly; we found no demographic bias, achieving an F1 score of ~0.82 for both male and female users.

---

## Experiment tracking

All 60 automl combinations and the final model are logged to mlflow.

```bash
mlflow ui
# open http://localhost:5000
```

---

## Power BI Dashboard

Open `dashboard/kkbox_dashboard.pbix` in Power BI Desktop. The dashboard is connected to `data/merged_datasets/model_ready_with_predictions.csv`.

To update the dashboard with new predictions: run `score.py` on new data, replace the csv, and click Home → Refresh.

---

## Tools

| Tool | Purpose |
|---|---|
| Pandas | Data loading, aggregation, cleaning, merging |
| Scikit-learn | Clustering, ColumnTransformer, preprocessing pipeline |
| Imbalanced-learn | Tomek links undersampling, ImbPipeline |
| PyCaret | AutoML: 60 model + sampling combinations |
| LightGBM | Final churn prediction model |
| Shap | Model explainability — feature importance and individual predictions |
| Mlflow | Experiment tracking — all 61 runs logged |
| Power bi | Interactive dashboard for the retention marketing team |

---

## Customer segments (k=6 clusters)

| Cluster | Churn rate | Users | Key characteristic | Recommended action |
|---|---|---|---|---|
| Regular subscribers | 5.2% | 648,464 | Always auto renews | No action needed |
| Highly engaged loyalists | 4.1% | 207,231 | Highest listening (9,475 secs/day) | Loyalty rewards |
| At-risk manual renewers | 18.6% | 69,310 | Almost never auto renews | Offer auto-enrol incentive |
| High-risk auto renewers | 49.9% | 2,107 | Auto renews but still churns | Investigate product fit |
| Churning users | 59.3% | 31,924 | High cancel rate (0.595) | Discount 7 days before expiry |
| Lapsed annual subscribers | 99.6% | 11,924 | Annual plan (avg 287 days) | Renewal outreach 60 days before |

---

## Notes

- `cuml` warnings from PyCaret can be safely ignored. Cuml is a gpu-only library that is not required for this project and cannot be installed on Windows via pip.
- `logs.log` is generated automatically by PyCaret. It is excluded from git via `.gitignore`.
- All random states are set to 42 throughout the project, making results fully reproducible.
- The `venv/`, `data/`, and `mlruns/` folders are excluded from git. See `.gitignore`.