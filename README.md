# KKBox churn prediction

End-to-end machine learning pipeline to predict subscriber churn for KKBox, Asia's largest music streaming platform. Built for DSAI 4103 Business Analytics at the University of Doha for Science and Technology.

- **Student:** Tehreem Masroor | 60302531
- **Course:** DSAI 4103 — Business Analytics
- **University:** University of Doha for Science and Technology
- **Submission:** April 4, 2026

PowerBI Dashboard: https://app.powerbi.com/links/gVHVWOaBC0?ctid=b30f4b44-46c6-4070-9997-f87b38d4771c&pbi_source=linkShare

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
| Holdout F1 (500 never-seen users) | 0.9176 |
| Holdout AUC | 0.9971 |

---

## Dataset

**Source:** https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

Join the competition on Kaggle (free) to access the files. Download the 4 files below and place them in the `data/` folder. Data files are not included in this repo because the raw files total approximately 1.8 GB and exceed GitHub's 100 MB limit.

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
├── notebooks/
│   ├── 01_EDA.ipynb                     <- data loading, aggregation, merging, cleaning, 7 eda charts
│   ├── 02_features.ipynb                <- feature engineering, k=6 clustering, cluster profiles
│   └── 03_modeling.ipynb                <- automl (60 combinations), threshold tuning, shap, bias, mlflow
│
├── models/
│   ├── churn_model.pkl                  <- trained pipeline: tomek links + lightgbm
│   ├── feature_cols.json                <- list of 23 feature column names used in training
│   ├── threshold.json                   <- optimal decision threshold (0.8526)
│   ├── kmeans_model.pkl                 <- fitted kmeans(k=6) for assigning clusters to new users
│   ├── cluster_scaler.pkl               <- fitted standardscaler used for clustering
│   └── cluster_names.json               <- cluster number to cluster name mapping
│
├── scoring/
│   ├── prepare_data.py                  <- stage 1: aggregate raw csvs, clean, engineer features, assign clusters
│   └── score.py                         <- stage 2: load pkl, score prepared data, output predictions
│
├── dashboard/
│   └── kkbox_dashboard.pbix             <- power bi dashboard connected to model_ready_with_predictions.csv
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
├── data/                                <- not tracked by git (too large, see .gitignore)
│   ├── members_v3.csv
│   ├── transactions_v2.csv
│   ├── user_logs_v2.csv
│   ├── train_v2.csv
│   └── merged_datasets/
│       ├── merged_df.csv                <- output of 01_EDA.ipynb
│       ├── model_ready.csv              <- output of 02_features.ipynb
│       ├── model_ready_with_predictions.csv  <- output of 03_modeling.ipynb, connect to power bi
│       ├── holdout_new_users.csv        <- 500 users set aside for deployment testing
│       └── holdout_scored.csv           <- scored holdout output (deployment demo)
│
├── mlruns/                              <- not tracked by git (mlflow experiment logs)
├── venv/                                <- not tracked by git (virtual environment)
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

# 5. download data from kaggle and place in data/ folder
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

**Stage 1 — prepare raw data** (5-10 minutes):

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

# Expected: F1=0.9176  AUC=0.9971  predicted churners: 40/500
```

`score.py` can be run multiple times without issues. Each run overwrites the output file with fresh predictions.

**Refresh power bi:** replace `model_ready_with_predictions.csv` with the new scored file and click Home -> Refresh in Power BI Desktop.

---

## Why prepare_data.py was not used for the holdout set

The holdout set (`holdout_new_users.csv`) was created inside notebook 3 by splitting off 500 rows from `model_ready.csv`, which is already the fully processed and feature-engineered dataset. This means the holdout already has all the cleaning, feature engineering, and cluster assignments applied. It does not need to go through `prepare_data.py`.

`prepare_data.py` exists for the scenario where completely new raw data arrives in the same format as the original Kaggle files — one row per user-day in user_logs, one row per transaction in transactions. That script runs the same aggregation, cleaning, and feature engineering steps that notebooks 1 and 2 performed, then assigns clusters using the saved KMeans model, producing a file that `score.py` can score.

If this project continued after submission, the correct test for `prepare_data.py` would be to hold back a small group of users with their complete raw event history in the original unaggregated tables, then run both stages end to end. This was not done because the preprocessing was completed in the notebooks before the project was structured this way.

---

## Why Tomek Links

The dataset has a 9% churn rate. This class imbalance means most models see so many retained users during training that they can achieve high accuracy by mostly ignoring churners. Tomek Links is an undersampling technique that removes retained users that are the nearest neighbour of a churned user, cleaning the boundary between the two classes and making it easier for the model to learn the distinction.

Tomek Links was selected over oversampling techniques like SMOTE because the AutoML comparison of 60 combinations showed that LightGBM + Tomek Links achieved the highest validation F1 score. It only runs during `.fit()` — during prediction on new data, the step is automatically skipped by the ImbPipeline.

---

## Why threshold tuning

By default, a classifier predicts churn if the estimated probability is above 0.5. At threshold 0.5, the model had precision=0.47, meaning 53% of users flagged as churners were actually going to renew — over half of all targeted discounts would be wasted.

The precision-recall curve was used to test every possible threshold on the held-out test set with the real 9% churn ratio preserved. The threshold that maximised F1 was 0.8526, giving precision=0.84 and recall=0.78. At this threshold, 84% of targeted users are genuine churners. The recall decrease from 0.93 to 0.78 is an acceptable trade-off because sending a discount to someone who was going to renew anyway has a direct cost, while missing a churner means losing a subscriber entirely.

---

## SHAP and bias analysis — what they are and how to act on results

**SHAP** explains why the model made each prediction. The summary plot shows that `auto_renew_rate` is the single most influential feature — users who rarely auto-renew are far more likely to be predicted as churners. The waterfall plot for a specific user shows exactly which features drove that individual prediction.

Practically, a retention analyst would use SHAP output to inform campaign design. If SHAP shows that low auto_renew_rate is the dominant driver for a segment, the most effective intervention is an offer to enable auto-renew rather than a price discount. If low listening time is the dominant driver, the intervention might be re-engagement content.

**Bias analysis** checks whether the model performs fairly across demographic groups. F1 was computed separately for each gender (male: 0.82, female: 0.83, unknown: 0.78) and the top 5 cities. If any group had a significantly lower F1, that group would be systematically excluded from retention campaigns not because they are low-risk but because the model is less accurate for them. Results showed no significant bias — maximum difference across gender groups was 0.05, within the acceptable range. No corrective action was required, but the analysis is documented so that fairness can be re-evaluated when the model is retrained.

---

## Why two stages in deployment

Sklearn's `ColumnTransformer` operates row by row. It cannot aggregate 18 million daily events by user id to compute averages — that requires seeing all events for a user before any result can be produced. `prepare_data.py` handles aggregation (Stage 1). `score.py` handles sklearn inference (Stage 2). This two-stage architecture is standard in production ML systems.

---

## Experiment tracking

All 60 AutoML combinations and the final model are logged to MLflow.

```bash
mlflow ui
# open http://localhost:5000
```

---

## Power BI dashboard

https://app.powerbi.com/links/gVHVWOaBC0?ctid=b30f4b44-46c6-4070-9997-f87b38d4771c&pbi_source=linkShare

Open `dashboard/kkbox_dashboard.pbix` in Power BI Desktop. The dashboard is connected to `data/merged_datasets/model_ready_with_predictions.csv`.

The dashboard includes four KPI cards (total users, churn rate, high-risk count, average probability), a donut chart, cluster and city bar charts, an engagement scatter plot, and three slicers for filtering by segment, gender, and churn prediction.

---

## Tools

| Tool | Purpose |
|---|---|
| Pandas | Data loading, aggregation, cleaning, merging |
| Scikit-learn | Clustering, ColumnTransformer, preprocessing pipeline |
| Imbalanced-learn | Tomek Links undersampling, ImbPipeline |
| PyCaret | AutoML: 60 model + sampling combinations |
| LightGBM | Final churn prediction model |
| SHAP | Model explainability — feature importance and individual predictions |
| MLflow | Experiment tracking — all 61 runs logged |
| Power BI | Interactive dashboard for the retention marketing team |

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

- `cuml` warnings from PyCaret can be safely ignored. cuml is a GPU-only library not required for this project.
- `logs.log` is generated automatically by PyCaret and excluded from git via `.gitignore`.
- All random states are set to 42 throughout, making results fully reproducible.
- The `venv/`, `data/`, and `mlruns/` folders are excluded from git. See `.gitignore`.