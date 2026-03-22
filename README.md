# KKBox Churn Prediction

- **Course:** DSAI 4103 — Business Analytics
- **University:** University of Doha for Science & Technology
- **Student:** Tehreem Masroor | 60302531
- **Due:** March 29, 2026

---

## Business Problem

KKBox is Asia's largest music streaming platform operating on a subscription model. 
Each month, thousands of subscribers choose not to renew which is a costly problem known as 
customer churn. This project builds an end-to-end machine learning pipeline to:

- Predict which subscribers are likely to churn before their subscription expires
- Segment customers into behavioral groups using clustering
- Identify the key drivers of churn using explainable AI (SHAP)
- Visualize insights through an interactive Power BI dashboard

Early churn detection allows KKBox to intervene with targeted retention offers, 
directly protecting revenue, making this a business problem worth solving for a business analytics course.

---

## Dataset

Downloaded from: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

You need to join the competition on Kaggle (free) to access the files. Download these 4 CSVs and place them in the `data/` folder:

- `members_v3.csv` — user demographics (city, age, gender)
- `transactions_v2.csv` — subscription and payment history
- `user_logs_v2.csv` — daily listening behavior
- `train_v2.csv` — churn labels (is_churn = 1 means churned)

Data files are not pushed to GitHub because user_logs_v2.csv is 1.4GB.

## How to get the data

This repo does not include data files due to file size limits.

1. Go to: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data
2. Join the competition (free) to unlock the download
3. Download these 4 files only:
   - train_v2.csv.7z
   - members_v3.csv.7z
   - transactions_v2.csv.7z
   - user_logs_v2.csv.7z
4. The files from Kaggle will be seperate zipped folder, from which you need to extract the single .csv files.
5. Place the single .csv files directly in the `data/` folder

> Files download as CSV directly from Kaggle, no extraction needed.
---

## Repo structure
```
kkbox-churn-prediction/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_features.ipynb
│   └── 03_modeling.ipynb
├── models/
│   └── 
├── scoring/
│   └── 
├── dashboard/
│   └── 
├── requirements.txt
└── README.md
```

---

## How to run
```bash
git clone https://github.com/thm-msror/kkbox-churn-prediction
pip install -r requirements.txt
# add data files to data/ folder then run notebooks in order
```

---

## Tools used

- Python, Pandas, Scikit-learn
- PyCaret (AutoML)
- XGBoost
- SHAP
- MLflow
- Power BI

---

## Results

*(will update after modeling)*