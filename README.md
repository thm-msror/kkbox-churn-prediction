# KKBox Churn Prediction

- **Course:** DSAI 4103 — Business Analytics  
- **University:** University of Doha for Science & Technology  
- **Student:** Tehreem Masroor | 60302531  
- **Due:** March 29, 2026

---

## Business problem

KKBox is a music streaming platform operating across East and Southeast Asia on a monthly subscription model. Each month, some subscribers choose not to renew which is called churn. 

This project builds an end-to-end machine learning pipeline to predict which users are likely to churn before their subscription expires, so KKBox can intervene with targeted retention offers.

---

## Dataset

**Source:** https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

Join the competition on Kaggle (free) to access the files. Download these 4 files and place them in the `data/` folder:

| File | Description |
|------|-------------|
| `members_v3.csv` | user demographics — city, age, gender |
| `transactions_v2.csv` | subscription and payment history |
| `user_logs_v2.csv` | daily listening behavior |
| `train_v2.csv` | churn labels (is_churn = 1 means churned) |

> data files are not included in this repo due to file size limits. the raw files total ~1.8GB and the merged dataset is ~208MB, both exceed GitHub's 100MB limit.

**If the data is needed, then here is how to get it:**

1. Go to: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data
2. Join the competition (free) to unlock the download
3. Download these 4 files only:
   - train_v2.csv.7z
   - members_v3.csv.7z
   - transactions_v2.csv.7z
   - user_logs_v2.csv.7z
4. The files from Kaggle will be seperate zipped folder, from which you need to extract the single .csv files.
5. Place the single .csv files directly in the `data/` folder

---

## Repo structure

```
kkbox-churn-prediction/
├── notebooks/
│   ├── 01_EDA.ipynb              ← data loading, cleaning, EDA charts
│   ├── 02_features.ipynb        
│   └── 03_modeling.ipynb         
├── models/
│   └── 
├── scoring/
│   └── 
├── dashboard/
│   └── 
├── Visualizations/               ← all charts saved from notebooks
├── requirements.txt
└── README.md
```

---

## How to run

```bash
# 1. clone the repo
git clone https://github.com/thm-msror/kkbox-churn-prediction

# 2. install dependencies
pip install -r requirements.txt

# 3. download data from Kaggle link above and place in data/ folder

# 4. run notebooks in order
#    01_EDA.ipynb → 02_features.ipynb → 03_modeling.ipynb
```

---

## Tools used

| Tool | Purpose |
|------|---------|
| Pandas | data loading, cleaning, merging |
| Scikit-learn | clustering, preprocessing |
| PyCaret | AutoML model comparison |
| XGBoost | final churn prediction model |
| SHAP | model explainability |
| MLflow | experiment tracking |
| Power BI | interactive dashboard |

---

## Results

*(will update after modeling is complete)*
