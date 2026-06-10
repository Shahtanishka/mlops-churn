# How to get good metrics (2 minutes)

The project ships with a synthetic dataset so you can run immediately.
Synthetic data has no real patterns → poor metrics (accuracy ~0.67, F1 ~0.03).

## Fix: download the real Kaggle dataset

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download** (free Kaggle account needed)
3. Unzip it — you'll get `WA_Fn-UseC_-Telco-Customer-Churn.csv`
4. Rename and move it:
   ```
   data/telco_churn.csv
   ```
5. Re-run training:
   ```bat
   python src/train.py
   ```

## Expected metrics on real data

| Metric      | Synthetic | Real dataset |
|-------------|-----------|--------------|
| Accuracy    | ~0.67     | ~0.81        |
| ROC-AUC     | ~0.53     | ~0.85        |
| F1 (churn)  | ~0.03     | ~0.62        |

The real dataset has 7,043 rows with actual churn patterns —
the model will learn properly and metrics jump dramatically.
