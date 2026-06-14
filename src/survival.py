import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

df_raw = pd.read_csv("../data/train_transaction.csv")


txn_counts = df_raw.groupby("card1")["TransactionID"].count()
valid_merchants = txn_counts[txn_counts >= 5].index
df_raw = df_raw[df_raw["card1"].isin(valid_merchants)]


dataset_end = df_raw["TransactionDT"].max()

merchant_time = (
    df_raw.groupby("card1")["TransactionDT"]
    .agg(first_txn="min", last_txn="max")
    .reset_index()
)

merchant_time["T"] = (merchant_time["last_txn"] - merchant_time["first_txn"]) / 86400
merchant_time["E"] = (merchant_time["last_txn"] < dataset_end * 0.95).astype(int)

print(merchant_time["T"].describe())
print(f"\nChurned merchants: {merchant_time['E'].sum()}")
print(f"Active merchants: {(merchant_time['E'] == 0).sum()}")


merchant_df = pd.read_csv("../data/merchant_anomalies.csv")


cox_df = merchant_df.merge(merchant_time[["card1", "T", "E"]], on="card1")


cox_features = [
    "T",
    "E",
    "fraud_rate",
    "avg_amount",
    "cv",
    "pct_product_c",
    "pct_credit",
    "pct_night_txn",
    "total_transactions",
    "is_anomaly",
    "cluster",
]

cox_input = cox_df[cox_features].copy()


cph = CoxPHFitter()
cph.fit(cox_input, duration_col="T", event_col="E")

cph.print_summary()
joblib.dump(cph, "../models/cox_model.pkl")


survival_df = cph.predict_survival_function(cox_input.drop(columns=["T", "E"]))


def get_survival_at(survival_df, day):
    closest_idx = (survival_df.index - day).to_series().abs().argmin()
    return survival_df.iloc[closest_idx].values


cox_df["survival_30"] = get_survival_at(survival_df, 30)
cox_df["survival_60"] = get_survival_at(survival_df, 60)
cox_df["survival_90"] = get_survival_at(survival_df, 90)

cox_df.to_csv("../data/merchant_scores.csv", index=False)
print(
    cox_df[
        [
            "card1",
            "cluster",
            "is_anomaly",
            "fraud_rate",
            "survival_30",
            "survival_60",
            "survival_90",
        ]
    ].head(10)
)
