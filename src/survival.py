import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

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

train_df = pd.read_csv("../data/merchant_anomalies_train.csv")
test_df = pd.read_csv("../data/merchant_anomalies_test.csv")

train_df = train_df.merge(merchant_time[["card1", "T", "E"]], on="card1")
test_df = test_df.merge(merchant_time[["card1", "T", "E"]], on="card1")

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

train_cox = train_df[cox_features].copy()
test_cox = test_df[cox_features].copy()

penalizer_values = [0.0, 0.1, 0.5, 1.0]

print(f"{'penalizer':<12} {'train_concordance':<20} {'test_concordance'}")
print("-" * 50)

for p in penalizer_values:
    cph = CoxPHFitter(penalizer=p)
    cph.fit(train_cox, duration_col="T", event_col="E")
    train_c = cph.concordance_index_
    test_c = cph.score(test_cox, scoring_method="concordance_index")
    print(f"{p:<12} {train_c:<20.4f} {test_c:.4f}")

print(
    "\nSelected penalizer: 0.1 (resolves convergence warning while maintaining concordance)"
)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(train_cox, duration_col="T", event_col="E")
cph.print_summary()

joblib.dump(cph, "../models/cox_model.pkl")


def get_survival_at(survival_df, day):
    closest_idx = (survival_df.index - day).to_series().abs().argmin()
    return survival_df.iloc[closest_idx].values


for df, cox_input, name in [
    (train_df, train_cox, "train"),
    (test_df, test_cox, "test"),
]:
    survival_df = cph.predict_survival_function(cox_input.drop(columns=["T", "E"]))
    df["survival_30"] = get_survival_at(survival_df, 30)
    df["survival_60"] = get_survival_at(survival_df, 60)
    df["survival_90"] = get_survival_at(survival_df, 90)
    df.to_csv(f"../data/merchant_scores_{name}.csv", index=False)
    print(
        f"\n{name} concordance: {cph.score(df[cox_features], scoring_method='concordance_index'):.4f}"
    )
    print(
        df[
            [
                "card1",
                "cluster",
                "is_anomaly",
                "fraud_rate",
                "survival_30",
                "survival_60",
                "survival_90",
            ]
        ].head(5)
    )
