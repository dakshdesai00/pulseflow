import pandas as pd

df = pd.read_csv("../data/train_transaction.csv")

txn_counts = df.groupby("card1")["TransactionID"].count()
valid_merchants = txn_counts[txn_counts >= 5].index
df = df[df["card1"].isin(valid_merchants)]

print(f"Transactions after filter: {len(df):,}")
print(f"Merchants after filter: {df['card1'].nunique():,}")

amount_features = (
    df.groupby("card1")["TransactionAmt"]
    .agg(
        total_gmv="sum",
        avg_amount="mean",
        std_amount="std",
        max_amount="max",
        min_amount="min",
    )
    .reset_index()
)

amount_features["max_avg_ratio"] = (
    amount_features["max_amount"] / amount_features["avg_amount"]
)

amount_features["cv"] = amount_features["std_amount"] / amount_features["avg_amount"]

print(amount_features.head())

fraud_features = (
    df.groupby("card1")
    .agg(
        total_transactions=("TransactionID", "count"),
        fraud_count=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"),
        pct_product_c=("ProductCD", lambda x: (x == "C").mean()),
        pct_product_w=("ProductCD", lambda x: (x == "W").mean()),
    )
    .reset_index()
)

print(fraud_features.head())

df["hour"] = (df["TransactionDT"] // 3600) % 24

time_card_features = (
    df.groupby("card1")
    .agg(
        avg_hour=("hour", "mean"),
        pct_credit=("card6", lambda x: (x == "credit").mean()),
        pct_night_txn=("hour", lambda x: ((x >= 0) & (x <= 6)).mean()),
    )
    .reset_index()
)

print(time_card_features.head())


merchant_features = amount_features.merge(fraud_features, on="card1").merge(
    time_card_features, on="card1"
)

print(merchant_features.shape)
print(merchant_features.head())
merchant_features.to_csv("../data/merchant_features.csv", index=False)
print(merchant_features.columns.tolist())
