import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

train_df = pd.read_csv("../data/merchant_clusters_train.csv")
test_df = pd.read_csv("../data/merchant_clusters_test.csv")

drop_cols = ["card1", "fraud_count", "cluster"]
train_features = train_df.drop(columns=drop_cols)
test_features = test_df.drop(columns=drop_cols)

scaler = joblib.load("../models/scaler.pkl")
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)

contamination_values = [0.03, 0.05, 0.07, 0.10]

print(
    f"{'contamination':<15} {'train_anomalies':<18} {'train_fraud_rate_anomaly':<25} {'train_fraud_rate_normal'}"
)
print("-" * 75)

for c in contamination_values:
    iso = IsolationForest(contamination=c, random_state=42, n_estimators=100)
    iso.fit(train_scaled)
    preds = iso.predict(train_scaled)
    train_df["is_anomaly"] = (preds == -1).astype(int)
    fraud_rates = train_df.groupby("is_anomaly")["fraud_rate"].mean()
    anomaly_fraud = fraud_rates.get(1, 0)
    normal_fraud = fraud_rates.get(0, 0)
    print(
        f"{c:<15} {train_df['is_anomaly'].sum():<18} {anomaly_fraud:<25.4f} {normal_fraud:.4f}"
    )

print(
    "\nSelected contamination: 0.05 (maximizes separation between anomaly and normal fraud rates)"
)

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
iso_forest.fit(train_scaled)

train_df["is_anomaly"] = (iso_forest.predict(train_scaled) == -1).astype(int)
train_df["anomaly_score"] = iso_forest.score_samples(train_scaled)

test_df["is_anomaly"] = (iso_forest.predict(test_scaled) == -1).astype(int)
test_df["anomaly_score"] = iso_forest.score_samples(test_scaled)

joblib.dump(iso_forest, "../models/isolation_forest.pkl")

train_df.to_csv("../data/merchant_anomalies_train.csv", index=False)
test_df.to_csv("../data/merchant_anomalies_test.csv", index=False)

print("\nTrain anomalies per cluster:")
print(train_df.groupby("cluster")["is_anomaly"].sum())

print("\nTest anomalies per cluster:")
print(test_df.groupby("cluster")["is_anomaly"].sum())

print("\nTrain fraud rate — normal vs anomaly:")
print(train_df.groupby("is_anomaly")["fraud_rate"].mean())

print("\nTest fraud rate — normal vs anomaly:")
print(test_df.groupby("is_anomaly")["fraud_rate"].mean())
