import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../data/merchant_clusters.csv")

card1 = df["card1"]
cluster = df["cluster"]

features = df.drop(columns=["card1", "fraud_count", "cluster"])

scaler = joblib.load("../models/scaler.pkl")
scaled_features = scaler.transform(features)


iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
iso_forest.fit(scaled_features)

predictions = iso_forest.predict(scaled_features)

scores = iso_forest.score_samples(scaled_features)

df["is_anomaly"] = (predictions == -1).astype(int)
df["anomaly_score"] = scores

joblib.dump(iso_forest, "../models/isolation_forest.pkl")
df.to_csv("../data/merchant_anomalies.csv", index=False)

print(f"Total anomalies detected: {df['is_anomaly'].sum()}")
print(f"\nAnomalies per cluster:")
print(df.groupby("cluster")["is_anomaly"].sum())
print(f"\nAvg fraud rate — normal vs anomaly:")
print(df.groupby("is_anomaly")["fraud_rate"].mean())
