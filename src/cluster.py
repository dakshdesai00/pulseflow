import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/merchant_features.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

drop_cols = ["card1", "fraud_count"]
train_features = train_df.drop(columns=drop_cols)
test_features = test_df.drop(columns=drop_cols)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

joblib.dump(scaler, "../models/scaler.pkl")

k_range = range(2, 11)
inertias = {}
silhouette_scores = {}

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(train_scaled)
    inertias[k] = km.inertia_
    silhouette_scores[k] = silhouette_score(train_scaled, labels)

print(f"{'k':<5} {'inertia':<15} {'silhouette':<10}")
print("-" * 30)
for k in k_range:
    print(f"{k:<5} {inertias[k]:<15.2f} {silhouette_scores[k]:<10.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest k by silhouette: {best_k}")
print(f"Selected k: 4 (balances silhouette score and business interpretability)")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(train_scaled)

train_df = train_df.copy()
test_df = test_df.copy()
train_df["cluster"] = kmeans.predict(train_scaled)
test_df["cluster"] = kmeans.predict(test_scaled)

joblib.dump(kmeans, "../models/kmeans.pkl")

train_df.to_csv("../data/merchant_clusters_train.csv", index=False)
test_df.to_csv("../data/merchant_clusters_test.csv", index=False)

print("\nTrain cluster fraud rates:")
print(train_df.groupby("cluster")["fraud_rate"].mean().sort_values(ascending=False))

print("\nTest cluster fraud rates:")
print(test_df.groupby("cluster")["fraud_rate"].mean().sort_values(ascending=False))
