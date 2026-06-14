import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/merchant_features.csv")
df = df.drop(columns=["card1", "fraud_count"])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
joblib.dump(scaler, "../models/scaler.pkl")
# inertias = []
# k_range = range(2, 11)

# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(scaled_features)
#     inertias.append(kmeans.inertia_)

# plt.plot(k_range, inertias, marker="o")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

merchant_df = pd.read_csv("../data/merchant_features.csv")
merchant_df["cluster"] = clusters

joblib.dump(kmeans, "../models/kmeans.pkl")
merchant_df.to_csv("../data/merchant_clusters.csv", index=False)

print(merchant_df["cluster"].value_counts())
print(merchant_df.groupby("cluster")["fraud_rate"].mean())

print(
    merchant_df[merchant_df["cluster"] == 3][
        [
            "total_transactions",
            "total_gmv",
            "avg_amount",
            "fraud_rate",
            "pct_product_c",
            "pct_night_txn",
            "cv",
        ]
    ]
)

print(
    merchant_df.groupby("cluster")[
        ["avg_amount", "pct_product_c", "pct_credit", "pct_night_txn", "cv"]
    ].mean()
)
