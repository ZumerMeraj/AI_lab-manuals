import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv("covid_world.csv")

# Check column names
print("\nColumns in dataset:\n", df.columns)

# -------------------------------
# 2. Select Countries
# -------------------------------

countries = ["Pakistan", "India", "China", "Iran", "Afghanistan"]

df_sel = df[df["Country"].isin(countries)]

# -------------------------------
# 3. Select Features
# -------------------------------

features = df_sel[["TotalCases", "TotalDeaths", "Population"]]

print("\nSelected Data:\n", features)

# -------------------------------
# 4. Normalize Features
# -------------------------------

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(features)

X_fcm = X_norm.T

# -------------------------------
# 5. Apply Fuzzy C-Means (2 Clusters)
# -------------------------------

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fcm, c=2, m=2, error=0.005, maxiter=1000, init=None
)

print("\nCluster Centers (Normalized):\n", cntr)
print("\nMembership Matrix:\n", u)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

# -------------------------------
# 6. Final Cluster Assignment
# -------------------------------

labels_fcm = np.argmax(u, axis=0)
df_sel["FCM Cluster"] = labels_fcm

print("\nFinal FCM Clusters:\n", df_sel[["Country", "FCM Cluster"]])

# -------------------------------
# 7. K-Means Clustering
# -------------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
k_labels = kmeans.fit_predict(X_norm)

df_sel["KMeans Cluster"] = k_labels

print("\nK-Means Clusters:\n", df_sel[["Country", "KMeans Cluster"]])
