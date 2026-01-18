import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -------------------------------
# 1. Create Customer Dataset
# -------------------------------

data = {
    "Age": [22, 25, 28, 35, 40, 45, 23, 31, 50, 48, 29, 34],
    "Income": [15, 18, 20, 35, 45, 50, 16, 30, 60, 55, 28, 33],
    "Spending": [80, 75, 70, 50, 40, 35, 78, 60, 25, 30, 65, 55]
}

df = pd.DataFrame(data)
print("\nOriginal Dataset:\n", df)

# -------------------------------
# 2. Normalize using MinMaxScaler
# -------------------------------

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(df)

# FCM requires transpose
X_fcm = X_norm.T

# -------------------------------
# 3. Apply Fuzzy C-Means
# -------------------------------

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fcm, c=3, m=2, error=0.005, maxiter=1000, init=None
)

print("\nCluster Centers (Normalized):\n", cntr)
print("\nMembership Matrix (first 10 customers):\n", u[:, :10])
print("\nFuzzy Partition Coefficient (FPC):", fpc)

# -------------------------------
# 4. Assign Cluster with Max Membership
# -------------------------------

fcm_labels = np.argmax(u, axis=0)
df["FCM Cluster"] = fcm_labels

print("\nCustomer Clusters using FCM:\n", df)

# -------------------------------
# 5. K-Means for Comparison
# -------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_norm)
df["KMeans Cluster"] = kmeans_labels

print("\nCustomer Clusters using K-Means:\n", df)
