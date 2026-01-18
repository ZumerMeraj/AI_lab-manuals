import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# -------------------------------
# 1. Generate 2D Dataset (3 Clusters)
# -------------------------------

np.random.seed(42)

cluster1 = np.random.normal([2, 2], 0.6, (100, 2))
cluster2 = np.random.normal([7, 7], 0.6, (100, 2))
cluster3 = np.random.normal([2, 8], 0.6, (100, 2))

data = np.vstack((cluster1, cluster2, cluster3))
X = data.T   # FCM requires transpose

# -------------------------------
# 2. Apply Fuzzy C-Means
# -------------------------------

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X, c=3, m=2, error=0.005, maxiter=1000, init=None
)

print("Cluster Centers (FCM):\n", cntr)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

# -------------------------------
# 3. Plot FCM Result
# -------------------------------

labels = np.argmax(u, axis=0)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='X', s=200)
plt.title("Fuzzy C-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# -------------------------------
# 4. Membership Values for 5 Random Points
# -------------------------------

print("\nMembership values for 5 random points:\n")

indices = np.random.choice(data.shape[0], 5, replace=False)

for i in indices:
    print(f"Point {data[i]} -> Memberships: {u[:, i]}")

# -------------------------------
# 5. K-Means Clustering for Comparison
# -------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(data)
k_centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=k_labels, cmap='viridis')
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='red', marker='X', s=200)
plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
