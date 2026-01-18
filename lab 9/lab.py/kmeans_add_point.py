# =========================================
# K-Means Re-clustering with a New Point
# =========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# STEP 1: Original 9 Points
# ------------------------------
labels = ["P1","P2","P3","P4","P5","P6","P7","P8","P9"]
X = np.array([
    [1,3],   # P1
    [2,2],   # P2
    [5,8],   # P3
    [8,5],   # P4
    [3,9],   # P5
    [10,7],  # P6
    [3,3],   # P7
    [9,4],   # P8
    [3,7]    # P9
])

print("STEP 1: Original Data Points")
for i in range(len(X)):
    print(labels[i], X[i])

# ------------------------------
# STEP 2: Run K-Means with K=3 on original points
# ------------------------------
k = 3
model_original = KMeans(n_clusters=k, random_state=42, n_init=10)
model_original.fit(X)

labels_original = model_original.labels_
centroids_original = model_original.cluster_centers_

print("\nSTEP 2: Original K-Means Clustering (K=3)")
for i in range(len(X)):
    print(f"{labels[i]} -> Cluster {labels_original[i]}")
print("Centroids:", centroids_original)

# ------------------------------
# STEP 3: Add new user point P10
# ------------------------------
new_label = "P10"
X_new = np.vstack([X, [6,2]])
labels_new = labels + [new_label]

print("\nSTEP 3: Added new point", new_label, "[6,2]")

# ------------------------------
# STEP 4: Run K-Means with K=3 on 10 points
# ------------------------------
model_new = KMeans(n_clusters=k, random_state=42, n_init=10)
model_new.fit(X_new)

labels_new_clusters = model_new.labels_
centroids_new = model_new.cluster_centers_

print("\nSTEP 4: New K-Means Clustering with P10")
for i in range(len(X_new)):
    print(f"{labels_new[i]} -> Cluster {labels_new_clusters[i]}")

print("\nNew Centroids after adding P10:")
for i, c in enumerate(centroids_new):
    print(f"C{i}: ({c[0]:.2f}, {c[1]:.2f})")

# ------------------------------
# STEP 5: Identify which cluster P10 joins
# ------------------------------
cluster_P10 = labels_new_clusters[-1]
print(f"\nP10 joins Cluster {cluster_P10}")

# ------------------------------
# STEP 6: Plot Graph
# ------------------------------
plt.figure()
plt.scatter(X_new[:,0], X_new[:,1])

# Label all points
for i, (x, y) in enumerate(X_new):
    plt.text(x + 0.05, y + 0.05, labels_new[i])

# Plot centroids
plt.scatter(centroids_new[:,0], centroids_new[:,1], marker='x', color='red')

# Label centroids
for i, (x, y) in enumerate(centroids_new):
    plt.text(x + 0.05, y + 0.05, f"C{i}", color='red')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering with New Point P10 (K=3)")
plt.show()

# ------------------------------
# STEP 7: Explanation
# ------------------------------
print("""
STEP 7: Explanation:
- Adding a new data point can change cluster assignments slightly.
- Centroids shift towards the new point depending on which cluster it joins.
- In this example, P10 joins Cluster {}, which causes its centroid to move closer to P10.
- The overall cluster structure remains similar, but centroids adjust to accommodate the new data.
""".format(cluster_P10))
