# ==============================
# K-Means using scikit-learn
# Step-by-step for Lab
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# STEP 1: Enter Data Points
# ------------------------------
labels = ["P1","P2","P3","P4","P5","P6","P7","P8","P9"]

X = np.array([
    [1, 3],   # P1
    [2, 2],   # P2
    [5, 8],   # P3
    [8, 5],   # P4
    [3, 9],   # P5
    [10, 7],  # P6
    [3, 3],   # P7
    [9, 4],   # P8
    [3, 7]    # P9
])

print("STEP 1: Data Points")
for i in range(len(X)):
    print(labels[i], X[i])

# ------------------------------
# STEP 2: Try Different K values
# ------------------------------
for k in [2, 3, 4]:

    print("\n===================================")
    print(f"STEP 2: Applying K-Means with K = {k}")
    print("===================================")

    # ------------------------------
    # STEP 3: Create KMeans Model
    # ------------------------------
    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    # ------------------------------
    # STEP 4: Fit Model to Data
    # ------------------------------
    model.fit(X)

    # ------------------------------
    # STEP 5: Get Cluster Labels
    # ------------------------------
    cluster_ids = model.labels_

    print("\nCluster assignment of each point:")
    for i in range(len(X)):
        print(f"{labels[i]} -> Cluster {cluster_ids[i]}")

    # ------------------------------
    # STEP 6: Count Points per Cluster
    # ------------------------------
    print("\nNumber of points in each cluster:")
    for i in range(k):
        count = list(cluster_ids).count(i)
        print(f"Cluster {i}: {count} points")

    # ------------------------------
    # STEP 7: Get Final Centroids
    # ------------------------------
    centroids = model.cluster_centers_

    print("\nFinal Centroid Locations:")
    for i, c in enumerate(centroids):
        print(f"Cluster {i} centroid = ({c[0]:.2f}, {c[1]:.2f})")

    # ------------------------------
    # STEP 8: Plot Graph
    # ------------------------------
    plt.figure()
    plt.scatter(X[:,0], X[:,1])

    # label points
    for i, (x, y) in enumerate(X):
        plt.text(x + 0.05, y + 0.05, labels[i])

    # plot centroids
    plt.scatter(centroids[:,0], centroids[:,1], marker='x')

    # label centroids
    for i, (x, y) in enumerate(centroids):
        plt.text(x + 0.05, y + 0.05, f"C{i}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"K-Means using scikit-learn (K = {k})")
    plt.show()

print("\n--- Program Finished Successfully ---")
