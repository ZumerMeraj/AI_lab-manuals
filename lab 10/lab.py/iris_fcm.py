import numpy as np
import skfuzzy as fuzz
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Iris Dataset
# -------------------------------

iris = datasets.load_iris()
X = iris.data
y = iris.target   # actual labels

print("Dataset shape:", X.shape)

# -------------------------------
# 2. Normalize Data (0–1 range)
# -------------------------------

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# FCM requires transpose
X_fcm = X_norm.T

# -------------------------------
# 3. Apply Fuzzy C-Means
# -------------------------------

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fcm, c=3, m=2, error=0.005, maxiter=1000, init=None
)

print("\nCluster Centers (FCM):\n", cntr)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

# Predicted cluster = highest membership
fcm_labels = np.argmax(u, axis=0)

# -------------------------------
# 4. Predicted clusters for first 20 samples
# -------------------------------

print("\nFirst 20 Predicted Clusters (FCM):")
print(fcm_labels[:20])

print("\nFirst 20 Actual Labels:")
print(y[:20])

# -------------------------------
# 5. Accuracy using Majority Mapping
# -------------------------------

def majority_mapping(true_labels, cluster_labels):
    label_map = {}
    for cluster in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster)
        most_common = np.bincount(true_labels[idx]).argmax()
        label_map[cluster] = most_common
    return label_map

mapping = majority_mapping(y, fcm_labels)

mapped_labels = np.array([mapping[c] for c in fcm_labels])
accuracy_fcm = accuracy_score(y, mapped_labels)

print("\nCluster to Label Mapping:", mapping)
print("FCM Accuracy:", accuracy_fcm)

# -------------------------------
# 6. K-Means Clustering
# -------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_norm)

mapping_k = majority_mapping(y, kmeans_labels)
mapped_kmeans = np.array([mapping_k[c] for c in kmeans_labels])
accuracy_kmeans = accuracy_score(y, mapped_kmeans)

print("\nK-Means Accuracy:", accuracy_kmeans)
