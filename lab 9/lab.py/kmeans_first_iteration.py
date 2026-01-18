# ============================================
# K-Means First Iteration (Manual Distance Table)
# ============================================

import math
import matplotlib.pyplot as plt

# ------------------------------
# STEP 1: Define Points
# ------------------------------
points = {
    "P1": (1, 3),
    "P2": (2, 2),
    "P3": (5, 8),
    "P4": (8, 5),
    "P5": (3, 9),
    "P6": (10, 7),
    "P7": (3, 3),
    "P8": (9, 4),
    "P9": (3, 7)
}

# ------------------------------
# STEP 2: Initial Centroids
# ------------------------------
centroids = {
    "C1": (3, 3),
    "C2": (3, 7),
    "C3": (9, 4)
}

# ------------------------------
# STEP 3: Euclidean Distance Function
# ------------------------------
def euclidean(p, c):
    return math.sqrt((p[0]-c[0])**2 + (p[1]-c[1])**2)

# ------------------------------
# STEP 4: Compute Distances & Assign Clusters (First Iteration)
# ------------------------------
distance_table = []

clusters = {"C1": [], "C2": [], "C3": []}

print("Distance Table (First Iteration):")
print("Point\tDist to C1\tDist to C2\tDist to C3\tAssigned Cluster")

for label, point in points.items():
    d1 = euclidean(point, centroids["C1"])
    d2 = euclidean(point, centroids["C2"])
    d3 = euclidean(point, centroids["C3"])

    # Assign cluster based on minimum distance
    min_dist = min(d1, d2, d3)
    if min_dist == d1:
        cluster = "C1"
    elif min_dist == d2:
        cluster = "C2"
    else:
        cluster = "C3"

    clusters[cluster].append(label)
    distance_table.append([label, d1, d2, d3, cluster])

    print(f"{label}\t{d1:.2f}\t\t{d2:.2f}\t\t{d3:.2f}\t\t{cluster}")

# ------------------------------
# STEP 5: Compute New Centroids
# ------------------------------
new_centroids = {}
for c, plist in clusters.items():
    xs = [points[p][0] for p in plist]
    ys = [points[p][1] for p in plist]
    new_centroids[c] = (sum(xs)/len(xs), sum(ys)/len(ys))

print("\nNew Centroids After First Iteration:")
for c, val in new_centroids.items():
    print(f"{c}: ({val[0]:.2f}, {val[1]:.2f})")

# ------------------------------
# STEP 6: Plot Graph (First Iteration)
# ------------------------------
plt.figure()
x_points = [p[0] for p in points.values()]
y_points = [p[1] for p in points.values()]
plt.scatter(x_points, y_points)

# Label all points
for label, (x, y) in points.items():
    plt.text(x + 0.1, y + 0.1, label)

# Plot new centroids
cx = [v[0] for v in new_centroids.values()]
cy = [v[1] for v in new_centroids.values()]
plt.scatter(cx, cy, marker='x', color='red')

# Label centroids
for c, (x, y) in new_centroids.items():
    plt.text(x + 0.1, y + 0.1, c, color='red')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means First Iteration")
plt.show()
