import math
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Define Points
# -------------------------
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

# -------------------------
# Step 2: Initial Centroids
# -------------------------
centroids = {
    "C1": points["P7"],  # (3,3)
    "C2": points["P9"],  # (3,7)
    "C3": points["P8"]   # (9,4)
}

# -------------------------
# Step 3: Distance Formula
# -------------------------
def euclidean(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

# -------------------------
# Step 4: K-Means (2 Iterations)
# -------------------------
for it in range(1, 3):

    print("\n==============================")
    print(f"ITERATION {it}")
    print("==============================")

    clusters = {"C1": [], "C2": [], "C3": []}

    # ---- Assignment Step ----
    print("\nDistance Calculation & Assignment:")

    for label, point in points.items():

        d1 = euclidean(point, centroids["C1"])
        d2 = euclidean(point, centroids["C2"])
        d3 = euclidean(point, centroids["C3"])

        print(f"{label} distances -> C1:{d1:.2f}, C2:{d2:.2f}, C3:{d3:.2f}")

        nearest = min(
            {"C1": d1, "C2": d2, "C3": d3},
            key=lambda x: {"C1": d1, "C2": d2, "C3": d3}[x]
        )

        clusters[nearest].append(label)
        print(f"→ {label} assigned to {nearest}")

    # ---- Update Step ----
    print("\nNew Centroid Calculation:")

    new_centroids = {}
    for c, plist in clusters.items():
        xs = [points[p][0] for p in plist]
        ys = [points[p][1] for p in plist]

        new_x = sum(xs) / len(xs)
        new_y = sum(ys) / len(ys)

        new_centroids[c] = (new_x, new_y)

        print(f"{c}: mean of {plist} = ({new_x:.2f}, {new_y:.2f})")

    centroids = new_centroids

# -------------------------
# Step 5: Final Result
# -------------------------
print("\n==============================")
print("FINAL CLUSTERS AFTER 2 ITERATIONS")
print("==============================")

for c, plist in clusters.items():
    print(c, ":", plist)

print("\nFINAL CENTROIDS:")
for c, val in centroids.items():
    print(c, ":", val)

# -------------------------
# Step 6: Plot Graph
# -------------------------
x = [p[0] for p in points.values()]
y = [p[1] for p in points.values()]

plt.figure()
plt.scatter(x, y)

# label points
for label, (px, py) in points.items():
    plt.text(px + 0.05, py + 0.05, label)

# plot centroids
cx = [v[0] for v in centroids.values()]
cy = [v[1] for v in centroids.values()]
plt.scatter(cx, cy, marker='x')

# label centroids
for c, (px, py) in centroids.items():
    plt.text(px + 0.05, py + 0.05, c)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering (K=3, After 2 Iterations)")
plt.show()
