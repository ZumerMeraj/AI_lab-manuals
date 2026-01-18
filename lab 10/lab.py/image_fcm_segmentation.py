import cv2
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans

# -------------------------------
# 1. Load Grayscale Image
# -------------------------------

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found. Please place image.jpg in project folder.")
    exit()

plt.figure()
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------
# 2. Convert Image to 1D Pixel Array
# -------------------------------

pixels = img.reshape(-1, 1)
pixels_norm = pixels / 255.0   # normalize

# Transpose for FCM
pixels_fcm = pixels_norm.T

# -------------------------------
# 3. Apply Fuzzy C-Means (3 clusters)
# -------------------------------

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels_fcm, c=3, m=2, error=0.005, maxiter=1000, init=None
)

print("FPC Value:", fpc)

labels_fcm = np.argmax(u, axis=0)

# -------------------------------
# 4. Reconstruct Segmented Image
# -------------------------------

segmented_fcm = labels_fcm.reshape(img.shape)

plt.figure()
plt.title("FCM Segmented Image (3 Clusters)")
plt.imshow(segmented_fcm, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------
# 5. K-Means Segmentation
# -------------------------------

kmeans = KMeans(n_clusters=3, random_state=0)
k_labels = kmeans.fit_predict(pixels_norm)

segmented_kmeans = k_labels.reshape(img.shape)

plt.figure()
plt.title("K-Means Segmented Image")
plt.imshow(segmented_kmeans, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------
# 6. Thresholding Segmentation
# -------------------------------

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.figure()
plt.title("Thresholding Segmentation")
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------
# 7. Effect of Different Clusters
# -------------------------------

for c in [2, 4, 5]:
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels_fcm, c=c, m=2, error=0.005, maxiter=1000
    )
    labels = np.argmax(u, axis=0)
    seg = labels.reshape(img.shape)

    plt.figure()
    plt.title(f"FCM Segmentation with c = {c}")
    plt.imshow(seg, cmap='gray')
    plt.axis('off')
    plt.show()
