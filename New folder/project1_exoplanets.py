from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# ==========================================
# PHASE 1: Data Simulation (Vector Data)
# ==========================================


def generate_exoplanet_data(n_samples=3000):
    """
    Simulates 5D Vector Data for Exoplanets.
    Features: [Radius, Period, Temp, Distance, Mass]
    """
    # Generate synthetic clusters (3 centers: Earth-like, Hot Jupiter, Cold Neptune)
    X, y_true = make_blobs(n_samples=n_samples, centers=3,
                           n_features=5, cluster_std=0.6, random_state=42)

    # Standardize the data (Mean=0, Std=1) for correct distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


# Load the data
print("--- Generating Exoplanet Vector Data ---")
X = generate_exoplanet_data()
print(f"Data Shape: {X.shape} (300 Planets, 5 Features)")
print("-" * 30)


# ==========================================
# PHASE 2: Algorithm A - K-Means From Scratch
# Constraint: Must use Vector L2 Norm (Euclidean)
# ==========================================

class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=0.001):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def _l2_norm(self, v1, v2):
        """
        Computes Euclidean Distance (Vector L2 Norm).
        Formula: sqrt(sum((x - y)^2))
        """
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def fit(self, X):
        n_samples, n_features = X.shape

        # 1. Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Assign clusters
            self.labels = self._assign_clusters(X)

            # 3. Update centroids
            new_centroids = np.array(
                [X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # Check convergence using L2 norm
            diff = 0
            for j in range(self.k):
                diff += self._l2_norm(self.centroids[j], new_centroids[j])

            if diff < self.tol:
                print(f"Algorithm A (K-Means): Converged at iteration {i}")
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        labels = []
        for point in X:
            # Calculate L2 distance to every centroid
            distances = [self._l2_norm(point, centroid)
                         for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)


# Execute Algorithm A
print("Running Algorithm A: K-Means (Scratch, L2 Norm)...")
kmeans = KMeansScratch(k=3)
kmeans.fit(X)


# ==========================================
# PHASE 3: Algorithm B - DBSCAN (Library)
# Constraint: Must use Vector L1 Norm (Manhattan)
# ==========================================

print("Running Algorithm B: DBSCAN (Sklearn, L1 Norm)...")

# metric='manhattan' forces the L1 Norm usage
dbscan = DBSCAN(eps=2, min_samples=5, metric='manhattan')
dbscan_labels = dbscan.fit_predict(X)


# ==========================================
# VISUALIZATION & COMPARISON
# ==========================================

def plot_clusters(X, labels, title, subplot_index):
    # Plotting Feature 0 (Radius) vs Feature 2 (Temperature)
    plt.subplot(1, 2, subplot_index)
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        # DBSCAN uses -1 for noise
        if lbl == -1:
            color = 'k'
            marker = 'x'
            label_name = "Noise"
        else:
            color = None
            marker = 'o'
            label_name = f"Cluster {lbl}"

        mask = (labels == lbl)
        plt.scatter(X[mask, 0], X[mask, 2], label=label_name,
                    c=color, marker=marker, alpha=0.7)

    plt.title(title)
    plt.xlabel("Radius (Scaled)")
    plt.ylabel("Temperature (Scaled)")
    plt.legend()


plt.figure(figsize=(12, 5))

# Plot 1: K-Means (L2)
plot_clusters(X, kmeans.labels, "Algo A: K-Means (L2 Euclidean)", 1)

# Plot 2: DBSCAN (L1)
plot_clusters(X, dbscan_labels, "Algo B: DBSCAN (L1 Manhattan)", 2)

plt.tight_layout()
print("Displaying plots...")
plt.show()

print("\n--- COMPARISON REPORT ---")
print("1. Norm Difference: K-Means used straight-line distance (L2). DBSCAN used grid-like distance (L1).")
print("2. Shape: K-Means forced 3 distinct spherical blobs.")
print("3. Noise: DBSCAN (L1) detected outliers (marked as 'x'). K-Means forced outliers into the nearest cluster.")
# --- ADDENDUM: Performance Metrics for Report ---

print("\n--- CALCULATING METRICS FOR REPORT ---")

# Calculate Silhouette Score for K-Means (L2)
# We use 'euclidean' because Algorithm A used L2
score_kmeans = silhouette_score(X, kmeans.labels, metric='euclidean')
print(f"K-Means (L2) Silhouette Score: {score_kmeans:.4f}")

# Calculate Silhouette Score for DBSCAN (L1)
# We use 'manhattan' because Algorithm B used L1
# Note: We filter out Noise (-1) for a fair score, or include it to see penalty
if len(set(dbscan_labels)) > 1:  # Ensure we have at least 1 cluster and noise
    score_dbscan = silhouette_score(X, dbscan_labels, metric='manhattan')
    print(f"DBSCAN (L1) Silhouette Score:  {score_dbscan:.4f}")
else:
    print("DBSCAN found only noise or 1 cluster, cannot calculate silhouette.")
