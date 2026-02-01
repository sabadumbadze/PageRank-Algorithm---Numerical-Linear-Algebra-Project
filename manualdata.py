import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==========================================
# PHASE 1: BIG DATA GENERATION (Data Augmentation)
# ==========================================

def generate_big_data(samples_per_group=1000):
    """
    Generates 'Big Data' by taking seed planets and creating 
    thousands of variations with random noise.
    """
    # 1. Define the 'Seed' types (The mechanical data)
    # [Radius, Period, Temp, Distance, Mass]
    seeds = np.array([
        [1.0, 365.0, 288.0, 10.0, 1.0],      # Earth-like
        [11.2, 4000.0, 165.0, 5.2, 317.0],   # Jupiter-like
        [12.1, 4.0, 1200.0, 150.0, 300.0],   # Hot Jupiter
        [3.9, 60000.0, 50.0, 30.0, 17.0],    # Neptune-like
        [1.5, 20.0, 450.0, 2.0, 4.0]         # Super-Earth
    ])
    
    generated_data = []
    
    print(f"Generating {samples_per_group * len(seeds)} exoplanets...")
    
    # 2. Loop through seeds and create variations
    for seed in seeds:
        # Create 'samples_per_group' copies of this seed
        # Add Random Gaussian Noise: mean=0, scale=varies per feature
        # We scale noise relative to the feature size to keep it realistic
        
        # Noise scales: [Radius, Period, Temp, Dist, Mass]
        noise_scales = [0.2, 50.0, 20.0, 5.0, 1.0] 
        
        # Generate random noise array
        noise = np.random.normal(loc=0.0, scale=noise_scales, size=(samples_per_group, 5))
        
        # Add noise to the seed
        new_batch = seed + noise
        
        # Ensure no negative values (Physics constraint: Radius > 0)
        new_batch = np.abs(new_batch)
        
        generated_data.append(new_batch)
    
    # Stack all groups into one big matrix
    X = np.vstack(generated_data)
    
    # Standardize (Z-Score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Load Big Data
# Change 1000 to 10000 if you want HUGE data
X = generate_big_data(samples_per_group=1000) 

print(f"Big Data Ready: {X.shape[0]} Total Planets created.")

# ==========================================
# PHASE 2: ALGORITHM A (K-Means Scratch - L2)
# ==========================================

class KMeansScratch:
    def __init__(self, k=5, max_iters=100, tol=0.001): # Increased k to 5 to match seeds
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def _l2_norm(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def fit(self, X):
        n_samples = X.shape[0]
        # Random initialization
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])
            
            # Handle empty clusters (rare in big data, but good practice)
            # If a cluster is empty, re-initialize it to a random point
            for j in range(self.k):
                if np.isnan(new_centroids[j]).any():
                     new_centroids[j] = X[np.random.randint(n_samples)]

            diff = sum(self._l2_norm(self.centroids[j], new_centroids[j]) for j in range(self.k))
            
            if diff < self.tol:
                print(f"K-Means converged at iteration {i}")
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # Optimized distance calculation for big data (Vectorized)
        # Doing a loop over 5000 points is slow in pure python, but fine for this assignment.
        # For speed, we just loop centroids.
        labels = []
        for point in X:
            dists = [self._l2_norm(point, c) for c in self.centroids]
            labels.append(np.argmin(dists))
        return np.array(labels)

print("Running K-Means (Scratch L2) on Big Data...")
kmeans = KMeansScratch(k=5) # 5 Clusters
kmeans.fit(X)

# ==========================================
# PHASE 3: ALGORITHM B (DBSCAN - L1) & VISUALIZATION
# ==========================================

print("Running DBSCAN (L1) on Big Data...")
# eps and min_samples tuned for the dense Big Data
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='manhattan')
dbscan_labels = dbscan.fit_predict(X)

# Plotting
plt.figure(figsize=(14, 6))

# Plot only a subset if data is HUGE (e.g. plot first 1000 points) to save rendering time
# or plot all with small marker size
plot_limit = 2000 
X_plot = X[:plot_limit]
labels_k_plot = kmeans.labels[:plot_limit]
labels_d_plot = dbscan_labels[:plot_limit]

plt.subplot(1, 2, 1)
plt.scatter(X_plot[:, 0], X_plot[:, 2], c=labels_k_plot, cmap='viridis', s=10, alpha=0.5)
plt.title(f"K-Means L2 (First {plot_limit} samples)")
plt.xlabel("Radius (Scaled)")
plt.ylabel("Temperature (Scaled)")

plt.subplot(1, 2, 2)
unique_labels = set(labels_d_plot)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1: col = [0, 0, 0, 1] # Black for Noise
    class_member_mask = (labels_d_plot == k)
    xy = X_plot[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 2], 'o', markerfacecolor=tuple(col), markeredgecolor='none', markersize=4, alpha=0.5)

plt.title(f"DBSCAN L1 (First {plot_limit} samples)")
plt.xlabel("Radius (Scaled)")
plt.ylabel("Temperature (Scaled)")

plt.tight_layout()
plt.show()