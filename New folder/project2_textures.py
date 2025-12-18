import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans

# --- PHASE 1: Matrix Data Generation ---

def generate_texture_matrices(n_samples=1500):
    """
    Generates dataset of N samples.
    Each sample is a 10x10 Matrix (Grayscale Image Patch).
    """
    X = [] # List to hold matrices
    labels = []
    
    # We generate 3 types of textures
    samples_per_cluster = n_samples // 3
    
    for _ in range(samples_per_cluster):
        # 1. Horizontal Lines (e.g., Brick)
        mat = np.zeros((10, 10))
        for r in range(0, 10, 2):
            mat[r, :] = 1.0 # White line
        # Add noise
        mat += np.random.normal(0, 0.1, (10, 10))
        X.append(mat)
        labels.append(0)
        
        # 2. Vertical Lines (e.g., Grating)
        mat = np.zeros((10, 10))
        for c in range(0, 10, 2):
            mat[:, c] = 1.0 # White line
        # Add noise
        mat += np.random.normal(0, 0.1, (10, 10))
        X.append(mat)
        labels.append(1)
        
        # 3. Random Noise (e.g., Sand)
        mat = np.random.rand(10, 10)
        X.append(mat)
        labels.append(2)
    
    # Convert to Numpy Array of shape (N, 10, 10)
    return np.array(X), np.array(labels)

print("Generating 10x10 Matrix Texture Data...")
X_matrices, y_true = generate_texture_matrices()
print(f"Data Shape: {X_matrices.shape} (N_samples, Height, Width)")
print("Note: We are strictly maintaining (10, 10) Matrix format.")
print("-" * 30)

# --- PHASE 2: Affinity Propagation with Induced Matrix L-Infinity Norm ---

def induced_matrix_inf_norm(matrix):
    """
    Calculates the Induced Matrix L-Infinity Norm.
    Definition: The maximum absolute row sum.
    """
    # Sum absolute values across columns (axis 1), then take the max of those sums
    row_sums = np.sum(np.abs(matrix), axis=1)
    return np.max(row_sums)

def compute_similarity_matrix(X):
    """
    Computes a generic (N x N) similarity matrix.
    Similarity = -1 * Distance
    Distance metric: Induced Matrix L-Infinity Norm of (MatA - MatB).
    """
    n = X.shape[0]
    similarity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate Difference Matrix
            diff_matrix = X[i] - X[j]
            # Calculate Norm of the difference
            dist = induced_matrix_inf_norm(diff_matrix)
            # Similarity is negative distance
            similarity[i, j] = -dist
            
    return similarity

print("Computing Similarity Matrix using L-Infinity Norm...")
sim_matrix = compute_similarity_matrix(X_matrices)

print("Running Affinity Propagation (Precomputed)...")
# affinity='precomputed' tells sklearn we calculated the norms ourselves
ap = AffinityPropagation(affinity='precomputed', damping=0.5, random_state=42)
ap.fit(sim_matrix)

print(f"AP identified {len(ap.cluster_centers_indices_)} clusters.")

# --- PHASE 3: K-Means with Matrix Frobenius Norm ---

def matrix_frobenius_norm(matrix):
    """
    Calculates Matrix Frobenius Norm.
    Formula: sqrt(sum of all elements squared)
    """
    return np.sqrt(np.sum(matrix ** 2))

print("\nRunning K-Means using Frobenius Norm logic...")

# Sklearn KMeans requires (N_samples, N_features) input.
# To use Frobenius norm on Matrices, we flatten (10,10) -> (100,)
# Mathematically: Euclidean(Flattened_Vector) == Frobenius(Matrix)
n_samples = X_matrices.shape[0]
X_flattened = X_matrices.reshape(n_samples, -1)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_flattened)

# Reshape the resulting centroids back into 10x10 Matrices for visualization
# This proves we are treating the results as 'Centroid Matrices'
centroid_matrices = kmeans.cluster_centers_.reshape(3, 10, 10)

# --- VISUALIZATION ---

def plot_texture_centroids(centroids, title):
    plt.figure(figsize=(10, 4))
    plt.suptitle(title)
    
    for i, matrix in enumerate(centroids):
        plt.subplot(1, 3, i + 1)
        plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Cluster Center {i}")
        plt.axis('off') # Hide axes for image look
    plt.show()

# Visualize the centers discovered by K-Means
plot_texture_centroids(centroid_matrices, "Algorithm B: K-Means Centroids (Frobenius Norm)")

# For Affinity Propagation, we find the 'exemplars' (actual data points that are centers)
ap_exemplars = X_matrices[ap.cluster_centers_indices_]
plot_texture_centroids(ap_exemplars, "Algorithm A: AP Exemplars (L-Infinity Norm)")

print("\n--- COMPARISON REPORT ---")
print("1. AP (L-Infinity): Selected actual images from the dataset as representatives (Exemplars).")
print("2. K-Means (Frobenius): Created 'average' matrices. Notice how the centroids look cleaner/smoother than raw data.")
print("3. Visual Check: The centroids should clearly show Horizontal lines, Vertical lines, and Noise.")