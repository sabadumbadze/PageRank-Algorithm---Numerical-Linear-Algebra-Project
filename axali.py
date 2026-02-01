import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from PIL import Image 
import os

# ==========================================
# PHASE 1: Real Image Processing (Patch Extraction)
# ==========================================

def get_image_paths():
    """
    Finds the images inside the SAME folder where this python script is located.
    This fixes the 'File not found' error.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Image names
    image_names = ["bric.jpg", "bric.jpg", "bric.jpg"]
    
    full_paths = []
    for name in image_names:
        full_path = os.path.join(script_dir, name)
        full_paths.append(full_path)
        
    return full_paths

def create_dataset_from_images(image_paths, patch_size=10, max_patches_per_image=50):
    """
    1. Loads images.
    2. Converts to Grayscale matrices.
    3. Slices them into 10x10 patches.
    """
    X = []
    
    print(f"Looking for images in: {os.path.dirname(image_paths[0])}")

    for path in image_paths:
        if not os.path.exists(path):
            print(f"❌ Error: File not found at: {path}")
            print("Make sure the .jpg file is in the EXACT same folder as this .py file.")
            continue
            
        print(f"✅ Processing {os.path.basename(path)}...")
        
        try:
            # Load image and convert to Grayscale ('L')
            img = Image.open(path).convert('L')
            img_arr = np.array(img)
            
            # Normalize pixel values to 0.0 - 1.0 range
            img_arr = img_arr / 255.0
            
            h, w = img_arr.shape
            patches_collected = 0
            
            # Extract 10x10 matrices
            # Step by patch_size to avoid overlap
            for r in range(0, h - patch_size, patch_size):
                for c in range(0, w - patch_size, patch_size):
                    if patches_collected >= max_patches_per_image:
                        break
                    
                    patch = img_arr[r : r+patch_size, c : c+patch_size]
                    X.append(patch)
                    patches_collected += 1
                if patches_collected >= max_patches_per_image:
                    break
        except Exception as e:
            print(f"Error reading image {path}: {e}")
                
    return np.array(X)

# 1. Get correct paths
paths = get_image_paths()

# 2. Extract Data
print("--- Extracting 10x10 Matrix Patches ---")
X_matrices = create_dataset_from_images(paths)

if len(X_matrices) == 0:
    print("\nCRITICAL ERROR: No data created. Check if image files exist.")
    exit()

print(f"Dataset Created: {X_matrices.shape} (Samples, 10, 10)")


# ==========================================
# PHASE 2: Algorithm A - Affinity Propagation 
# Metric: Induced Matrix L-Infinity Norm
# ==========================================

def induced_matrix_inf_norm(matrix):
    """ Maximum Absolute Row Sum """
    row_sums = np.sum(np.abs(matrix), axis=1)
    return np.max(row_sums)

def compute_similarity_matrix(X):
    """ Computes Similarity based on negative L-Infinity distance """
    n = X.shape[0]
    similarity = np.zeros((n, n))
    
    print(f"Calculating Similarity Matrix for {n} patches...")
    for i in range(n):
        for j in range(n):
            dist = induced_matrix_inf_norm(X[i] - X[j])
            similarity[i, j] = -dist
    return similarity

sim_matrix = compute_similarity_matrix(X_matrices)

print("Running Affinity Propagation...")
ap = AffinityPropagation(affinity='precomputed', damping=0.7, random_state=42)
ap.fit(sim_matrix)
print(f"AP identified {len(ap.cluster_centers_indices_)} representative textures.")


# ==========================================
# PHASE 3: Algorithm B - K-Means
# Metric: Matrix Frobenius Norm
# ==========================================

print("Running K-Means (Frobenius Logic)...")
# Flatten for Sklearn: (N, 10, 10) -> (N, 100)
n_samples = X_matrices.shape[0]
X_flattened = X_matrices.reshape(n_samples, -1)

# We assume K=3 (assuming you provided roughly 3 types of images)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_flattened)

# Reshape centroids back to Matrix format
centroid_matrices = kmeans.cluster_centers_.reshape(3, 10, 10)


# ==========================================
# VISUALIZATION
# ==========================================

def plot_patches(matrices, title):
    n = len(matrices)
    display_n = min(n, 5) 
    
    plt.figure(figsize=(10, 3))
    plt.suptitle(title)
    
    for i in range(display_n):
        plt.subplot(1, display_n, i + 1)
        plt.imshow(matrices[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"Center {i}")
    plt.show()

# 1. Show K-Means Centroids (The "Average" Textures)
plot_patches(centroid_matrices, "Algorithm B: K-Means Centroids (Averages)")

# 2. Show AP Exemplars (The "Actual" Patches chosen as leaders)
ap_exemplars = X_matrices[ap.cluster_centers_indices_]
plot_patches(ap_exemplars, "Algorithm A: AP Exemplars (Real Patches)")

print("\n--- Done ---")