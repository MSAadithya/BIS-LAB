from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

# -------------------------------
# Grey Wolf Optimizer for Clustering
# -------------------------------

def fitness_function(data, centroids):
    """Calculate the sum of squared distances (intra-cluster variance)."""
    labels, distances = pairwise_distances_argmin_min(data, centroids)
    return np.sum(distances ** 2)

def gwo_clustering(data, n_clusters=3, n_wolves=10, max_iter=50):
    n_features = data.shape[1]
    wolves = np.random.uniform(np.min(data), np.max(data), 
                               (n_wolves, n_clusters, n_features))

    alpha, beta, delta = np.copy(wolves[0]), np.copy(wolves[1]), np.copy(wolves[2])
    alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # linearly decreases from 2 to 0

        for i in range(n_wolves):
            fitness = fitness_function(data, wolves[i])

            # Update alpha, beta, delta
            if fitness < alpha_score:
                delta_score, beta_score, alpha_score = beta_score, alpha_score, fitness
                delta, beta, alpha = np.copy(beta), np.copy(alpha), np.copy(wolves[i])
            elif fitness < beta_score:
                delta_score, beta_score = beta_score, fitness
                delta, beta = np.copy(beta), np.copy(wolves[i])
            elif fitness < delta_score:
                delta_score = fitness
                delta = np.copy(wolves[i])

        # Update positions
        for i in range(n_wolves):
            for j in range(n_clusters):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i][j])
                X3 = delta[j] - A3 * D_delta

                wolves[i][j] = (X1 + X2 + X3) / 3

        # Optional: print progress
        if (t + 1) % 10 == 0:
            print(f"Iteration {t+1}/{max_iter} | Best Fitness = {alpha_score:.4f}")

    return alpha, alpha_score  # Best centroids and score


# -------------------------------
# Example Run
# -------------------------------

# Create synthetic dataset
data, true_labels = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.6, random_state=42)

# Run Grey Wolf Optimization Clustering
best_centroids, best_score = gwo_clustering(data, n_clusters=3, n_wolves=15, max_iter=50)

# Assign each point to its nearest centroid
labels, _ = pairwise_distances_argmin_min(data, best_centroids)

# Print Results
print("\nBest Centroids found by GWO:")
print(best_centroids)
print(f"\nFinal Fitness (Intra-cluster variance): {best_score:.4f}")

# (Optional visualization)
try:
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=30)
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='black', marker='X', s=200)
    plt.title("Clustering using Grey Wolf Optimization")
    plt.show()
except ImportError:
    pass
