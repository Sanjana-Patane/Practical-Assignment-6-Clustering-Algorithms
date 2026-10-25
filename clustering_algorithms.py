
# Practical Assignment 6: Clustering Algorithms

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data

print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 4: Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(
    n_clusters=3, linkage="ward"
)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Step 5: Evaluate the clustering algorithms
models = {
    "K-Means": kmeans_labels,
    "Hierarchical": hierarchical_labels
}

for name, labels in models.items():
    print("\n", name)
    print("Silhouette Score:",
          round(silhouette_score(X_scaled, labels), 4))
    print("Davies-Bouldin Index:",
          round(davies_bouldin_score(X_scaled, labels), 4))

# Step 6: Display cluster graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=kmeans_labels)
axes[0].set_title("K-Means Clustering")
axes[0].set_xlabel("Standardized Sepal Length")
axes[0].set_ylabel("Standardized Sepal Width")

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=hierarchical_labels)
axes[1].set_title("Hierarchical Clustering")
axes[1].set_xlabel("Standardized Sepal Length")
axes[1].set_ylabel("Standardized Sepal Width")

plt.tight_layout()
plt.savefig("clustering_results.png", dpi=180)
plt.show()
