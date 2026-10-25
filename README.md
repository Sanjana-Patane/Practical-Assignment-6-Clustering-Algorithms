
# Practical Assignment 6: Clustering Algorithms

## 1. Aim
To implement clustering algorithms on a dataset and evaluate the quality of clustering results.

## 2. Dataset
The Iris dataset is used in this practical. It contains 150 samples and four numerical features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The dataset is available directly through the scikit-learn library.

## 3. Algorithms Used
1. K-Means Clustering
2. Hierarchical Clustering

## 4. Libraries Used
- Python
- scikit-learn
- Matplotlib

## 5. Methodology
1. Load the Iris dataset.
2. Standardize the numerical features.
3. Apply K-Means clustering with three clusters.
4. Apply Agglomerative Hierarchical Clustering with three clusters.
5. Evaluate the results using Silhouette Score and Davies-Bouldin Index.
6. Visualize the clusters using scatter plots.

## 6. Evaluation Metrics
- Silhouette Score: Higher values generally indicate better-separated clusters.
- Davies-Bouldin Index: Lower values generally indicate better clustering.

## 7. How to Run
1. Install the required libraries.
2. Run clustering_algorithms.py.
3. Observe the evaluation metrics in the output.
4. View the generated clustering_results.png graph.

## 8. Conclusion
K-Means and Hierarchical Clustering were implemented on the Iris dataset. Their clustering quality was compared using evaluation metrics and visualizations.
