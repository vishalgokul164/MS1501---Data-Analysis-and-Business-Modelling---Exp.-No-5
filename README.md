# EXPERIMENT NO.5 : CLUSTER ANALYSIS FOR MARKET SEGMENTATION (K-MEANS ALGORITHM)

## AIM

To apply the K-Means Clustering algorithm to segment a dataset into distinct clusters and visualize how similar data points are grouped together for market segmentation.

## DATASET

The dataset consists of 16 manually entered customer data points representing two numerical features:
* **Feature 1 ($X$):** Annual Income (in Lakhs)
* **Feature 2 ($Y$):** Spending Score (`1–100`)

| Point | Annual Income ($X$) | Spending Score ($Y$) | Point | Annual Income ($X$) | Spending Score ($Y$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 20 | 30 | **9** | 32 | 33 |
| **2** | 22 | 35 | **10** | 24 | 31 |
| **3** | 25 | 40 | **11** | 28 | 45 |
| **4** | 70 | 80 | **12** | 75 | 70 |
| **5** | 75 | 85 | **13** | 71 | 83 |
| **6** | 80 | 90 | **14** | 83 | 89 |
| **7** | 30 | 70 | **15** | 28 | 60 |
| **8** | 35 | 75 | **16** | 27 | 67 |

## ALGORITHM / STEPS FOLLOWED

1. Import required Python libraries (`numpy`, `matplotlib`, `scikit-learn`).
2. Input data points as separate $X$ and $Y$ feature arrays.
3. Combine both variables into a single 2D feature matrix using `np.column_stack()`.
4. Initialize the `KMeans` model with $K = 3$ clusters.
5. Fit the model and assign each data point to a cluster using `fit_predict()`.
6. Retrieve cluster centroids (`cluster_centers_`) and assigned labels.
7. Visualize the clusters and their centroids using a Matplotlib scatter plot.
8. Interpret the resulting customer segments and centroid coordinates.

## FUNCTIONS AND METHODS USED

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `np.array()` | Creates arrays for storing numerical feature values ($X$ and $Y$) |
| | `np.column_stack()` | Combines multiple 1D arrays into a single 2D dataset for clustering |
| **Model Initialization** | `KMeans()` | Initializes the K-Means clustering model with $K$ clusters |
| **Model Training & Prediction** | `fit_predict()` | Fits the K-Means model to the dataset and assigns cluster labels |
| **Cluster Properties** | `cluster_centers_` | Retrieves the coordinates (centroids) of the cluster centers |
| | `labels_` | Stores the cluster labels assigned to each data point |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plots the data points and centroids in a 2D scatter plot |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`, `plt.show()` | Configures labels, legend, gridlines, and displays the final graph |

## PROGRAM / OUTPUT

import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
annual_income = np.array([20, 22, 25, 70, 75, 80, 30, 35, 32, 24, 28, 75, 71, 83, 28, 27])  
spending_score = np.array([30, 35, 40, 80, 85, 90, 70, 75, 33, 31, 45, 70, 83, 89, 60, 67])  
X = np.column_stack((annual_income, spending_score))  
print("--- 1. Feature Matrix Created Successfully ---")  
print(f"Dataset Shape (Samples, Features): {X.shape}")  
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  
y_kmeans = kmeans.fit_predict(X)  
centroids = kmeans.cluster_centers_  
print("\n--- 2. Model Parameters ---")  
print("Cluster Labels Assigned to Each Point:")  
print(y_kmeans)  
print("\nCluster Centroids (Coordinates):")  
for i, centroid in enumerate(centroids):  
    print(f"  Cluster {i+1} Centroid: Annual Income = {centroid[0]:.2f}, Spending Score = {centroid[1]:.2f}")  
plt.figure(figsize=(8, 6))  
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=80, c='red', label='Cluster 1 (High Income, High Spending)')  
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=80, c='blue', label='Cluster 2 (Low Income, Low Spending)')  
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=80, c='green', label='Cluster 3 (Moderate Income/Spending)')  
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='yellow', marker='*', edgecolor='black', label='Centroids')  
plt.title('K-Means Clustering: Customer Market Segmentation')  
plt.xlabel('Annual Income (in Lakhs)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend(loc='upper left', fontsize=9)  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.tight_layout()  
plt.show()  

<img width="536" height="177" alt="image" src="https://github.com/user-attachments/assets/a7d351c2-2a7b-4016-b91c-2ccde64de46e" />
<img width="732" height="545" alt="image" src="https://github.com/user-attachments/assets/331d932d-6d3f-4746-8f4a-6e6257a4249e" />


## Result

This segmentation provides actionable marketing targets, ranging from budget-conscious shoppers to premium consumers with high spending power.
