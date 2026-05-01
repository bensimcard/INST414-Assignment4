import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# 1. DATA COLLECTION & PREPARATION
# ==========================================
url = "https://raw.githubusercontent.com/melaniewalsh/sample-social-network-datasets/master/sample-datasets/marvel/marvel-unimodal-edges.csv"
edges_df = pd.read_csv(url)
edges_df.columns = [col.lower() for col in edges_df.columns]

top_heroes = edges_df['source'].value_counts().head(50).index
filtered_edges = edges_df[edges_df['source'].isin(top_heroes) & edges_df['target'].isin(top_heroes)]

matrix = pd.crosstab(filtered_edges['source'], filtered_edges['target'])

# ==========================================
# 2. THE ELBOW METHOD (Finding k)
# ==========================================
wcss = []
k_range = range(1, 10)

for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(matrix)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', color='b')
plt.title('The Elbow Method for Marvel Hero Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (Cohesion)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("elbow_plot.png")
plt.show()

# ==========================================
# 3. FINAL CLUSTERING
# ==========================================
k = 3
model = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = model.fit_predict(matrix)

results = pd.DataFrame({'Hero': matrix.index, 'Cluster': clusters})

print("\n--- Marvel Hero Factions (K-Means Clustering) ---")
for i in range(k):
    cluster_members = results[results['Cluster'] == i]['Hero'].tolist()
    print(f"\nCluster {i}:")
    print(", ".join(cluster_members))