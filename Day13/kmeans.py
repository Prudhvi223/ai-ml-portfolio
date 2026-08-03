from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
iris = load_iris()

X = iris.data
model = KMeans(
    n_clusters=3,
    random_state=42
)
model.fit(X)
print("Cluster Labels")

print(model.labels_)

print("\nCentroids")

print(model.cluster_centers_)
