from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

pca = PCA(n_components=2)

X_new = pca.fit_transform(X)

print(X.shape)
print(X_new.shape)

plt.scatter(
    X_new[:, 0],
    X_new[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")

plt.show()
