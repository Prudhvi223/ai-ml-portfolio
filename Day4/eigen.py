import numpy as np

A = np.array([
    [2, 1],
    [1, 2]
])

values, vectors = np.linalg.eig(A)

print("Eigenvalues:")
print(values)

print()

print("Eigenvectors:")
print(vectors)
