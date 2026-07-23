import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix:")
print(A)

print("\nTotal Sum:", A.sum())

print("\nColumn-wise Sum (axis=0):")
print(A.sum(axis=0))

print("\nRow-wise Sum (axis=1):")
print(A.sum(axis=1))

print("\nColumn-wise Mean:")
print(A.mean(axis=0))

print("\nRow-wise Mean:")
print(A.mean(axis=1))

print("\nMaximum in each Column:")
print(A.max(axis=0))

print("\nMaximum in each Row:")
print(A.max(axis=1))
