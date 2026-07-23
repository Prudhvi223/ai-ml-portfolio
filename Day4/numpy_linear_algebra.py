import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("Addition:")
print(A + B)

print("\nSubtraction:")
print(A - B)

print("\nElement-wise Multiplication:")
print(A * B)

print("\nMatrix Multiplication:")
print(A @ B)

print("\nTranspose:")
print(A.T)

print("\nDeterminant:")
print(np.linalg.det(A))

print("\nInverse:")
print(np.linalg.inv(A))

print("\nDot Product:")
print(np.dot(np.array([1, 2]), np.array([3, 4])))
print(np.dot(A, B))
