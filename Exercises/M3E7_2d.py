import numpy as np


# Exercise 2d: check if B is strictly diagonally dominant
B = np.array([[7, 1, -2], [0, 2, 2], [1, 3, 6]])


def is_strictly_diagonally_dominant(matrix):
    # Get the absolute values of the matrix elements
    abs_matrix = np.abs(matrix)

    # Sum of non-diagonal elements in each row
    row_sum = np.sum(abs_matrix, axis=1) - np.diag(abs_matrix)

    # Check if each diagonal element is greater than the sum of non-diagonal elements in its row
    return np.all(np.diag(abs_matrix) > row_sum)


# Check if the matrix is strictly diagonally dominant
print(is_strictly_diagonally_dominant(B))
