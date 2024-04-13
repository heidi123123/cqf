import numpy as np


# Exercise 2c, Doolittle's method using LU decomposition to solve linear equation system A*x = b
A = np.array([[2, 1, -2], [3, 2, 2], [5, 4, 3]])
b = np.array([10, 1, 4])


def get_doolittle_lu_decomp(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        for k in range(i, n):
            if i == k:
                L[i, i] = 1
            else:
                L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
    return L, U


def solve_lu_equation_system(L, U, b):
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward substitution
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    # Backward substitution
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return x


L, U = get_doolittle_lu_decomp(A)
print(f"L={L}\nU={U}")
print(L @ U)  # should be A again

x = solve_lu_equation_system(L, U, b)
print(x)
