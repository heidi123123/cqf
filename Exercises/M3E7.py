import numpy as np


# Exercise 2b, Gauss-Seidel method to solve linear equation system A*x = b with relaxation factor w = 1.25
A = np.array([[4, 3, 0, 0, 0], [3, 4, -1, 0, 0], [0, -1, 4, 0, 0], [0, 0, 0, 4, -1], [0, 0, 0, -1, 4]])
b = np.array([24, 30, -24, 10, 20])
w = 1.25


def gauss_seidel(a, x, b, w, num_iterations, print_iterations=False):
    # Finding length of a(n)
    n = len(a)

    # Performing Gauss-Seidel iterations
    for k in range(num_iterations):
        for j in range(n):
            # Using temporary copy of x to prevent simultaneous update
            temp_x = np.copy(x)

            # Calculate sum of a[j][i]*x[i] for all i != j
            temp_sum = np.dot(a[j, :], temp_x) - a[j, j] * temp_x[j]

            # Update x[j] with relaxation
            x[j] = (1 - w) * temp_x[j] + (w / a[j, j]) * (b[j] - temp_sum)

        # Print x at each iteration if print_iterations is True
        if print_iterations:
            print(f"After {k + 1} iterations, x = {x}")

    return x


initial_guess = np.array([1, 1, 1, 1, 1])
x = gauss_seidel(A, initial_guess, b, w, 3, print_iterations=True)
