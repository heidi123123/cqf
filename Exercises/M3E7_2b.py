import numpy as np


# Exercise 2b, Gauss-Seidel method to solve linear equation system A*x = b with relaxation factor w = 1.25
A = np.array([[4, 3, 0, 0, 0], [3, 4, -1, 0, 0], [0, -1, 4, 0, 0], [0, 0, 0, 4, -1], [0, 0, 0, -1, 4]])
b = np.array([24, 30, -24, 10, 20])
w = 1.25


def gauss_seidel(a, x, b, w, num_iterations, print_iterations=False, tol=1e-5):
    n = len(a)

    # Performing Gauss-Seidel iterations
    for k in range(num_iterations):
        x_new = np.copy(x)  # Copy of x to check convergence by comparing x_new to the last iteration's x
        for j in range(n):
            # Calculate sum of a[j][i] * x_new[i] for all i != j
            temp_sum = np.dot(a[j, :], x_new) - a[j, j] * x_new[j]

            # Update x_new[j] with relaxation
            x_new[j] = (1 - w) * x_new[j] + (w / a[j, j]) * (b[j] - temp_sum)

        # Print x at each iteration if print_iterations is True
        if print_iterations:
            print(f"After {k + 1} iterations, x = {x_new}")

        # Check for convergence with L-infinity norm
        if np.linalg.norm(x_new - x, np.inf) < tol:
            print("Convergence achieved - no more iterations needed")
            break
        x = x_new

    return x


initial_guess = np.array([1, 1, 1, 1, 1])
x = gauss_seidel(A, initial_guess, b, w, 18, print_iterations=True)
