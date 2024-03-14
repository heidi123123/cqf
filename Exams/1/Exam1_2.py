import numpy as np
from scipy.stats import norm


# Define all vector + matrices
mu = np.array([[0., 0., 0.]]).T
std_dev = np.array([0.3, 0.2, 0.15]).T
w = np.array([[0.5, 0.2, 0.3]]).T
rho = np.array([[1, 0.8, 0.5],
                [0.8, 1, 0.3],
                [0.5, 0.3, 1]])

# Calculating covariance
covariance = np.outer(std_dev, std_dev) * rho

# Confidence level
alpha = 0.99
factor = norm.ppf(1 - alpha)

# Calculating VaR
var = np.dot(w.T, mu) + np.dot(np.dot(factor, covariance), w) / np.sqrt(np.dot(np.dot(w.T, covariance), w))

# Calculating ES
es = np.dot(w.T, mu) - np.dot(np.dot(norm.pdf(factor), covariance), w) / (np.dot((1 - alpha), np.sqrt(np.dot(np.dot(w.T, covariance), w))))

# Print results
for idx in range(len(var)):
    for col in range(len(var[idx])):
        print(f"dVaR(w)/dw{idx + 1} = {np.round(var[idx][col] * 100, 2)}%")
        print(f"dES(w)/dw{idx + 1} = {np.round(es[idx][col] * 100, 2)}%")
        print("")
