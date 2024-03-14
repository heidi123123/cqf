import numpy as np

np.set_printoptions(precision=3)  # round all printed results to precision decimals max. 3

mu = np.array([[0.08, 0.10, 0.10, 0.14]]).T
ones = np.ones((4, 1))
std_dev = np.array([0.12, 0.12, 0.15, 0.20]).T
R = np.array([[1, 0.3, 0.3, 0.3],
              [0.3, 1, 0.6, 0.6],
              [0.3, 0.6, 1, 0.6],
              [0.3, 0.6, 0.6, 1]])


# ******************** question 1.1 ********************

def get_covariance_matrix(rho):
    return np.outer(std_dev, std_dev) * rho  # this is the same like std_dev * std_dev^T if std_dev is a column vector


def get_inverse_covariance_matrix(covariance):
    try:
        return np.linalg.inv(covariance)
    except np.linalg.LinAlgError as exc:
        print(f"Covariance matrix not invertible: {exc}")


def get_a_b_c(covariance_inv):  # slide 68
    a = np.linalg.multi_dot([ones.T, covariance_inv, ones])
    b = np.linalg.multi_dot([mu.T, covariance_inv, ones])
    c = np.linalg.multi_dot([mu.T, covariance_inv, mu])
    return a.item(), b.item(), c.item()  # item converts the np.array elements back into scalars


def get_optimized_weights(rho, target_return):  # sl. 69
    covariance = get_covariance_matrix(rho)
    covariance_inv = get_inverse_covariance_matrix(covariance)
    a, b, c = get_a_b_c(covariance_inv)
    optimized_weights = np.dot(covariance_inv, (a * mu - b * ones) * target_return + c * ones - b * mu)
    return optimized_weights / sum(optimized_weights)  # normalize weights so they sum up to 1


cov = get_covariance_matrix(R)
cov_inv = get_inverse_covariance_matrix(cov)
weights = get_optimized_weights(R, 0.045)
print(f"The weight allocation vector is\n{weights}.")
sigma_pf = np.sqrt(np.linalg.multi_dot([weights.T, cov, weights]).item())
print(f"The portfolio risk equals {round(sigma_pf, 3)}.")


# ******************** question 1.2 ********************

import matplotlib.pyplot as plt


def generate_random_weights(number_of_assets, sample_size):
    # generate a matrix with random weights of dimensions (number_of_assets) x (sample_size)
    random_weights = np.random.rand(number_of_assets, sample_size)
    # ensure the weights (column vectors of the matrix) all sum up to one by normalizing the last row of the matrix
    random_weights[(number_of_assets-1), :] = 1 - random_weights[:(number_of_assets-1), :].sum(axis=0)
    return random_weights


def get_pf_risk_and_return(covariance_matrix, weights, mu):
    pf_risk = np.sqrt(np.linalg.multi_dot([weights.T, covariance_matrix, weights]).item())
    pf_return = np.dot(weights.T, mu)
    return pf_risk, pf_return


def calculate_efficient_frontier(rho, covariance_matrix, target_returns, mu):
    pf_risks_ef = []
    pf_returns_ef = []
    for target_return in target_returns:
        optimized_weight = get_optimized_weights(rho, target_return)
        pf_risk_ef, pf_return_ef = get_pf_risk_and_return(covariance_matrix, optimized_weight, mu)
        pf_risks_ef.append(pf_risk_ef)
        pf_returns_ef.append(pf_return_ef)
    return pf_risks_ef, pf_returns_ef


# sample size
N = 700
number_of_assets = 4

# calculate necessary plot arguments
random_weights = generate_random_weights(number_of_assets, N)
pf_risks = np.zeros(N)
pf_returns = np.zeros(N)
sharpe_ratio = 0.
k_tangency = -1
global_min_risk = 1000.
k_global_min_risk = -1

# calculate portfolio risk and returns, tangency pf + GMVP
for k in range(N):
    weights = random_weights[:, k]
    pf_risks[k], pf_returns[k] = get_pf_risk_and_return(cov, weights, mu)
    ratio = pf_returns[k] / pf_risks[k]
    if ratio > sharpe_ratio:
        sharpe_ratio = ratio
        k_tangency = k
    if pf_risks[k] < global_min_risk:
        global_min_risk = pf_risks[k]
        k_global_min_risk = k

# calculate the efficient frontier
ef_number_of_points = 100
target_returns = np.linspace(pf_returns[k_global_min_risk], 0.2, ef_number_of_points)
pf_risks_ef, pf_returns_ef = calculate_efficient_frontier(R, cov, target_returns, mu)


# plot
plt.figure(figsize=(10, 8))
plt.plot(np.reshape(pf_risks_ef, (ef_number_of_points, )), np.reshape(pf_returns_ef, (ef_number_of_points, )),
color="green", label="Efficient Frontier", linewidth=3, zorder=6)
plt.scatter(x=pf_risks, y=pf_returns, marker="o", s=15, zorder=5)
plt.scatter(x=pf_risks[k_tangency], y=pf_returns[k_tangency],
            marker="*", color="red", s=300, label="Tangency Portfolio for R=0", zorder=10)
plt.scatter(x=pf_risks[k_global_min_risk], y=pf_returns[k_global_min_risk],
            marker="*", color="black", s=300, label="Global Minimum Variance Portfolio", zorder=9)
plt.xlabel("Portfolio risk sigma")
plt.ylabel("Portfolio returns mu")
plt.legend()
plt.show()
