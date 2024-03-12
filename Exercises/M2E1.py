import numpy as np


np.set_printoptions(precision=3)  # round all printed results to precision decimals max. 3

mu = np.transpose(np.array([[0.08, 0.10, 0.10, 0.14]]))
ones = np.ones((4, 1))

S = np.array([
    [0.12, 0., 0., 0.],
    [0., 0.12, 0., 0.],
    [0., 0., 0.15, 0.],
    [0., 0., 0., 0.20]
])


def get_covariance_matrix(rho):
    return np.linalg.multi_dot([rho, S, rho])


def get_inverse_covariance_matrix(covariance):
    try:
        return np.linalg.inv(covariance)
    except np.linalg.LinAlgError as exc:
        print(f"Covariance matrix not invertible: {exc}")


def get_a_b_c(covariance_inv):  # slide 68
    a = np.linalg.multi_dot([np.transpose(ones), covariance_inv, ones])
    b = np.linalg.multi_dot([np.transpose(mu), covariance_inv, ones])
    c = np.linalg.multi_dot([np.transpose(mu), covariance_inv, mu])
    return a.item(), b.item(), c.item()  # item converts the np.array elements back into scalars


def get_optimized_weights(rho, target_return):  # sl. 69
    covariance = get_covariance_matrix(rho)
    covariance_inv = get_inverse_covariance_matrix(covariance)
    a, b, c = get_a_b_c(covariance_inv)
    optimized_weights = np.dot(covariance_inv, (a * mu - b * ones) * target_return + c * ones - b * mu)
    return optimized_weights / sum(optimized_weights)  # normalize weights so they sum up to 1
