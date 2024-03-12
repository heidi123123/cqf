import numpy as np
import matplotlib as plt

np.set_printoptions(precision=3)  # round all printed results to precision decimals max. 3

S = 100
r = 0.05
E = 100
T = 1

# ******************** question 3.1 ********************


# From Python Labs with Kannan, slightly adapted
# Create a user defined function
def binomial_option(spot: float, strike: float, rate: float, sigma: float, time: float, steps: int,
                    output: int = 0) -> np.ndarray:
    """
    binomial_option(spot, strike, rate, sigma, time, steps, output=0)
    Function for building binomial option tree for european call option payoff
    Params
    ------
    spot int or float - spot price
    strike int or float - strike price
    rate float - interest rate
    sigma float - volatility
    time int or float - expiration time
    steps int - number of trees
    output int - [0: price, 1: payoff, 2: option value, 3: option delta]
    Returns
    -------
    out: ndarray
    An array object of price, payoff, option value and delta specified by the output parameter
    """

    # params
    dt = time / steps
    u = 1 + sigma * np.sqrt(dt)
    v = 1 - sigma * np.sqrt(dt)
    p = 0.5 + rate * np.sqrt(dt) / (2 * sigma)
    df = 1 / (1 + rate * dt)

    # initialize arrays
    px = np.zeros((steps + 1, steps + 1))  # creates quadratic matrix of dimension steps+1 x steps+1 for the price
    cp = np.zeros((steps + 1, steps + 1))  # for call payoff
    V = np.zeros((steps + 1, steps + 1))  # for option price
    d = np.zeros((steps + 1, steps + 1))  # for delta
    # binomial loop
    # forward loop
    for j in range(steps + 1):
        for i in range(j + 1):
            px[i, j] = spot * np.power(v, i) * np.power(u, j - i)  # create asset path here
            cp[i, j] = np.maximum(px[i, j] - strike, 0)
    # reverse loop
    for j in range(steps + 1, 0, -1):
        for i in range(j):
            if j == steps + 1:  # the end of the tree (i.e. if reversed loop, technically the beginning)
                V[i, j - 1] = cp[i, j - 1]
                d[i, j - 1] = 0  # delta
            else:
                V[i, j - 1] = df * (p * V[i, j] + (1 - p) * V[i + 1, j])
                d[i, j - 1] = (V[i, j] - V[i + 1, j]) / (px[i, j] - px[i + 1, j])

    results = np.around(px, 2), np.around(cp, 2), np.around(V, 2), np.around(d, 4)
    return results[output]


# Option value is given by argument 2 as described in docstring
vols = np.linspace(0.05, 0.80, 100)
opt_values = []
# Calculate vector containing all option values for a given standard deviation
for vol in vols:
    sigma = np.sqrt(vol)
    opt_values.append(binomial_option(S, E, r, sigma, T, 1, 2)[0, 0])

# plot
plt.figure(figsize=(10, 8))
plt.scatter(x=vols, y=opt_values, marker="o", s=15)
plt.xlabel("Volatility")
plt.ylabel("Option Value")
plt.title("European call option value with T=1, S=K=100")
plt.show()
