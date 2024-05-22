import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_sigmoid():
    z = np.linspace(-10, 10, 100)
    p = sigmoid(z)
    plt.figure()
    plt.plot(z, p)
    plt.title("Sigmoid function $\sigma(z)$")
    plt.xlabel("z")
    plt.ylabel("p")
    plt.show()


if __name__ == "__main__":
    plot_sigmoid()
