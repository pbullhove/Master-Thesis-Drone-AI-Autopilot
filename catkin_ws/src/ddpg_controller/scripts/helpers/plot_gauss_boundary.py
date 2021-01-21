import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

amplitude = 1
mu = 0
variance = 1
sigma = math.sqrt(variance)
scale = np.sqrt(2 * np.pi) * sigma
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = amplitude * np.exp(-(x-mu)**2 / (2*sigma**2))
plt.plot([-4, -1, -1, 1, 1, 4], [0, 0, 1.0, 1.0, 0, 0], label="Boundary function")
plt.plot(x, y, label="Gaussian function")
plt.xlabel("Error")
plt.ylabel("Reward")
plt.legend(loc="best")
plt.show()