import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

dy = 2*x

plt.plot(x, dy)

plt.title("Derivative of y = x²")
plt.xlabel("x")
plt.ylabel("y")

plt.grid(True)

plt.show()
