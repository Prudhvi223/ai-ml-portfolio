import numpy as np
import matplotlib.pyplot as plt

marks = np.random.normal(70, 10, 1000)

plt.hist(marks, bins=30)

plt.title("Normal Distribution")
plt.xlabel("Marks")
plt.ylabel("Number of Students")

plt.show()
