import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

X = np.array([
    [1000],
    [1200],
    [1500],
    [1800],
    [2000],
    [2200],
    [2500],
    [3000]
])

y = np.array([
    20,
    25,
    30,
    36,
    40,
    45,
    50,
    60
])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[2400]])

print(prediction)

print(model.coef_)

print(model.intercept_)

plt.scatter(X, y, color="blue")

plt.plot(X, model.predict(X), color="red")

plt.xlabel("Area")

plt.ylabel("Price")

plt.title("House Price Prediction")

plt.show()
