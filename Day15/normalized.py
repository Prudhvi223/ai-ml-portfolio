from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Salary": [30000, 50000, 70000, 90000, 110000]
})

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data)

print(scaled)
