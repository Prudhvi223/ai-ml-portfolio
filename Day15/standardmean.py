from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.DataFrame({
    "Age": [20, 25, 30],
    "Salary": [30000, 50000, 80000]
})

scaler = StandardScaler()

scaled = scaler.fit_transform(data)

print(scaled)
