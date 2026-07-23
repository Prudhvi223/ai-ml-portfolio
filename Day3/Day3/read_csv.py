import pandas as pd

# Read the Titanic dataset
df = pd.read_csv("Day3/titanic.csv")

print("First 5 Rows:")
print(df.head())
print("\nDataset Information:")
df.info()
