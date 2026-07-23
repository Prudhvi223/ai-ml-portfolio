import pandas as pd

df = pd.read_csv("Day3/titanic.csv")

print(df.groupby("Sex")["Age"].mean())
