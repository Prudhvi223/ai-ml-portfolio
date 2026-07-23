import pandas as pd

df = pd.read_csv("Day3/titanic.csv")

pivot = df.pivot_table(
    values="Fare",
    index="Pclass",
    aggfunc="max"
)

print(pivot)
