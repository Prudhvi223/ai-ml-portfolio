import pandas as pd

df = pd.read_csv("Day3/titanic.csv")
df.drop(columns=["Cabin"])


print(df.isnull().sum())
new_df = df.dropna()

print("Original Shape:", df.shape)
print("After dropna():", new_df.shape)
