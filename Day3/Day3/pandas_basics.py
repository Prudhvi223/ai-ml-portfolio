import pandas as pd

students = {
    "Name": ["Alice", "Bob"],
    "Age": [21, 22],
    "Marks": [85, 90]
}

df = pd.DataFrame(students)

print(df)
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nFirst 2 rows:")
print(df.head(2))

print("\nLast row:")
print(df.tail(1))

print("\nName column:")
print(df["Name"])

print("\nAge and Marks columns:")
print(df[["Age", "Marks"]])

print("\nFirst row using iloc:")
print(df.iloc[0])

print("\nSecond row using iloc:")
print(df.iloc[1])
