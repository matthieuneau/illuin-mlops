import pandas as pd

data = pd.read_csv("tiny-dataset.csv")
data.drop(["metadata", "prompt"], inplace=True, axis=1)

# print(data.head())

