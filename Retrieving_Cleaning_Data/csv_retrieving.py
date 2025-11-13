import pandas as pd

filepath = 'data/species.csv'

data = pd.read_csv(filepath)

print(data.iloc[:5])