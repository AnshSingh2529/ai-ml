import pandas as pd

filepath = 'data/species.json'

data = pd.read_json(filepath)

data.to_json('data/outputjson.json')