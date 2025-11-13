import sqlite3 as sq3
import pandas as pd

path = "data/species.db"

con = sq3.Connection(path)

query = """SELECT * FROM species"""
data = pd.read_sql(query, con)
print(data)
con.close()
