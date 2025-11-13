
"""
Try Making a New Database or any existing one to connect it to retrive data for model.
"""

from pymongo import MongoClient
import pandas as pd
# Creates a Mongo Connection
con = MongoClient()

# Choose Database (con.list_databse_names() will display available database)
db = con.database_name

# Create a cursor object using a query
cursor = db.collection_name.find(query)

# Expand cursor and contruct DataFrame
df = pd.DataFrame(list(cursor))

