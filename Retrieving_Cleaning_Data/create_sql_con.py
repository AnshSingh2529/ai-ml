import sqlite3

# Sample data
data = [
    (1, 5.1, 3.5, 1.4, 0.2, 'setosa'),
    (2, 4.9, 3.0, 1.4, 0.2, 'setosa'),
    (3, 5.8, 2.7, 5.1, 1.9, 'virginica'),
    (4, 6.7, 3.1, 4.7, 1.5, 'versicolor'),
    (5, 5.0, 3.6, 1.4, 0.2, 'setosa'),
    (6, 6.3, 2.9, 5.6, 1.8, 'virginica'),
    (7, 5.7, 2.8, 4.1, 1.3, 'versicolor'),
    (8, 6.4, 2.8, 5.6, 2.2, 'virginica'),
    (9, 5.5, 2.3, 4.0, 1.3, 'versicolor'),
    (10, 5.4, 3.9, 1.7, 0.4, 'setosa')
]

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('data/species.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY,
    sepal_length REAL,
    sepal_width REAL,
    petal_length REAL,
    petal_width REAL,
    species TEXT
)
''')

# Insert data
cursor.executemany('INSERT INTO species VALUES (?, ?, ?, ?, ?, ?)', data)
conn.commit()
conn.close()