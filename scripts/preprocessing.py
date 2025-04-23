import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

# Check if SQL file exists
sql_file_path = "sql/fetch_data.sql"
if not os.path.exists(sql_file_path):
    raise FileNotFoundError(f"SQL file not found at {sql_file_path}")

# Load SQL query from file
with open(sql_file_path, "r") as file:
    query = file.read()

# Execute query and load data into a DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Preprocessing
df.dropna(inplace=True)
df_encoded = pd.get_dummies(data=df, drop_first=True)

# Save preprocessed data
output_path = "data/preprocessed_data.csv"
df_encoded.to_csv(output_path, index=False)

# Logging
print("Data preprocessing is complete, saved to", output_path)
print("Shape after preprocessing:", df_encoded.shape)
print(df_encoded.head())