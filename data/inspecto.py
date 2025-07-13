import pandas as pd

# Read the parquet file
df = pd.read_parquet('EH_features_2024-01-31_to_2025-06-14.parquet')

# Get unique values from a specific column
unique_values = df['earnings_classification'].unique()
print(unique_values)    