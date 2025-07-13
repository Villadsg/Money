import pandas as pd

file_path = '/home/villadsg/Documents/GitHub/Money/data/AVAV_features_2024-01-23_to_2025-06-06.parquet'
df = pd.read_parquet(file_path)
print(f'Features in {file_path}:')
print(list(df.columns))
