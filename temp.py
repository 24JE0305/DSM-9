import os
import pandas as pd

file_path = f"DSM-9/data_cache/ITC.NS.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} does not exist!")
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
print(df.head())
