import pandas as pd

matches  = pd.read_parquet("data/matches.parquet")
shooting = pd.read_parquet("data/players_shooting.parquet")
passing  = pd.read_parquet("data/players_passing.parquet")
defense  = pd.read_parquet("data/players_defense.parquet")

print(matches.head())
print(shooting.head())
