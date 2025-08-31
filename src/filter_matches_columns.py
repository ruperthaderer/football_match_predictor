# 8) filter_matches_columns.py
from pathlib import Path
import duckdb

BASE = Path(__file__).resolve().parent
INTERIM = BASE / "data" / "interim"

inp = INTERIM / "matches_big5.csv"
out = INTERIM / "matches_core.csv"

# without complex betting stuff and match type clustering
keep_core = [
    "Division","MatchDate","MatchTime","HomeTeam","AwayTeam",
    "HomeElo","AwayElo","Form3Home","Form5Home","Form3Away","Form5Away",
    "FTHome","FTAway","FTResult","HTHome","HTAway","HTResult",
    "OddHome","OddDraw","OddAway","MaxHome","MaxDraw","MaxAway","HomeShots","AwayShots","HomeTarget",
    "AwayTarget","HomeFouls","AwayFouls","HomeCorners","AwayCorners","HomeYellow","AwayYellow","HomeRed","AwayRed",
]

con = duckdb.connect()

cols_df = con.execute(f"""
DESCRIBE SELECT * FROM read_csv_auto('{inp.as_posix()}', header=true, all_varchar=true);
""").df()

# DuckDB uses 'column_name' in DESCRIBE
if "column_name" in cols_df.columns:
    cols = cols_df["column_name"].tolist()
else:
    cols = cols_df["name"].tolist()  # Fallback

present_core = [c for c in keep_core if c in cols]
missing_core = [c for c in keep_core if c not in cols]

select_cols = present_core
if not select_cols:
    raise SystemExit("Keine der gewünschten Spalten gefunden. Bitte Input prüfen.")

print("Spalten übernommen:", ", ".join(select_cols))
if missing_core:
    print("Nicht gefunden (core):", ", ".join(missing_core))

con.execute(f"""
COPY (
  SELECT {", ".join(select_cols)}
  FROM read_csv_auto('{inp.as_posix()}', header=true, all_varchar=true)
) TO '{out.as_posix()}' (HEADER, DELIMITER ',');
""")

print(f"Geschrieben: {out}")
