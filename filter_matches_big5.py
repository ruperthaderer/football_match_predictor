# filter_matches_big5.py
import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
INTERIM = BASE / "data" / "interim"

RAW.mkdir(parents=True, exist_ok=True)
INTERIM.mkdir(parents=True, exist_ok=True)

matches_csv = RAW / "matches.csv"
out_csv = INTERIM / "matches_big5.csv"

con = duckdb.connect()

# Alle Zeilen mit den 5 Div-Codes durchlassen
con.execute(f"""
COPY (
    SELECT *
    FROM read_csv_auto('{matches_csv.as_posix()}', header=true, all_varchar=true)
    WHERE Division IN ('E0','SP1','F1','D1','I1')
) TO '{out_csv.as_posix()}' WITH (HEADER, DELIMITER ',');
""")

print(f"✅ Gefilterte Datei erstellt: {out_csv}")
