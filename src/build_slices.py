# 1. build_slices.py
import duckdb, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = RAW / "players_fbref_2000_2025.csv"
MATCHES_CSV = RAW / "Matches.csv"
ELO_CSV     = RAW / "EloRatings.csv"

con = duckdb.connect(str(ROOT / "warehouse.duckdb"))

# converting / to \ to not cause problems in SQL
players_path = PLAYERS_CSV.as_posix()
matches_path = MATCHES_CSV.as_posix()
elo_path     = ELO_CSV.as_posix()

# csv -> VIEW (postgres cant do that)
# all_varchar=true -> read everything as string
# ignore_errors=true -> 'weird' columns do not hinder the view creation (postgres also cant do that)
# null_padding=true -> empty columns get filled with NULL
con.execute(f"""
CREATE OR REPLACE VIEW players AS
  SELECT * FROM read_csv_auto('{players_path}',
    delim=',',
    quote='"',
    escape='"',
    header=true,
    sample_size=-1,
    all_varchar=true,
    ignore_errors=true,
    null_padding=true
  );
""")

con.execute(f"""
CREATE OR REPLACE VIEW matches AS
  SELECT * FROM read_csv_auto('{matches_path}',
    delim=',',
    quote='"',
    escape='"',
    header=true,
    sample_size=-1,
    all_varchar=true,
    ignore_errors=true,
    null_padding=true
  );
""")

con.execute(f"""
CREATE OR REPLACE VIEW elo AS
  SELECT * FROM read_csv_auto('{elo_path}',
    delim=',',
    quote='"',
    escape='"',
    header=true,
    sample_size=-1,
    all_varchar=true,
    ignore_errors=true,
    null_padding=true
  );
""")

# sanity check
print("\n== Players: counts by league & season_end ==")
print(con.execute("""
SELECT league_slug, season_end, COUNT(*) AS rows
FROM players
GROUP BY 1,2
ORDER BY 1,2
LIMIT 20;
""").df())

# Train / Valid / Test Slices
con.execute(f"""
COPY (
  SELECT *
  FROM players
  WHERE CAST(season_end AS INTEGER) BETWEEN 2005 AND 2016
) TO '{(INTERIM/'players_train_2005_2016.csv').as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

con.execute(f"""
COPY (
  SELECT *
  FROM players
  WHERE CAST(season_end AS INTEGER) BETWEEN 2017 AND 2019
) TO '{(INTERIM/'players_valid_2017_2019.csv').as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

con.execute(f"""
COPY (
  SELECT *
  FROM players
  WHERE CAST(season_end AS INTEGER) BETWEEN 2020 AND 2025
) TO '{(INTERIM/'players_test_2020_2025.csv').as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

print("\n Slices geschrieben nach:", INTERIM.as_posix())
