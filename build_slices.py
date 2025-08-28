# build_slices.py
import duckdb, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = RAW / "players_fbref_2000_2025.csv"
MATCHES_CSV = RAW / "Matches.csv"        # CFMD / dein Matches-File
ELO_CSV     = RAW / "EloRatings.csv"

con = duckdb.connect(str(ROOT / "warehouse.duckdb"))

# Windows-Pfade als POSIX-Strings
players_path = PLAYERS_CSV.as_posix()
matches_path = MATCHES_CSV.as_posix()
elo_path     = ELO_CSV.as_posix()

# Views mit expliziten CSV-Optionen.
# WICHTIG: all_varchar=true -> keine Typen-Gerate, alles als Text einlesen (stabil).
# ignore_errors=true + null_padding=true -> toleranter bei unregelmäßigen Zeilen
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

# --- Sanity: schneller Überblick ---
print("\n== Players: counts by league & season_end ==")
print(con.execute("""
SELECT league_slug, season_end, COUNT(*) AS rows
FROM players
GROUP BY 1,2
ORDER BY 1,2
LIMIT 20;
""").df())

# --- DEV-Slice: letzte zwei Saisons ---
con.execute(f"""
COPY (
  SELECT *
  FROM players
  WHERE CAST(season_end AS INTEGER) >= 2023
) TO '{(INTERIM/'players_dev_2seasons.csv').as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

# --- Train / Valid / Test Slices (Zeitfenster anpassen wie gewünscht) ---
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

# Beispiel-Slice: nur PL 2019–2020 für EDA
con.execute(f"""
COPY (
  SELECT *
  FROM players
  WHERE league_slug='Premier-League'
    AND CAST(season_end AS INTEGER) IN (2019, 2020)
) TO '{(INTERIM/'players_PL_2019_2020.csv').as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

print("\n✅ Slices geschrieben nach:", INTERIM.as_posix())
