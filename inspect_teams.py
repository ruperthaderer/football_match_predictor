# inspect_teams.py
import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
INTERIM = DATA / "interim"
RAW = DATA / "raw"

# Dateien (ggf. anpassen)
players_enriched = INTERIM / "team_players_agg_enriched.csv"  # aus enrich_team_players.py
matches_csv = INTERIM / "matches_big5.csv"                             # CFMD: enthält HomeTeam, AwayTeam

con = duckdb.connect()

# 1) Quellen laden
con.execute(f"""
CREATE OR REPLACE VIEW fbref_enriched AS
SELECT * FROM read_csv_auto('{players_enriched.as_posix()}', header=true, all_varchar=true);
""")
con.execute(f"""
CREATE OR REPLACE VIEW matches_raw AS
SELECT * FROM read_csv_auto('{matches_csv.as_posix()}', header=true, all_varchar=true);
""")

# 2) Distinct Teamlisten
con.execute("""
CREATE OR REPLACE VIEW cfdm_teams AS
SELECT DISTINCT HomeTeam AS team_name FROM matches_raw
UNION
SELECT DISTINCT AwayTeam FROM matches_raw;
""")
con.execute("""
CREATE OR REPLACE VIEW fbref_teams AS
SELECT DISTINCT home_team_fb AS team_name FROM fbref_enriched WHERE home_team_fb IS NOT NULL
UNION
SELECT DISTINCT away_team_fb FROM fbref_enriched WHERE away_team_fb IS NOT NULL;
""")

print("== Counts ==")
print(con.execute("""
SELECT 'fbref' AS src, COUNT(*) AS n FROM fbref_teams
UNION ALL
SELECT 'cfdm' , COUNT(*) FROM cfdm_teams;
""").df())

print("\n== Beispiele (FBref) ==")
print(con.execute("SELECT team_name FROM fbref_teams ORDER BY 1 LIMIT 25;").df())
print("\n== Beispiele (CFMD) ==")
print(con.execute("SELECT team_name FROM cfdm_teams ORDER BY 1 LIMIT 25;").df())

# 3) Normalisierung (ohne Akzentlogik – laut dir nicht nötig in CFMD)
NORMALIZE_SQL = """
lower(
  regexp_replace(
    regexp_replace(
      regexp_replace(team_name, '[^a-zA-Z0-9 ]', '', 'g'),
      '\\b(fc|cf|ac|sc|ssc|athletic|atletico|club)\\b', '', 'g'
    ),
    '\\s+', ' ', 'g'
  )
)
"""

con.execute(f"""
CREATE OR REPLACE VIEW fbref_norm AS
SELECT DISTINCT
  team_name,
  {NORMALIZE_SQL} AS norm
FROM fbref_teams;
""")
con.execute(f"""
CREATE OR REPLACE VIEW cfdm_norm AS
SELECT DISTINCT
  team_name,
  {NORMALIZE_SQL} AS norm
FROM cfdm_teams;
""")

# 4) Auto-Mapping (exakte Norm-Übereinstimmung)
con.execute("""
CREATE OR REPLACE VIEW team_auto_map AS
SELECT f.team_name AS fbref_team,
       c.team_name AS cfdm_team,
       f.norm
FROM fbref_norm f
JOIN cfdm_norm c USING(norm)
ORDER BY fbref_team, cfdm_team;
""")

print("\n== Matches (fbref/cfdm/auto_map counts) ==")
print(con.execute("""
SELECT 'fbref' AS src, COUNT(DISTINCT team_name) AS n FROM fbref_norm
UNION ALL
SELECT 'cfdm' , COUNT(DISTINCT team_name) FROM cfdm_norm
UNION ALL
SELECT 'auto_map', COUNT(*) FROM team_auto_map;
""").df())

print("\n== Auto matches (sample) ==")
print(con.execute("SELECT * FROM team_auto_map LIMIT 30;").df())

# 5) Exporte
out_auto = INTERIM / "team_name_map_auto.csv"
out_review = INTERIM / "team_name_map_review.csv"

con.execute(f"COPY team_auto_map TO '{out_auto.as_posix()}' WITH (HEADER, DELIMITER ',');")

# FBref-Teams ohne Auto-Match + Kandidatenliste (alle CFMD-Varianten zur Auswahl)
con.execute(f"""
COPY (
  SELECT f.team_name AS fbref_team, f.norm,
         LIST(c.team_name) AS possible_cfdm_candidates
  FROM fbref_norm f
  LEFT JOIN cfdm_norm c ON TRUE
  WHERE f.team_name NOT IN (SELECT fbref_team FROM team_auto_map)
  GROUP BY f.team_name, f.norm
  ORDER BY fbref_team
) TO '{out_review.as_posix()}' WITH (HEADER, DELIMITER ',');
""")

print(f"\n✅ Exportiert:\n  {out_auto}\n  {out_review}")
