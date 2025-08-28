# build_team_mapping.py
import duckdb, pathlib, re, unicodedata
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW  = DATA / "raw"
INTERIM = DATA / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

DB = ROOT / "warehouse.duckdb"
PLAYERS = INTERIM / "team_players_agg.csv"   # aus build_players_views.py
MATCHES = RAW / "Matches.csv"                 # CFMD matches mit home/away, date, score
OUT_AUTO = INTERIM / "team_name_map_auto.csv"
OUT_REVIEW = INTERIM / "team_name_map_review.csv"

def strip_accents(s: str) -> str:
    if s is None:
        return None
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# einfache Normalisierung in SQL nachbauen
NORMALIZE_SQL = r"""
lower(
  regexp_replace(
    regexp_replace(
      regexp_replace(
        regexp_replace(
          strip_accents(name),                -- Akzente weg
          '[^a-z0-9 ]', '', 'g'          -- Sonderzeichen raus
        ),
        '\b(fc|cf|ac|sc|ssc|athletic|atletico|club)\b', '', 'g'
      ),
      '\s+', ' ', 'g'
    ),
    '^\s+|\s+$', '', 'g'
  )
)
"""

con = duckdb.connect(str(DB))

# 1) Rohquellen als Views
con.execute(f"""
CREATE OR REPLACE VIEW v_fbref AS
SELECT DISTINCT team_hint AS team_name
FROM read_csv_auto('{PLAYERS.as_posix()}', header=true, all_varchar=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW v_cfdm AS
SELECT DISTINCT HomeTeam AS team_name FROM read_csv_auto('{MATCHES.as_posix()}', header=true, all_varchar=true)
UNION
SELECT DISTINCT AwayTeam FROM read_csv_auto('{MATCHES.as_posix()}', header=true, all_varchar=true);
""")

# 2) Normalisierte Namen
con.execute(f"""
CREATE OR REPLACE VIEW v_fbref_norm AS
SELECT team_name,
       {NORMALIZE_SQL.replace('name','team_name')} AS norm
FROM v_fbref;
""")
con.execute(f"""
CREATE OR REPLACE VIEW v_cfdm_norm AS
SELECT team_name,
       {NORMALIZE_SQL.replace('name','team_name')} AS norm
FROM v_cfdm;
""")

# 3) Auto-Matches (exakte Norm)
auto_df = con.execute("""
WITH c1 AS (SELECT DISTINCT team_name, norm FROM v_fbref_norm),
     c2 AS (SELECT DISTINCT team_name, norm FROM v_cfdm_norm)
SELECT c1.team_name AS fbref_team,
       c2.team_name AS cfdm_team,
       c1.norm AS norm
FROM c1 JOIN c2 USING(norm)
ORDER BY fbref_team, cfdm_team;
""").df()
auto_df.to_csv(OUT_AUTO, index=False)

# 4) Kandidaten zur Prüfung (fbref ohne Match oder mehrere Matches)
review_df = con.execute("""
WITH fb AS (SELECT DISTINCT team_name, norm FROM v_fbref_norm),
     cf AS (SELECT DISTINCT team_name, norm FROM v_cfdm_norm),
     auto AS (
        SELECT fb.team_name AS fbref_team, cf.team_name AS cfdm_team
        FROM fb JOIN cf USING (norm)
     )
SELECT fb.team_name AS fbref_team, fb.norm,
       LIST(cf.team_name) AS possible_cfdm_candidates
FROM fb
LEFT JOIN cf ON true
WHERE fb.team_name NOT IN (SELECT fbref_team FROM auto)
GROUP BY fb.team_name, fb.norm
ORDER BY fbref_team;
""").df()
review_df.to_csv(OUT_REVIEW, index=False)

print("✅ Auto-Mapping:", OUT_AUTO)
print("📝 Review-Liste (manuell prüfen/ergänzen):", OUT_REVIEW)
print("Tipp: Erzeuge danach eine endgültige Datei data/interim/team_name_map_final.csv mit Spalten:")
print("  fbref_team, cfdm_team [, valid_from, valid_to]")
