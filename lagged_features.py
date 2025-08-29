# lagged_features.py (ohne Streaks)
# Rolling Features + Momentum (pre-match) für alle Teams
# Output: data/processed/features_lagged.csv

from pathlib import Path
import duckdb

DATA_PROCESSED = Path("data/processed")
BASE_FEATS = DATA_PROCESSED / "features_base.csv"
OUT = DATA_PROCESSED / "features_lagged.csv"

con = duckdb.connect()

# 0) Basis laden
con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT * FROM read_csv_auto('{BASE_FEATS.as_posix()}', header=true, all_varchar=true);
""")

# Uhrzeit für stabile Sortierung (falls leer -> 00:00)
con.execute("""
CREATE OR REPLACE VIEW base_cast AS
SELECT
  TRY_CAST(MatchDate AS DATE) AS match_date,
  COALESCE(TRY_STRPTIME(MatchTime, '%H:%M')::TIME, TIME '00:00') AS match_time_sort,
  Division,
  HomeTeam, AwayTeam,
  TRY_CAST(FTHome AS INTEGER) AS FTHome,
  TRY_CAST(FTAway AS INTEGER) AS FTAway,
  TRY_CAST(HomeElo AS DOUBLE) AS HomeElo,
  TRY_CAST(AwayElo AS DOUBLE) AS AwayElo,
  *
FROM base;
""")

# 1) Team-Ansicht (jede Partie pro Team genau 1x)
con.execute("""
CREATE OR REPLACE VIEW team_matches AS
SELECT
  match_date, match_time_sort, Division,
  HomeTeam AS team, AwayTeam AS opponent,
  FTHome AS goals_for, FTAway AS goals_against,
  HomeElo AS elo,
  CASE WHEN FTHome > FTAway THEN 3
       WHEN FTHome = FTAway THEN 1 ELSE 0 END AS points,
  CASE WHEN FTHome > FTAway THEN 1 ELSE 0 END AS win,
  CASE WHEN FTHome = FTAway THEN 1 ELSE 0 END AS draw,
  CASE WHEN FTHome < FTAway THEN 1 ELSE 0 END AS loss
FROM base_cast
UNION ALL
SELECT
  match_date, match_time_sort, Division,
  AwayTeam AS team, HomeTeam AS opponent,
  FTAway AS goals_for, FTHome AS goals_against,
  AwayElo AS elo,
  CASE WHEN FTAway > FTHome THEN 3
       WHEN FTAway = FTHome THEN 1 ELSE 0 END AS points,
  CASE WHEN FTAway > FTHome THEN 1 ELSE 0 END AS win,
  CASE WHEN FTAway = FTHome THEN 1 ELSE 0 END AS draw,
  CASE WHEN FTAway < FTHome THEN 1 ELSE 0 END AS loss
FROM base_cast;
""")

# Einheitliche Sortierklausel für Windows
ORDER_BY = "PARTITION BY team ORDER BY match_date, match_time_sort, Division, opponent"

# 2) Lag-Basis: elo_change & rest_days
con.execute(f"""
CREATE OR REPLACE VIEW tm_lagbase AS
SELECT
  tm.*,
  (elo - LAG(elo,1) OVER ({ORDER_BY})) AS elo_change,
  (match_date - LAG(match_date,1) OVER ({ORDER_BY})) AS rest_days
FROM team_matches tm;
""")

# 3) Rolling-Fenster (nur vorherige Spiele)
con.execute(f"""
CREATE OR REPLACE VIEW team_rolling AS
SELECT
  t.*,

  -- last 3
  SUM(goals_for)      OVER w3 AS gf_last3,
  SUM(goals_against)  OVER w3 AS ga_last3,
  SUM(points)         OVER w3 AS pts_last3,
  AVG(elo_change)     OVER w3 AS elo_trend3,

  -- last 5
  SUM(goals_for)      OVER w5 AS gf_last5,
  SUM(goals_against)  OVER w5 AS ga_last5,
  SUM(points)         OVER w5 AS pts_last5,
  AVG(elo_change)     OVER w5 AS elo_trend5,

  -- FormMomentum: Punkte der letzten 3 minus Punkte der 3 davor
  (SUM(points) OVER w3 - SUM(points) OVER w6) AS form_momentum

FROM tm_lagbase t
WINDOW
  w3 AS ({ORDER_BY} ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
  w5 AS ({ORDER_BY} ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
  w6 AS ({ORDER_BY} ROWS BETWEEN 6 PRECEDING AND 4 PRECEDING);
""")

# 4) Join zurück auf Match-Zeilen (Home/Away) – ohne Streak-Spalten
con.execute("""
CREATE OR REPLACE VIEW lagged_per_match AS
SELECT
  b.*,

  -- Home
  sh.gf_last3         AS home_gf_last3,
  sh.ga_last3         AS home_ga_last3,
  sh.pts_last3        AS home_pts_last3,
  sh.elo_trend3       AS home_elo_trend3,
  sh.gf_last5         AS home_gf_last5,
  sh.ga_last5         AS home_ga_last5,
  sh.pts_last5        AS home_pts_last5,
  sh.elo_trend5       AS home_elo_trend5,
  sh.rest_days        AS home_rest_days,
  sh.form_momentum    AS home_form_momentum,

  -- Away
  sa.gf_last3         AS away_gf_last3,
  sa.ga_last3         AS away_ga_last3,
  sa.pts_last3        AS away_pts_last3,
  sa.elo_trend3       AS away_elo_trend3,
  sa.gf_last5         AS away_gf_last5,
  sa.ga_last5         AS away_ga_last5,
  sa.pts_last5        AS away_pts_last5,
  sa.elo_trend5       AS away_elo_trend5,
  sa.rest_days        AS away_rest_days,
  sa.form_momentum    AS away_form_momentum

FROM base_cast b
LEFT JOIN team_rolling sh ON sh.team = b.HomeTeam AND sh.match_date = b.match_date AND sh.match_time_sort = b.match_time_sort
LEFT JOIN team_rolling sa ON sa.team = b.AwayTeam AND sa.match_date = b.match_date AND sa.match_time_sort = b.match_time_sort;
""")

# 5) Export
con.execute(f"COPY lagged_per_match TO '{OUT.as_posix()}' (HEADER, DELIMITER ',');")
print(f"✅ Geschrieben: {OUT}")
