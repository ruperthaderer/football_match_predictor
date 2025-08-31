# test_streaks_one_team.py
# Prüft WinStreak / LosingStreak / NoWinStreak / NoLossStreak für EIN Team

from pathlib import Path
import duckdb

TEAM = "Man United"

DATA_PROCESSED = Path("../data/processed")
BASE_FEATS = DATA_PROCESSED / "features_base.csv"

con = duckdb.connect()

# 1) Base laden & casten
con.execute(f"""
CREATE OR REPLACE TABLE base AS
SELECT * FROM read_csv_auto('{BASE_FEATS.as_posix()}', header=true, all_varchar=true);
""")

con.execute("""
CREATE OR REPLACE VIEW base_cast AS
SELECT
  TRY_CAST(MatchDate AS DATE) AS match_date,
  -- Uhrzeit parsen (falls leer -> 00:00)
  COALESCE(TRY_STRPTIME(MatchTime, '%H:%M')::TIME, TIME '00:00') AS match_time_sort,
  Division,
  HomeTeam, AwayTeam,
  TRY_CAST(FTHome AS INTEGER) AS FTHome,
  TRY_CAST(FTAway AS INTEGER) AS FTAway
FROM base;
""")

# 2) Team-sicht (jede Partie einmal pro Team)
con.execute("""
CREATE OR REPLACE VIEW team_matches AS
SELECT
  match_date,
  match_time_sort,
  Division,
  HomeTeam AS team,
  AwayTeam AS opponent,
  FTHome AS goals_for,
  FTAway AS goals_against,
  CASE WHEN FTHome > FTAway THEN 1 ELSE 0 END AS win,
  CASE WHEN FTHome = FTAway THEN 1 ELSE 0 END AS draw,
  CASE WHEN FTHome < FTAway THEN 1 ELSE 0 END AS loss
FROM base_cast
UNION ALL
SELECT
  match_date,
  match_time_sort,
  Division,
  AwayTeam AS team,
  HomeTeam AS opponent,
  FTAway AS goals_for,
  FTHome AS goals_against,
  CASE WHEN FTAway > FTHome THEN 1 ELSE 0 END AS win,
  CASE WHEN FTAway = FTHome THEN 1 ELSE 0 END AS draw,
  CASE WHEN FTAway < FTHome THEN 1 ELSE 0 END AS loss
FROM base_cast;
""")

# 3) Streaks über Break-Gruppen:
#    - WinStreak: zählt aufwärts solange win=1; bricht bei draw/loss
#    - LosingStreak: zählt aufwärts solange loss=1; bricht bei draw/win
#    - NoWinStreak: zählt solange win=0; bricht bei win=1
#    - NoLossStreak: zählt solange loss=0; bricht bei loss=1
ORDER_BY = "PARTITION BY team ORDER BY match_date, match_time_sort, Division, opponent"

con.execute(f"""
WITH t AS (
  SELECT * FROM team_matches WHERE team = '{TEAM}'
),
g_win AS (
  SELECT
    t.*,
    SUM(CASE WHEN win=1 THEN 0 ELSE 1 END)
      OVER ({ORDER_BY} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp_win
  FROM t
),
g_loss AS (
  SELECT
    g_win.*,
    SUM(CASE WHEN loss=1 THEN 0 ELSE 1 END)
      OVER ({ORDER_BY} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp_loss
  FROM g_win
),
g_nowin AS (
  SELECT
    g_loss.*,
    SUM(CASE WHEN win=0 THEN 0 ELSE 1 END)
      OVER ({ORDER_BY} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp_nowin
  FROM g_loss
),
g_noloss AS (
  SELECT
    g_nowin.*,
    SUM(CASE WHEN loss=0 THEN 0 ELSE 1 END)
      OVER ({ORDER_BY} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp_noloss
  FROM g_nowin
),
s AS (
  SELECT
    *,
    CASE WHEN win=1
         THEN ROW_NUMBER() OVER (PARTITION BY team, grp_win   ORDER BY match_date, match_time_sort, Division, opponent)
         ELSE 0 END AS win_streak,
    CASE WHEN loss=1
         THEN ROW_NUMBER() OVER (PARTITION BY team, grp_loss  ORDER BY match_date, match_time_sort, Division, opponent)
         ELSE 0 END AS losing_streak,
    CASE WHEN win=0
         THEN ROW_NUMBER() OVER (PARTITION BY team, grp_nowin ORDER BY match_date DESC, match_time_sort DESC, Division DESC, opponent DESC)
         ELSE 0 END AS nowin_streak,
    CASE WHEN loss=0
         THEN ROW_NUMBER() OVER (PARTITION BY team, grp_noloss ORDER BY match_date DESC, match_time_sort DESC, Division DESC, opponent DESC)
         ELSE 0 END AS noloss_streak
  FROM g_noloss
)
SELECT
  match_date, team, opponent,
  goals_for, goals_against,
  CASE WHEN goals_for>goals_against THEN 'W'
       WHEN goals_for=goals_against THEN 'D' ELSE 'L' END AS result,
  COALESCE(LAG(win_streak, 1) OVER ({ORDER_BY}), 0) AS win_streak_in,
  COALESCE(LAG(losing_streak, 1) OVER ({ORDER_BY}), 0) AS losing_streak_in,
  COALESCE(LAG(nowin_streak, 1) OVER ({ORDER_BY}), 0) AS nowin_streak_in,
  COALESCE(LAG(noloss_streak, 1) OVER ({ORDER_BY}), 0) AS noloss_streak_in
FROM s
ORDER BY match_date DESC, match_time_sort DESC
LIMIT 20;
""")

df = con.fetch_df()
print(f"\nStreak-Check für: {TEAM}")
print(df)
