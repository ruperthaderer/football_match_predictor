# 9) build_match_map.py  (final: dedup + distinct-count + same-day priority, no PRAGMA)
import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
INTERIM = BASE / "data" / "interim"

players_enriched = INTERIM / "team_players_agg_enriched.csv"
matches_big5     = INTERIM / "matches_big5.csv"
teammap_final    = INTERIM / "team_name_map_final.csv"

out_map    = INTERIM / "match_map.csv"
out_review = INTERIM / "match_map_review.csv"

con = duckdb.connect()

# load sources + parse date
con.execute(f"""
CREATE OR REPLACE VIEW fbref_raw AS
SELECT
  match_url,
  home_team_fb,
  away_team_fb,
  try_strptime(match_date_fb, '%B %d, %Y')::DATE AS match_date_fb
FROM read_csv_auto('{players_enriched.as_posix()}', header=true, all_varchar=true)
WHERE home_team_fb IS NOT NULL AND away_team_fb IS NOT NULL;
""")

con.execute(f"""
CREATE OR REPLACE VIEW teammap AS
SELECT fbref_team, cfdm_team
FROM read_csv_auto('{teammap_final.as_posix()}', header=true, all_varchar=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW matches_raw AS
SELECT
  Division,
  try_cast(MatchDate AS DATE) AS MatchDate,
  HomeTeam,
  AwayTeam,
  coalesce(FTHome, '') AS FTHome,
  coalesce(FTAway, '') AS FTAway,
  concat_ws('|', Division, MatchDate, HomeTeam, AwayTeam) AS cfmd_match_id
FROM read_csv_auto('{matches_big5.as_posix()}', header=true, all_varchar=true)
WHERE Division IN ('E0','SP1','F1','D1','I1');
""")

# sanity checks
print("Rows fbref_raw:", con.execute("SELECT COUNT(*) FROM fbref_raw;").fetchone()[0])
print("Rows matches_raw:", con.execute("SELECT COUNT(*) FROM matches_raw;").fetchone()[0])
print("Rows teammap:", con.execute("SELECT COUNT(*) FROM teammap;").fetchone()[0])
print("FBref rows with non-null date:", con.execute("SELECT COUNT(*) FROM fbref_raw WHERE match_date_fb IS NOT NULL;").fetchone()[0])

# map team names + normalizing
con.execute("""
CREATE OR REPLACE VIEW fbref_mapped AS
SELECT
  f.match_url,
  f.match_date_fb,
  COALESCE(tm1.cfdm_team, f.home_team_fb) AS home_team_cf,
  COALESCE(tm2.cfdm_team, f.away_team_fb) AS away_team_cf,
  lower(trim(COALESCE(tm1.cfdm_team, f.home_team_fb))) AS home_norm,
  lower(trim(COALESCE(tm2.cfdm_team, f.away_team_fb))) AS away_norm
FROM fbref_raw f
LEFT JOIN teammap tm1 ON tm1.fbref_team = f.home_team_fb
LEFT JOIN teammap tm2 ON tm2.fbref_team = f.away_team_fb;
""")

con.execute("""
CREATE OR REPLACE VIEW matches_norm AS
SELECT
  *,
  lower(trim(HomeTeam)) AS home_norm,
  lower(trim(AwayTeam)) AS away_norm
FROM matches_raw;
""")

# candidates: same day (Priority 1) and ±1 day (Priority 2)
con.execute("""
CREATE OR REPLACE VIEW mmc_same_day AS
SELECT
  fb.match_url,
  fb.match_date_fb,
  fb.home_team_cf,
  fb.away_team_cf,
  m.cfmd_match_id,
  m.MatchDate AS match_date_cf,
  m.HomeTeam, m.AwayTeam, m.Division, m.FTHome, m.FTAway,
  0 AS days_diff,
  1 AS priority_flag
FROM fbref_mapped fb
JOIN matches_norm m
  ON m.home_norm = fb.home_norm
 AND m.away_norm = fb.away_norm
 AND m.MatchDate = fb.match_date_fb;
""")

con.execute("""
CREATE OR REPLACE VIEW mmc_plusminus1 AS
SELECT
  fb.match_url,
  fb.match_date_fb,
  fb.home_team_cf,
  fb.away_team_cf,
  m.cfmd_match_id,
  m.MatchDate AS match_date_cf,
  m.HomeTeam, m.AwayTeam, m.Division, m.FTHome, m.FTAway,
  abs(date_diff('day', m.MatchDate, fb.match_date_fb)) AS days_diff,
  2 AS priority_flag
FROM fbref_mapped fb
JOIN matches_norm m
  ON m.home_norm = fb.home_norm
 AND m.away_norm = fb.away_norm
 AND m.MatchDate <> fb.match_date_fb
 AND abs(date_diff('day', m.MatchDate, fb.match_date_fb)) <= 1;
""")

# Union + dedupe on (match_url, cfmd_match_id)
con.execute("""
CREATE OR REPLACE VIEW match_map_candidates AS
SELECT * FROM mmc_same_day
UNION ALL
SELECT * FROM mmc_plusminus1;
""")

con.execute("""
CREATE OR REPLACE VIEW mmc_dedup AS
SELECT
  match_url,
  any_value(match_date_fb) AS match_date_fb,
  any_value(home_team_cf) AS home_team_cf,
  any_value(away_team_cf) AS away_team_cf,
  cfmd_match_id,
  any_value(match_date_cf) AS match_date_cf,
  any_value(HomeTeam) AS HomeTeam,
  any_value(AwayTeam) AS AwayTeam,
  any_value(Division) AS Division,
  any_value(FTHome) AS FTHome,
  any_value(FTAway) AS FTAway,
  MIN(priority_flag) AS priority_flag,
  MIN(days_diff) AS days_diff
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY match_url, cfmd_match_id
      ORDER BY priority_flag ASC, days_diff ASC
    ) AS rn
  FROM match_map_candidates
)
WHERE rn = 1
GROUP BY match_url, cfmd_match_id;
""")

print("Candidates total (pre-dedup):", con.execute("SELECT COUNT(*) FROM match_map_candidates;").fetchone()[0])
print("Candidates after dedup:", con.execute("SELECT COUNT(*) FROM mmc_dedup;").fetchone()[0])

# final: games with one distinct cfmd_match_id
con.execute("""
CREATE OR REPLACE VIEW match_map AS
WITH cnt AS (
  SELECT match_url, COUNT(DISTINCT cfmd_match_id) AS k
  FROM mmc_dedup
  GROUP BY 1
)
SELECT d.*
FROM mmc_dedup d
JOIN cnt c USING (match_url)
WHERE c.k = 1;
""")

resolved = con.execute("SELECT COUNT(DISTINCT match_url) FROM match_map;").fetchone()[0]
unresolved = con.execute("""
WITH cnt AS (
  SELECT match_url, COUNT(DISTINCT cfmd_match_id) AS k
  FROM mmc_dedup
  GROUP BY 1
)
SELECT COUNT(*) FROM cnt WHERE k <> 1;
""").fetchone()[0]

print("Resolved (unique) matches:", resolved)
print("Unresolved (0 or >1 candidates):", unresolved)

# review file gets created if there are games w/o a match
if unresolved:
    review_df = con.execute("""
    WITH agg AS (
      SELECT
        match_url,
        any_value(match_date_fb) AS match_date_fb,
        any_value(home_team_cf) AS home_team_cf,
        any_value(away_team_cf) AS away_team_cf,
        LIST(DISTINCT cfmd_match_id ORDER BY priority_flag, days_diff) AS cfmd_candidates,
        MIN(priority_flag) AS best_priority
      FROM mmc_dedup
      GROUP BY match_url
    ),
    cnt AS (
      SELECT match_url, COUNT(DISTINCT cfmd_match_id) AS k
      FROM mmc_dedup
      GROUP BY 1
    )
    SELECT a.match_url, a.match_date_fb, a.home_team_cf, a.away_team_cf,
           c.k AS candidate_count, a.cfmd_candidates
    FROM agg a
    JOIN cnt c USING(match_url)
    WHERE c.k <> 1
    ORDER BY candidate_count DESC, best_priority ASC;
    """).df()
    review_df.to_csv(out_review, index=False)
    print(f"Review-Liste geschrieben: {out_review}")

# write to file
con.execute(f"COPY match_map TO '{out_map.as_posix()}' WITH (HEADER, DELIMITER ',');")
print(f"match_map.csv: {out_map}")
