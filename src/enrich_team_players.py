# 3. enrich_team_players.py
import re
import duckdb
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "interim"

players_alt = DATA / "players_summary_alt.csv"
players_neu = DATA / "players_summary_neu.csv"
team_agg    = DATA / "team_players_agg.csv"
out_file    = DATA / "team_players_agg_enriched.csv"

con = duckdb.connect()

# load raw player data
con.execute(f"""
CREATE OR REPLACE VIEW players_raw AS
SELECT match_url, match_title
FROM read_csv_auto('{players_alt.as_posix()}', header=true, all_varchar=true)
UNION ALL
SELECT match_url, match_title
FROM read_csv_auto('{players_neu.as_posix()}', header=true, all_varchar=true);
""")

# one col per match_url
con.execute("""
CREATE OR REPLACE VIEW match_titles AS
SELECT
  match_url,
  any_value(match_title) AS match_title
FROM players_raw
GROUP BY match_url;
""")

# load team aggregates
con.execute(f"""
CREATE OR REPLACE VIEW team_agg AS
SELECT * FROM read_csv_auto('{team_agg.as_posix()}', header=true, all_varchar=true);
""")

# join
con.execute("""
CREATE OR REPLACE VIEW team_agg_enriched_raw AS
SELECT t.*, m.match_title
FROM team_agg t
LEFT JOIN match_titles m USING(match_url);
""")

# parse functions to extract team names
con.create_function(
    "parse_home",
    lambda s: re.split(r" vs\. | vs |–", s)[0] if s else None,
    return_type=str
)
con.create_function(
    "parse_away",
    lambda s: (
        re.split(r" vs\. | vs |–", s)[1].split(" Match Report")[0]
        if s and ((" vs" in s) or ("–" in s))
        else None
    ),
    return_type=str
)
con.create_function(
    "parse_date",
    lambda s: (m.group(1) if s and (m := re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", s)) else None),
    return_type=str
)

con.execute("""
CREATE OR REPLACE VIEW team_agg_enriched AS
SELECT
  *,
  parse_home(match_title) AS home_team_fb,
  parse_away(match_title) AS away_team_fb,
  parse_date(match_title) AS match_date_fb
FROM team_agg_enriched_raw;
""")

# (optional) check for duplicates
dups = con.execute("""
SELECT match_url, team_hint, COUNT(*) AS cnt
FROM team_agg_enriched
GROUP BY 1,2
HAVING COUNT(*) > 1
ORDER BY cnt DESC
LIMIT 5;
""").df()

if len(dups) > 0:
    print("Unerwartete Duplikate nach dem Join:\n", dups)

# write to file
con.execute(f"""
COPY team_agg_enriched TO '{out_file.as_posix()}' (HEADER, DELIMITER ',');
""")

print(f"Enriched file geschrieben (ohne Duplikate): {out_file}")
