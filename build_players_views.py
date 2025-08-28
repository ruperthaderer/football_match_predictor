# build_players_views.py
# Liest players_summary_alt/neu, harmonisiert Spalten und erzeugt Team-Aggregate je Match.

import duckdb
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
INTERIM = DATA / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

CSV_ALT = INTERIM / "players_summary_alt.csv"
CSV_NEU = INTERIM / "players_summary_neu.csv"
OUT_CSV = INTERIM / "team_players_agg.csv"
DB_PATH = ROOT / "warehouse.duckdb"

# DuckDB-Verbindung
con = duckdb.connect(str(DB_PATH))

# 1) Views registrieren (mit Header = true, die Split-Dateien haben Header)
con.execute(f"""
CREATE OR REPLACE VIEW summary_alt AS
SELECT *
FROM read_csv_auto('{CSV_ALT.as_posix()}',
                   header=true, delim=',', quote='"', escape='"',
                   sample_size=-1, all_varchar=true, ignore_errors=true, null_padding=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW summary_neu AS
SELECT *
FROM read_csv_auto('{CSV_NEU.as_posix()}',
                   header=true, delim=',', quote='"', escape='"',
                   sample_size=-1, all_varchar=true, ignore_errors=true, null_padding=true);
""")

# kurze Übersicht, hilft beim Anpassen der cNN-Indizes, falls nötig
print("== summary_alt (erste 3 Spaltennamen) ==")
print(con.execute("PRAGMA table_info('summary_alt');").df().head(5))
print("== summary_neu (erste 3 Spaltennamen) ==")
print(con.execute("PRAGMA table_info('summary_neu');").df().head(5))

# 2) Harmonisierung ALT -> exakt gleiche Spaltenreihenfolge wie NEU
con.execute("""
CREATE OR REPLACE VIEW players_alt_harmonized AS
SELECT
  -- ===== Kanonische Reihenfolge =====
  c01                           AS player,
  c03                           AS nation,
  c04                           AS pos,
  c05                           AS age,
  TRY_CAST(c06 AS INTEGER)      AS min,
  TRY_CAST(c07 AS INTEGER)      AS gls,
  TRY_CAST(c08 AS INTEGER)      AS ast,
  TRY_CAST(c09 AS INTEGER)      AS pk,
  TRY_CAST(c10 AS INTEGER)      AS pkatt,
  TRY_CAST(c11 AS INTEGER)      AS sh,
  TRY_CAST(c12 AS INTEGER)      AS sot,
  TRY_CAST(c13 AS INTEGER)      AS crdy,
  TRY_CAST(c14 AS INTEGER)      AS crdr,
  TRY_CAST(c15 AS INTEGER)      AS fls,
  TRY_CAST(c16 AS INTEGER)      AS fld,
  TRY_CAST(c17 AS INTEGER)      AS off,
  TRY_CAST(c18 AS INTEGER)      AS crs,
  TRY_CAST(c19 AS INTEGER)      AS tklw,
  CAST(NULL AS INTEGER)         AS tkl,         -- alt hat "Tackles won", nicht "Tackles"
  TRY_CAST(c20 AS INTEGER)      AS intr,
  CAST(NULL AS INTEGER)         AS blocks,
  TRY_CAST(c21 AS INTEGER)      AS og,
  TRY_CAST(c22 AS INTEGER)      AS pkwon,
  TRY_CAST(c23 AS INTEGER)      AS pkcon,
  CAST(NULL AS INTEGER)         AS touches,
  CAST(NULL AS DOUBLE)          AS xg,
  CAST(NULL AS DOUBLE)          AS npxg,
  CAST(NULL AS DOUBLE)          AS xag,
  CAST(NULL AS INTEGER)         AS sca,
  CAST(NULL AS INTEGER)         AS gca,
  CAST(NULL AS INTEGER)         AS pass_cmp,
  CAST(NULL AS INTEGER)         AS pass_att,
  CAST(NULL AS DOUBLE)          AS pass_cmp_pct,
  CAST(NULL AS INTEGER)         AS pass_prg,
  CAST(NULL AS INTEGER)         AS carries,
  CAST(NULL AS INTEGER)         AS carries_prg,
  CAST(NULL AS INTEGER)         AS take_ons_att,
  CAST(NULL AS INTEGER)         AS take_ons_succ,
  -- Meta / Schlüssel
  match_url, league, season, table_type, team_hint, match_title, league_slug, season_end
FROM summary_alt
WHERE LOWER(table_type) = 'summary'
  AND COALESCE(LOWER(c01),'') NOT LIKE '%players%';
""")

# 3) Harmonisierung NEU -> gleiche Reihenfolge/Anzahl
con.execute("""
CREATE OR REPLACE VIEW players_neu_harmonized AS
SELECT
  -- ===== Kanonische Reihenfolge =====
  c01                           AS player,
  c03                           AS nation,
  c04                           AS pos,
  c05                           AS age,
  TRY_CAST(c06 AS INTEGER)      AS min,
  TRY_CAST(c07 AS INTEGER)      AS gls,
  TRY_CAST(c08 AS INTEGER)      AS ast,
  TRY_CAST(c09 AS INTEGER)      AS pk,
  TRY_CAST(c10 AS INTEGER)      AS pkatt,
  TRY_CAST(c11 AS INTEGER)      AS sh,
  TRY_CAST(c12 AS INTEGER)      AS sot,
  TRY_CAST(c13 AS INTEGER)      AS crdy,
  TRY_CAST(c14 AS INTEGER)      AS crdr,
  CAST(NULL AS INTEGER)         AS fls,
  CAST(NULL AS INTEGER)         AS fld,
  CAST(NULL AS INTEGER)         AS off,
  CAST(NULL AS INTEGER)         AS crs,
  CAST(NULL AS INTEGER)         AS tklw,
  TRY_CAST(c16 AS INTEGER)      AS tkl,
  TRY_CAST(c17 AS INTEGER)      AS intr,
  TRY_CAST(c18 AS INTEGER)      AS blocks,
  CAST(NULL AS INTEGER)         AS og,
  CAST(NULL AS INTEGER)         AS pkwon,
  CAST(NULL AS INTEGER)         AS pkcon,
  TRY_CAST(c15 AS INTEGER)      AS touches,
  TRY_CAST(c19 AS DOUBLE)       AS xg,
  TRY_CAST(c20 AS DOUBLE)       AS npxg,
  TRY_CAST(c21 AS DOUBLE)       AS xag,
  TRY_CAST(c22 AS INTEGER)      AS sca,
  TRY_CAST(c23 AS INTEGER)      AS gca,
  TRY_CAST(c24 AS INTEGER)      AS pass_cmp,
  TRY_CAST(c25 AS INTEGER)      AS pass_att,
  TRY_CAST(REPLACE(c26,'%','') AS DOUBLE)  AS pass_cmp_pct,
  TRY_CAST(c27 AS INTEGER)      AS pass_prg,
  TRY_CAST(c28 AS INTEGER)      AS carries,
  TRY_CAST(c29 AS INTEGER)      AS carries_prg,
  TRY_CAST(c30 AS INTEGER)      AS take_ons_att,
  TRY_CAST(c31 AS INTEGER)      AS take_ons_succ,
  -- Meta / Schlüssel
  match_url, league, season, table_type, team_hint, match_title, league_slug, season_end
FROM summary_neu
WHERE LOWER(table_type) = 'summary'
  AND COALESCE(LOWER(c01),'') NOT LIKE '%players%';
""")

# 4) Union (jetzt gleiche Spaltenzahl/-reihenfolge)
con.execute("""
CREATE OR REPLACE VIEW players_unified AS
SELECT * FROM players_alt_harmonized
UNION ALL
SELECT * FROM players_neu_harmonized;
""")


# 5) Team-Aggregate je Match (Summe + Per-90, Basis)
con.execute("""
CREATE OR REPLACE VIEW team_players_agg AS
WITH base AS (
  SELECT
    match_url, league_slug, season_end, team_hint,
    SUM(min)                  AS min_sum,
    SUM(gls)                  AS gls_sum,
    SUM(ast)                  AS ast_sum,
    SUM(pk)                   AS pk_sum,
    SUM(pkatt)                AS pkatt_sum,
    SUM(sh)                   AS sh_sum,
    SUM(sot)                  AS sot_sum,
    SUM(crdy)                 AS yel_sum,
    SUM(crdr)                 AS red_sum,
    SUM(fls)                  AS fls_sum,
    SUM(fld)                  AS fld_sum,
    SUM(off)                  AS off_sum,
    SUM(crs)                  AS crs_sum,
    SUM(tklw)                 AS tklw_sum,
    SUM(tkl)                  AS tkl_sum,
    SUM(intr)                 AS int_sum,
    SUM(blocks)               AS blk_sum,
    SUM(xg)                   AS xg_sum,
    SUM(npxg)                 AS npxg_sum,
    SUM(xag)                  AS xag_sum,
    SUM(sca)                  AS sca_sum,
    SUM(gca)                  AS gca_sum,
    SUM(pass_cmp)             AS pass_cmp_sum,
    SUM(pass_att)             AS pass_att_sum,
    AVG(pass_cmp_pct)         AS pass_cmp_pct_avg,
    SUM(pass_prg)             AS pass_prg_sum,
    SUM(carries)              AS carries_sum,
    SUM(carries_prg)          AS carries_prg_sum,
    SUM(take_ons_att)         AS take_att_sum,
    SUM(take_ons_succ)        AS take_succ_sum
  FROM players_unified
  GROUP BY 1,2,3,4
)
SELECT
  *,
  CASE WHEN min_sum>0 THEN gls_sum * 90.0 / min_sum END  AS gls_p90,
  CASE WHEN min_sum>0 THEN ast_sum * 90.0 / min_sum END  AS ast_p90,
  CASE WHEN min_sum>0 THEN sh_sum  * 90.0 / min_sum END  AS sh_p90,
  CASE WHEN min_sum>0 THEN sot_sum * 90.0 / min_sum END  AS sot_p90,
  CASE WHEN min_sum>0 THEN xg_sum  * 90.0 / min_sum END  AS xg_p90,
  CASE WHEN min_sum>0 THEN sca_sum * 90.0 / min_sum END  AS sca_p90,
  CASE WHEN min_sum>0 THEN pass_prg_sum * 90.0 / min_sum END AS pass_prg_p90,
  CASE WHEN take_att_sum>0 THEN take_succ_sum * 1.0 / take_att_sum END AS take_on_succ_rate
FROM base;
""")

# 6) Export nach CSV
con.execute(f"""
COPY team_players_agg
TO '{OUT_CSV.as_posix()}'
WITH (HEADER, DELIMITER ',');
""")

# Kurzer Erfolgshinweis
rows = con.execute("SELECT COUNT(*) FROM team_players_agg;").fetchone()[0]
print(f"✅ team_players_agg.csv geschrieben: {OUT_CSV}  (Zeilen: {rows:,})")
print(f"ℹ️  DuckDB-DB mit Views liegt unter: {DB_PATH}")

print(con.execute("""
    SELECT league, season, COUNT(*) AS rows
    FROM players_unified
    GROUP BY league, season
    ORDER BY season, league
    LIMIT 20
""").df())

