from pathlib import Path
import duckdb

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Eingangsdatei auswählen (bevorzugt "dicke" Variante)
candidates = [
    INTERIM / "matches_core.csv",
    INTERIM / "matches_big5.csv",
    INTERIM / "matches_slim.csv",
]
inp = next((p for p in candidates if p.exists()), None)
if not inp:
    raise SystemExit("❌ Keine Eingabedatei gefunden. Erwartet eine der Dateien: "
                     f"{', '.join([p.as_posix() for p in candidates])}")

out_csv = PROCESSED / "features_base.csv"
print(f"🔧 Eingabe: {inp}")

con = duckdb.connect()

# Spalten der CSV ermitteln
cols_df = con.execute(f"""
DESCRIBE SELECT * FROM read_csv_auto('{inp.as_posix()}', header=true, all_varchar=true);
""").df()
cols = (cols_df["column_name"].tolist()
        if "column_name" in cols_df.columns else cols_df["name"].tolist())
colset = set(cols)

def has(*names): return all(n in colset for n in names)

# Pflichtfelder prüfen (für IDs & Label)
required_min = ["Division","MatchDate","HomeTeam","AwayTeam"]
missing_req = [c for c in required_min if c not in colset]
if missing_req:
    raise SystemExit(f"❌ Pflichtspalten fehlen: {missing_req}")

# SELECT-Liste dynamisch aufbauen
S = []

# Basis-IDs
S += [
    "Division",
    "MatchDate",
    "COALESCE(MatchTime,'') AS MatchTime",
    "HomeTeam",
    "AwayTeam",
    # für konsistente Sortierung im Training:
    "try_cast(MatchDate AS DATE) AS match_date",
    "concat_ws('|', Division, MatchDate, HomeTeam, AwayTeam) AS cfmd_match_id",
]

# Zielvariable (aus FT-Ergebnis)
if has("FTHome","FTAway"):
    S += [
        "try_cast(FTHome AS INTEGER) AS FTHome",
        "try_cast(FTAway AS INTEGER) AS FTAway",
        """CASE
             WHEN try_cast(FTHome AS INTEGER) > try_cast(FTAway AS INTEGER) THEN 'H'
             WHEN try_cast(FTHome AS INTEGER) < try_cast(FTAway AS INTEGER) THEN 'A'
             ELSE 'D'
           END AS y_outcome""",
        """CASE
             WHEN try_cast(FTHome AS INTEGER) > try_cast(FTAway AS INTEGER) THEN 1
             ELSE 0
           END AS y_home_win""",
    ]
elif "FTResult" in colset:
    S += [
        "COALESCE(FTResult,'') AS FTResult",
        "NULL::INTEGER AS FTHome",
        "NULL::INTEGER AS FTAway",
        "FTResult AS y_outcome",
        "CASE WHEN FTResult='H' THEN 1 ELSE 0 END AS y_home_win",
    ]
else:
    S += ["NULL AS y_outcome", "NULL::INTEGER AS y_home_win"]

# Halbzeit (falls vorhanden)
if has("HTHome","HTAway","HTResult"):
    S += [
        "try_cast(HTHome AS INTEGER) AS HTHome",
        "try_cast(HTAway AS INTEGER) AS HTAway",
        "COALESCE(HTResult,'') AS HTResult",
    ]

# ELO & abgeleitete ELO-Features
if has("HomeElo","AwayElo"):
    S += [
        "try_cast(HomeElo AS DOUBLE) AS HomeElo",
        "try_cast(AwayElo AS DOUBLE) AS AwayElo",
        "(try_cast(HomeElo AS DOUBLE) - try_cast(AwayElo AS DOUBLE)) AS elo_diff",
        "(try_cast(HomeElo AS DOUBLE) / NULLIF(try_cast(HomeElo AS DOUBLE) + try_cast(AwayElo AS DOUBLE),0)) AS elo_ratio",
    ]

# Form & Differenzen
if has("Form3Home","Form3Away"):
    S += [
        "try_cast(Form3Home AS DOUBLE) AS Form3Home",
        "try_cast(Form3Away AS DOUBLE) AS Form3Away",
        "(try_cast(Form3Home AS DOUBLE) - try_cast(Form3Away AS DOUBLE)) AS form3_diff",
    ]
if has("Form5Home","Form5Away"):
    S += [
        "try_cast(Form5Home AS DOUBLE) AS Form5Home",
        "try_cast(Form5Away AS DOUBLE) AS Form5Away",
        "(try_cast(Form5Home AS DOUBLE) - try_cast(Form5Away AS DOUBLE)) AS form5_diff",
    ]

# Odds → implizite Wahrscheinlichkeiten + Overround
if has("OddHome","OddDraw","OddAway"):
    S += [
        "try_cast(OddHome AS DOUBLE) AS OddHome",
        "try_cast(OddDraw AS DOUBLE) AS OddDraw",
        "try_cast(OddAway AS DOUBLE) AS OddAway",
        # inverse Quoten (implizite *nicht* normalisierte Wahrscheinlichkeiten)
        "(1.0 / NULLIF(try_cast(OddHome AS DOUBLE),0)) AS qh",
        "(1.0 / NULLIF(try_cast(OddDraw AS DOUBLE),0)) AS qd",
        "(1.0 / NULLIF(try_cast(OddAway AS DOUBLE),0)) AS qa",
        # Summe (Overround)
        "(qh + qd + qa) AS qsum",
        # normalisierte implizite Wahrscheinlichkeiten
        "(qh / NULLIF(qsum,0)) AS p_home",
        "(qd / NULLIF(qsum,0)) AS p_draw",
        "(qa / NULLIF(qsum,0)) AS p_away",
        "qsum AS overround"
    ]
    # optional: MaxHome/MaxDraw/MaxAway (falls vorhanden)
    if has("MaxHome","MaxDraw","MaxAway"):
        S += [
            "try_cast(MaxHome AS DOUBLE) AS MaxHome",
            "try_cast(MaxDraw AS DOUBLE) AS MaxDraw",
            "try_cast(MaxAway AS DOUBLE) AS MaxAway",
            "ln(NULLIF(try_cast(MaxAway AS DOUBLE),0)) - ln(NULLIF(try_cast(MaxHome AS DOUBLE),0)) AS log_odds_away_vs_home"
        ]

# Schüsse / Torschüsse
if has("HomeShots","AwayShots"):
    S += [
        "try_cast(HomeShots AS DOUBLE) AS HomeShots",
        "try_cast(AwayShots AS DOUBLE) AS AwayShots",
        "(try_cast(HomeShots AS DOUBLE) - try_cast(AwayShots AS DOUBLE)) AS shots_diff",
    ]
if has("HomeTarget","AwayTarget"):
    S += [
        "try_cast(HomeTarget AS DOUBLE) AS HomeTarget",
        "try_cast(AwayTarget AS DOUBLE) AS AwayTarget",
        "(try_cast(HomeTarget AS DOUBLE) - try_cast(AwayTarget AS DOUBLE)) AS shots_on_target_diff",
    ]
    if has("HomeShots","AwayShots"):
        S += [
            "(try_cast(HomeTarget AS DOUBLE) / NULLIF(try_cast(HomeShots AS DOUBLE),0)) AS shot_acc_home",
            "(try_cast(AwayTarget AS DOUBLE) / NULLIF(try_cast(AwayShots AS DOUBLE),0)) AS shot_acc_away",
            "((try_cast(HomeTarget AS DOUBLE) / NULLIF(try_cast(HomeShots AS DOUBLE),0))"
            " - (try_cast(AwayTarget AS DOUBLE) / NULLIF(try_cast(AwayShots AS DOUBLE),0))) AS shot_acc_diff",
        ]

# Fouls (optional)
if has("HomeFouls","AwayFouls"):
    S += [
        "try_cast(HomeFouls AS DOUBLE) AS HomeFouls",
        "try_cast(AwayFouls AS DOUBLE) AS AwayFouls",
        "(try_cast(HomeFouls AS DOUBLE) - try_cast(AwayFouls AS DOUBLE)) AS fouls_diff",
    ]

# Karten (optional)
if has("HomeYellow","AwayYellow"):
    S += [
        "try_cast(HomeYellow AS DOUBLE) AS HomeYellow",
        "try_cast(AwayYellow AS DOUBLE) AS AwayYellow",
        "(try_cast(HomeYellow AS DOUBLE) - try_cast(AwayYellow AS DOUBLE)) AS yellow_diff",
    ]
if has("HomeRed","AwayRed"):
    S += [
        "try_cast(HomeRed AS DOUBLE) AS HomeRed",
        "try_cast(AwayRed AS DOUBLE) AS AwayRed",
        "(try_cast(HomeRed AS DOUBLE) - try_cast(AwayRed AS DOUBLE)) AS red_diff",
    ]

# Ecken (optional)
if has("HomeCorners","AwayCorners"):
    S += [
        "try_cast(HomeCorners AS DOUBLE) AS HomeCorners",
        "try_cast(AwayCorners AS DOUBLE) AS AwayCorners",
        "(try_cast(HomeCorners AS DOUBLE) - try_cast(AwayCorners AS DOUBLE)) AS corners_diff",
    ]
    if has("HomeShots","AwayShots"):
        S += [
            "((try_cast(HomeCorners AS DOUBLE) - try_cast(AwayCorners AS DOUBLE))"
            " + (try_cast(HomeShots AS DOUBLE) - try_cast(AwayShots AS DOUBLE))) / 2.0 AS dominance_index"
        ]

# Form-Momentum
if has("Form5Home","Form3Home"):
    S += ["(try_cast(Form5Home AS DOUBLE) - try_cast(Form3Home AS DOUBLE)) AS form_momentum_home"]
if has("Form5Away","Form3Away"):
    S += ["(try_cast(Form5Away AS DOUBLE) - try_cast(Form3Away AS DOUBLE)) AS form_momentum_away"]

select_sql = ",\n  ".join(S)

# Features berechnen und schreiben
con.execute(f"""
CREATE OR REPLACE VIEW features_base AS
SELECT
  {select_sql}
FROM read_csv_auto('{inp.as_posix()}', header=true, all_varchar=true);
""")

con.execute(f"""
COPY features_base TO '{out_csv.as_posix()}' (HEADER, DELIMITER ',');
""")

# kurze Übersicht
summary = con.execute("""
SELECT
  COUNT(*) AS rows,
  SUM(CASE WHEN y_outcome IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_label
FROM features_base;
""").df()

print(f"✅ Geschrieben: {out_csv}")
print(summary)
