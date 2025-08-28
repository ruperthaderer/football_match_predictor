# compile_team_mapping.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
INTERIM = BASE / "data" / "interim"

auto_csv   = INTERIM / "team_name_map_auto.csv"     # aus inspect_teams.py
manual_csv = INTERIM / "team_name_map_manual.csv"   # von dir gepflegt: fbref_team,cfdm_team
final_csv  = INTERIM / "team_name_map_final.csv"

# Laden
auto = pd.read_csv(auto_csv) if auto_csv.exists() else pd.DataFrame(columns=["fbref_team","cfdm_team","norm"])
auto = auto[["fbref_team","cfdm_team"]].dropna().drop_duplicates()

manual = pd.read_csv(manual_csv)
manual = manual.rename(columns={c:c.strip() for c in manual.columns})
manual = manual[["fbref_team","cfdm_team"]].dropna().drop_duplicates()

# Zusammenführen: manuell > auto
final = pd.concat([auto, manual], ignore_index=True)
final = final.drop_duplicates(subset=["fbref_team"], keep="last").sort_values("fbref_team")

# Sanity-Checks
dup_fb = final[final.duplicated("fbref_team", keep=False)]
dup_cf = final[final.duplicated("cfdm_team", keep=False)]

if len(dup_fb):
    print("⚠️ FBref-Name mehrfach gemappt (prüfen):")
    print(dup_fb.head(20))
if len(dup_cf):
    print("ℹ️ Mehrere FBref-Namen → gleicher CFMD-Name (kann ok sein bei Namensvarianten):")
    print(dup_cf.head(20))

final.to_csv(final_csv, index=False)
print(f"✅ Geschrieben: {final_csv}  (Rows: {len(final)})")
