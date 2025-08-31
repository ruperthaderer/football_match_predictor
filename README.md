# Football Match Predictor  
  
End-to-end ML-Projekt: Daten → Features → Training (Optuna) → Vorhersage (CLI) → REST-API (FastAPI) → UI (Streamlit).  
Ziel: ein vorzeigbares, reproduzierbares Tech-Demo-Projekt für Bewerbungen.  
  
## Inhaltsverzeichnis
- [Quickstart](#quickstart)
- [Pipeline](#pipeline)
- [Projektstruktur](#projektstruktur)
- [Training](#training)
- [Serving](#serving)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
  
## Quickstart  

```
# (Windows PowerShell) 
python -V                # Python 3.11+ empfohlen  
pip install -r requirements-api.txt  
pip install -r requirements-streamlit.txt  
  
# 1) Training (optional – wenn models/ schon vorhanden ist, überspringen)  
python train_optuna.py --trials 60 
  
# 2) Schnelle Prediction aus CSV 
python predict.py --fixtures mock_fixtures.csv --out predictions.csv --kelly --edge-threshold 0.02  
  
# 3) API starten (lokal)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# -> Swagger unter http://127.0.0.1:8000/docs
  
# 4) Streamlit UI starten (lokal)  
streamlit run streamlit_app.py  
# -> http://localhost:8501  

```
## Ordnerstruktur

```  
football_match_predictor/  
│  
├─ data/  
│  ├─ raw/         # Rohdaten (Spiele, Quoten, ggf. Spieler); NICHT einchecken  
│  ├─ interim/     # Zwischenstände (z.B. gemappte Teams, manuelle Mappings)  
│  └─ processed/   # Trainingsfertige CSVs (z.B. features.csv)  
│  
├─ models/         # Trainingsartefakte  
│  ├─ model_calibrated.joblib  
│  ├─ feature_schema.json       # Spalten & Typen (Trainingsreihenfolge)  
│  └─ best_params.json  
│  
├─ tests/  
│  ├─ test_api_app.py           # Health/Schema/Predict smoke  
│  ├─ test_predict_cli.py       # CLI smoke  
│  └─ test_streak_one_team.py   # Bsp-Test für Feature-Logik  
│  
├─ notebooks/      # (optional) EDA/Spielwiesen  
├─ src/            # (optional) Lib-Code, wenn modularisiert wird  
│  
├─ api.py  
├─ streamlit_app.py  
├─ predict.py  
├─ train_optuna.py  
├─ feature_builder.py           # Featureerstellung aus Raw/Interim  
│  
├─ docker-compose.yml  
├─ Dockerfile.api  
├─ Dockerfile.streamlit  
│  
├─ requirements-api.txt  
├─ requirements-streamlit.txt  
└─ README.md  
```
  
## Daten-Pipeline (Reihenfolge)  
  
1) Rohdaten bereitstellen  
  
	- Lege deine Quellen in data/raw/ ab (Matches, Quoten, …).  
  
	- Falls Team-Namen nicht sauber sind: data/interim/ enthält z. B. eine manuelle Team-Map.  
  
2) Feature Engineering  
  
	- Script: feature_builder.py  
  
	- Output: data/processed/features.csv (oder features_lagged.csv, je nach Konfiguration)  
  
3) Training  
  
	- Script: train_optuna.py  
  
	- Nimmt data/processed/features.csv  
  
	- Output: models/ mit  
  
		- model_calibrated.joblib  
  
		- feature_schema.json (relevant für API/CLI/Streamlit)  
  
		- best_params.json  
  
4) Serving / UI  
  
	- CLI: predict.py liest models/ + fixtures.csv und schreibt predictions.csv.  
	  
	- API: api.py lädt models/, bietet /predict, /schema, /health.  
	  
	- UI: streamlit_app.py verbindet sich mit API (oder lädt lokal die Artefakte) und lässt dich interaktiv tippen.  
  
Wichtig: feature_schema.json ist die Wahrheit über Spaltennamen, Datentypen & Reihenfolge. Jede Inferenz (CLI/API/UI) richtet sich danach.  
  
## Training  

```
# Standard-Run (Optuna Tuning + CalibratedClassifierCV)
python train_optuna.py --trials 60
# Wichtige Artefakte landen in models/ 
```

- Train/Val/Test werden intern gesplittet (siehe Konsole).
- LogLoss/Accuracy werden für Val/Test ausgegeben.
- Die Kalibrierung wird nur auf dem trainierten Best-Modell gemacht.
- Du kannst `--trials` reduzieren, um schnelle Smoke-Runs zu machen.

---

## Vorhersagen (CLI)

Beispiel:

`python predict.py --fixtures mock_fixtures.csv --out predictions.csv --kelly --edge-threshold 0.02`

- `--fixtures`: CSV mit Spalten gemäß `models/feature_schema.json`.
- `match_date` wird als **YYYYMMDD** (int) erwartet.
- **Wichtig:** Quoten/Derived (OddHome/OddDraw/OddAway, qh/qd/qa, p_home/…) möglichst NICHT 0 lassen – sie tragen viel Signal.
- Output `predictions.csv`: enthält proba__, edge__, Kelly-Beträge (falls `--kelly`), usw.

---

## API

Start (lokal):

`uvicorn api:app --host 0.0.0.0 --port 8000 --reload`

Wichtige Endpunkte:
- `GET /health` → `{status:"ok"}`
- `GET /schema` → Feature-Schema aus `models/feature_schema.json`
- `POST /predict` → Liste von Fixture-Objekten (JSON-Array)

Beispiel-Request:
```
curl -X POST "http://localhost:8000/predict?kelly=true&edge_threshold=0.02" ^   -H "accept: application/json" ^   -H "Content-Type: application/json" ^   -d "[{     \"match_date\": 20250901,     \"HomeElo\": 1500, \"AwayElo\": 1450, \"elo_diff\": 50, \"elo_ratio\": 1.03,     \"Form3Home\": 2, \"Form3Away\": 1, \"form3_diff\": 1,     \"Form5Home\": 3, \"Form5Away\": 2, \"form5_diff\": 1,     \"OddHome\": 2.1, \"OddDraw\": 3.3, \"OddAway\": 3.5,     \"qh\": 0.33, \"qd\": 0.29, \"qa\": 0.28, \"qsum\": 0.9,     \"p_home\": 0.42, \"p_draw\": 0.29, \"p_away\": 0.29, \"overround\": 1.04,     \"MaxHome\": 2.2, \"MaxDraw\": 3.4, \"MaxAway\": 3.6,     \"log_odds_away_vs_home\": -0.15,     \"shot_acc_home\": 0.45, \"shot_acc_away\": 0.38, \"shot_acc_diff\": 0.07,     \"form_momentum_home\": 0.10, \"form_momentum_away\": -0.05   }]"
```
Swagger/Docs: `http://127.0.0.1:8000/docs`

---

## UI

Lokal:
`streamlit run streamlit_app.py # -> http://localhost:8501`
- Links oben „API Base URL“ (bei Docker Compose: `http://api:8000`, lokal: `http://127.0.0.1:8000`).
- Tabelle mit Beispiel-Row vorgefüllt; du kannst mehrere Zeilen eingeben/hochladen.
- Floats für Quoten etc. sind aktiviert; `match_date` wird automatisch in YYYYMMDD konvertiert (falls als YYYY-MM-DD eingegeben).

---

## Docker (API + UI zusammen)

Build & Start (detached):

`docker compose up --build -d` 
`# API:       http://127.0.0.1:8000 `
`# Streamlit: http://127.0.0.1:8501`

Stoppen:

`docker compose down`

**Hinweis:** In `Dockerfile.api` / `Dockerfile.streamlit` bleiben die Hosts `0.0.0.0`, damit die Services im Container erreichbar sind. Lokal im Browser  `127.0.0.1` nutzen.

---

## Tests

Einfache Smokes:
```
# API muss laufen: pytest -q tests/test_api_app.py  
# CLI Smoke: pytest -q tests/test_predict_cli.py
```

---

## Dateiformate & Schema

### `models/feature_schema.json`

- Enthält:
    - `num_cols`: numerische Spalten
    - `cat_cols`: kategoriale Spalten (bei uns i. d. R. nur `match_date`)
    - `all_feature_cols_in_training_order`: Reihenfolge im Training
    - `classes`: `["A","D","H"]` (Auswärtssieg, Remis, Heimsieg)

**Regel:** Alle Inferenzwege (CLI/API/UI) müssen exakt dieselben Spaltennamen und Typen liefern.  
`match_date` wird als **YYYYMMDD (int)** erwartet.

### Fixtures CSV (für CLI)

- Header = Spalten aus `all_feature_cols_in_training_order`
- Beispiel  in `mock_fixtures.csv`.

---

## Troubleshooting

- **422 bei API `/predict`**: Body muss ein **JSON-Array** von Objekten sein (`[ { … }, { … } ]`). Keine einzelne Map senden.
- **„feature_columns.json not found“**: Wir nutzen `feature_schema.json`. API/UI laden genau diese Datei; stelle sicher, dass `models/feature_schema.json` vorhanden ist.
- **Datetime/Float Mix-Error**: `match_date` als `YYYYMMDD` (int) schicken – keine Strings/ISO-Dates.
- **Viele 0-Werte → seltsame Prognose**: Quoten/abgeleitete Wahrscheinlichkeiten nicht auf 0 lassen. Bei fehlenden Feldern lieber Fehler werfen (Input unplausibel) oder realistische Defaults setzen.
- **Docker „Not reachable“**: In Compose bleiben Services auf `0.0.0.0`; greife extern über `127.0.0.1` zu.

---

## Ideen für Weiterentwicklung

- Feature Importance & Mini-Ablation (nur Elo vs. nur Odds vs. beides).
- Mehr Kalibrierungs-Vergleich (ohne/mit).
- Persistente DB (DuckDB/SQLite) für Spielstände & Logs.
- CI: GitHub Actions (flake8 + Smoke-Tests).
- Live-Datenquelle + CRON-Job/Workflow.
- Evtl. bessere Genauigkeit durch Einbau von Spielerdaten (Schlüsselspieler fehlt -> P(Sieg) sinkt)

---

**Kontakt**: _Rupert Haderer - ruprup3@protonmail.com_  
**Lizenz**: MIT