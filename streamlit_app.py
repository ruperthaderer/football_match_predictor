# streamlit_app.py
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------- Basis-Setup -----------------------------
st.set_page_config(page_title="Football Match Predictor – UI", layout="wide")

DEFAULT_API_BASE = "http://api:8000"  # im Docker-Compose-Setup erreichbar
MODELS_DIR = Path("models")
LOCAL_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Spalten, die wir explizit als Float behandeln wollen (hilft bei st.data_editor)
FORCE_FLOAT_COLS = [
    "OddHome", "OddDraw", "OddAway",
    "qh", "qd", "qa", "qsum",
    "p_home", "p_draw", "p_away",
    "overround", "MaxHome", "MaxDraw", "MaxAway",
    "log_odds_away_vs_home",
    "shot_acc_home", "shot_acc_away", "shot_acc_diff",
    "form_momentum_home", "form_momentum_away",
    "elo_ratio",
]

# ----------------------------- Hilfsfunktionen -----------------------------
@st.cache_data(show_spinner=False)
def api_health(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def load_schema_from_api(api_base: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{api_base.rstrip('/')}/schema"
        r = requests.get(url, timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def load_schema_local() -> Optional[Dict[str, Any]]:
    if LOCAL_SCHEMA_PATH.exists():
        try:
            return json.loads(LOCAL_SCHEMA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def coerce_df_types(df: pd.DataFrame, float_cols: List[str]) -> pd.DataFrame:
    for c in float_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            except Exception:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Leitet fehlende/abgeleitete Features her, ohne Nutzereingaben zu überschreiben."""
    df = df.copy()

    def _safe_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # Elo
    if {"HomeElo", "AwayElo"}.issubset(df.columns):
        if "elo_diff" in df.columns:
            mask = df["elo_diff"].isna() | (df["elo_diff"] == 0)
            df.loc[mask, "elo_diff"] = df.loc[mask, ["HomeElo", "AwayElo"]].apply(
                lambda r: _safe_num(r["HomeElo"]) - _safe_num(r["AwayElo"]), axis=1
            )
        if "elo_ratio" in df.columns:
            mask = df["elo_ratio"].isna() | (df["elo_ratio"] == 0)
            df.loc[mask, "elo_ratio"] = df.loc[mask, ["HomeElo", "AwayElo"]].apply(
                lambda r: ( _safe_num(r["HomeElo"]) / _safe_num(r["AwayElo"]) )
                if _safe_num(r["AwayElo"]) not in (0, np.nan) else np.nan,
                axis=1
            )

    # Form-Diffs
    if {"Form3Home", "Form3Away", "form3_diff"}.issubset(df.columns):
        mask = df["form3_diff"].isna() | (df["form3_diff"] == 0)
        df.loc[mask, "form3_diff"] = df.loc[mask, "Form3Home"] - df.loc[mask, "Form3Away"]

    if {"Form5Home", "Form5Away", "form5_diff"}.issubset(df.columns):
        mask = df["form5_diff"].isna() | (df["form5_diff"] == 0)
        df.loc[mask, "form5_diff"] = df.loc[mask, "Form5Home"] - df.loc[mask, "Form5Away"]

    # Shot accuracy diff
    if {"shot_acc_home", "shot_acc_away"}.issubset(df.columns):
        if "shot_acc_diff" in df.columns:
            mask = df["shot_acc_diff"].isna() | (df["shot_acc_diff"] == 0)
            df.loc[mask, "shot_acc_diff"] = df.loc[mask, "shot_acc_home"] - df.loc[mask, "shot_acc_away"]

    # Odds -> qh/qd/qa, p_*, overround, log_odds_away_vs_home
    if {"OddHome", "OddDraw", "OddAway"}.issubset(df.columns):
        # Hilfsspalten qh/qd/qa
        for c_src, c_q in [("OddHome", "qh"), ("OddDraw", "qd"), ("OddAway", "qa")]:
            if c_q in df.columns:
                mask = df[c_q].isna() | (df[c_q] == 0)
                df.loc[mask, c_q] = df.loc[mask, c_src].apply(lambda o: 1.0 / _safe_num(o) if _safe_num(o) not in (0, np.nan) else np.nan)

        # qsum
        if {"qh", "qd", "qa"}.issubset(df.columns) and "qsum" in df.columns:
            mask = df["qsum"].isna() | (df["qsum"] == 0)
            df.loc[mask, "qsum"] = df.loc[mask, ["qh", "qd", "qa"]].sum(axis=1, min_count=1)

        # p_*
        if {"p_home", "p_draw", "p_away", "qsum", "qh", "qd", "qa"}.issubset(df.columns):
            for p_col, q_col in [("p_home", "qh"), ("p_draw", "qd"), ("p_away", "qa")]:
                mask = df[p_col].isna() | (df[p_col] == 0)
                df.loc[mask, p_col] = df.loc[mask, [q_col, "qsum"]].apply(
                    lambda r: _safe_num(r[q_col]) / _safe_num(r["qsum"])
                    if np.all(pd.notna([_safe_num(r[q_col]), _safe_num(r["qsum"])]) ) and _safe_num(r["qsum"]) != 0
                    else np.nan,
                    axis=1,
                )

        # overround
        if "overround" in df.columns and "qsum" in df.columns:
            mask = df["overround"].isna() | (df["overround"] == 0)
            df.loc[mask, "overround"] = df.loc[mask, "qsum"]

        # log_odds_away_vs_home
        if "log_odds_away_vs_home" in df.columns:
            mask = df["log_odds_away_vs_home"].isna() | (df["log_odds_away_vs_home"] == 0)
            df.loc[mask, "log_odds_away_vs_home"] = df.loc[mask, ["OddAway", "OddHome"]].apply(
                lambda r: math.log(_safe_num(r["OddAway"]) / _safe_num(r["OddHome"]))
                if np.all(pd.notna([_safe_num(r["OddAway"]), _safe_num(r["OddHome"])])) and _safe_num(r["OddAway"]) > 0 and _safe_num(r["OddHome"]) > 0
                else np.nan,
                axis=1,
            )

    return df


def to_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Datum ggf. in ISO konvertieren, ansonsten alles als natives Python
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = None
                continue
            if "date" in k.lower():
                try:
                    rec[k] = pd.to_datetime(v).date().isoformat()
                except Exception:
                    rec[k] = str(v)
            else:
                if isinstance(v, (np.floating, np.float32, np.float64)):
                    rec[k] = float(v)
                elif isinstance(v, (np.integer, np.int32, np.int64)):
                    rec[k] = int(v)
                else:
                    rec[k] = v
        out.append(rec)
    return out


def call_predict(
    api_base: str,
    fixtures: List[Dict[str, Any]],
    kelly: bool,
    edge_threshold: float,
) -> List[Dict[str, Any]]:
    url = (
        f"{api_base.rstrip('/')}/predict"
        f"?kelly={'true' if kelly else 'false'}"
        f"&edge_threshold={edge_threshold}"
    )
    r = requests.post(url, json=fixtures, timeout=20)  # Body = reine Liste
    r.raise_for_status()
    return r.json()

# ----------------------------- Sidebar --------------------------------
st.sidebar.header("⚙️ Einstellungen")
api_base = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE)

healthy = api_health(api_base)
st.sidebar.markdown(
    f"**API Status:** {'✅ erreichbar' if healthy else '❌ nicht erreichbar'}  \n"
    f"`{api_base}`"
)

# ----------------------------- Schema laden ----------------------------
schema = load_schema_from_api(api_base) if healthy else None
if schema is None:
    schema = load_schema_local()

if not schema:
    st.title("Football Match Predictor – UI")
    st.error(
        "Konnte das Feature-Schema weder von der API noch lokal laden. "
        "Stelle sicher, dass die API läuft **oder** `models/feature_schema.json` vorhanden ist."
    )
    st.stop()

num_cols: List[str] = schema.get("num_cols", [])
cat_cols: List[str] = schema.get("cat_cols", [])
all_cols: List[str] = schema.get("all_feature_cols_in_training_order", [])

# ----------------------------- Titel -----------------------------------
st.title("⚽ Football Match Predictor – UI")

# ----------------------------- Eingabe-Form -----------------------------
st.subheader("🔢 Eingabe")
st.caption("Gib eine oder mehrere Partien ein. Spalten müssen die Trainings-Features matchen.")

# Beispielzeile: standardmäßig NICHT 0, sondern NaN (leeres Feld)
example = {c: np.nan for c in num_cols}
if "match_date" in all_cols:
    example["match_date"] = "2025-09-01"

# DataFrame exakt in Trainings-Reihenfolge
df_init = pd.DataFrame([{c: example.get(c, "") for c in all_cols}])
df_init = coerce_df_types(df_init, FORCE_FLOAT_COLS)

# Option: abgeleitete Werte automatisch berechnen (Standard: an)
auto_derive = st.checkbox("Abgeleitete Werte automatisch berechnen (Odds → p_*, Elo → Diffs, …)", value=True)

edited_df = st.data_editor(
    df_init,
    num_rows="dynamic",
    use_container_width=True,
    key="fixtures_editor",
)

# Optionen
col1, col2 = st.columns([1, 1])
with col1:
    use_kelly = st.checkbox("Kelly berechnen", value=True)
with col2:
    edge_threshold = st.number_input(
        "Edge-Threshold (z.B. 0.02 = 2%)",
        min_value=0.0, max_value=0.2, step=0.005, value=0.02, format="%.3f"
    )

# ----------------------------- Aktion: Vorhersagen ----------------------
predict_btn = st.button("🔮 Vorhersagen", type="primary")

if predict_btn:
    try:
        # Floats coercen
        edited_df = coerce_df_types(edited_df, FORCE_FLOAT_COLS)

        # Optional: Ableitungen berechnen
        if auto_derive:
            edited_df = recompute_derived(edited_df)

        # Nur Spalten, die das Modell kennt – und in richtiger Reihenfolge
        payload_df = edited_df[[c for c in all_cols if c in edited_df.columns]].copy()

        fixtures = to_payload(payload_df)
        if len(fixtures) == 0:
            st.warning("Keine Eingaben vorhanden.")
            st.stop()

        if not healthy:
            st.error("API nicht erreichbar – bitte API starten oder Base URL prüfen.")
            st.stop()

        with st.spinner("Sende an API..."):
            result = call_predict(
                api_base=api_base,
                fixtures=fixtures,
                kelly=use_kelly,
                edge_threshold=float(edge_threshold),
            )

        # Ergebnis anzeigen
        res_df = pd.DataFrame(result)
        st.success("Vorhersagen erhalten ✅")

        show_df = pd.concat([payload_df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
        st.dataframe(show_df, use_container_width=True)

        # Download
        csv_bytes = show_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Ergebnisse als CSV herunterladen",
            data=csv_bytes,
            file_name="predictions_ui.csv",
            mime="text/csv",
        )

    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        st.error(f"API-Fehler: {detail}")
    except Exception as e:
        st.exception(e)

# ----------------------------- Footer -----------------------------------
st.markdown("---")
st.caption("Model: `models/model_calibrated.joblib` • Schema: `models/feature_schema.json`")
