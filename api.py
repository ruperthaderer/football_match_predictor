# api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import logging
import traceback

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("api")

# -----------------------------------------------------------------------------
# Pfade & Laden von Modell + Schema
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
MODELS_DIR = APP_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_calibrated.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at: {MODEL_PATH}")
if not SCHEMA_PATH.exists():
    raise RuntimeError(f"Feature schema not found at: {SCHEMA_PATH}")

schema: Dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
FEATURE_COLS: List[str] = schema.get("all_feature_cols_in_training_order", [])
NUM_COLS: List[str] = schema.get("num_cols", [])
CAT_COLS: List[str] = schema.get("cat_cols", [])
LABEL_ORDER: List[str] = schema.get("classes", ["A", "D", "H"])

if not FEATURE_COLS:
    raise RuntimeError("Feature schema has empty 'all_feature_cols_in_training_order'.")

model = load(MODEL_PATH)
log.info("Model & schema loaded. Features=%d | Labels=%s", len(FEATURE_COLS), LABEL_ORDER)

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(title="Football Match Predictor API", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _coerce_dataframe(fixtures: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Records -> DataFrame
    - match_date als datetime64[ns] (passt zum Training)
    - NUM_COLS numerisch, CAT_COLS unverändert/konsistent
    - Reihenfolge exakt wie im Training, fehlende Spalten auffüllen
    """
    df = pd.DataFrame(fixtures).copy()

    # 1) match_date als echte Datetime (keine Strings/Zahlen)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=False)
        # sicherstellen, dass es naive ns-Datetimes sind (kein tz)
        if getattr(df["match_date"].dt, "tz", None) is not None:
            df["match_date"] = df["match_date"].dt.tz_localize(None)

    # 2) andere evtl. datetime-artige Objekte in CAT_COLS auch zu datetime64[ns]
    for col in CAT_COLS:
        if col in df.columns and col != "match_date":
            # nur casten, wenn es wirklich nach Datum aussieht
            if df[col].map(lambda x: isinstance(x, (np.datetime64, pd.Timestamp, str))).any():
                try:
                    s = pd.to_datetime(df[col], errors="coerce", utc=False)
                    if getattr(s.dt, "tz", None) is not None:
                        s = s.dt.tz_localize(None)
                    df[col] = s
                except Exception:
                    # wenn’s kein Datum ist, einfach so lassen (OHE kann auch object)
                    pass

    # 3) NUM_COLS numerisch; CAT_COLS NICHT anrühren
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) fehlende Trainingsspalten ergänzen
    for col in FEATURE_COLS:
        if col not in df.columns:
            if col in NUM_COLS:
                df[col] = np.nan
            else:
                # kategorial: fehlende als NaT (für Datetime) bzw. leeres Objekt
                if col == "match_date":
                    df[col] = pd.NaT
                else:
                    df[col] = pd.Series([None] * len(df), dtype="object")

    # 5) Reihenfolge exakt wie im Training
    df = df[FEATURE_COLS]

    # 6) numerische Spalten endgültig float64 + NaNs auf 0 (optional)
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def _implied_probs(odd: Any) -> float | None:
    if odd is None:
        return None
    try:
        odd = float(odd)
        if odd <= 0:
            return None
        return 1.0 / odd
    except Exception:
        return None


def _kelly_fraction(p: float, odds: float) -> float:
    b = max(odds - 1.0, 0.0)
    f = (p * (b + 1) - 1) / b if b > 0 else 0.0
    return max(f, 0.0)


def _one_result_row(
    probs: Dict[str, float],
    row: Dict[str, Any],
    kelly: bool,
    edge_threshold: float,
) -> Dict[str, Any]:
    oh, od, oa = row.get("OddHome"), row.get("OddDraw"), row.get("OddAway")
    imp_h, imp_d, imp_a = _implied_probs(oh), _implied_probs(od), _implied_probs(oa)

    edge_home = probs["H"] - (imp_h if imp_h is not None else 0.0)
    edge_draw = probs["D"] - (imp_d if imp_d is not None else 0.0)
    edge_away = probs["A"] - (imp_a if imp_a is not None else 0.0)

    out = {
        "proba_home": probs["H"],
        "proba_draw": probs["D"],
        "proba_away": probs["A"],
        "edge_home": edge_home,
        "edge_draw": edge_draw,
        "edge_away": edge_away,
        "kelly_home": 0.0,
        "kelly_draw": 0.0,
        "kelly_away": 0.0,
    }

    if kelly:
        if oh is not None and edge_home > edge_threshold:
            out["kelly_home"] = _kelly_fraction(probs["H"], float(oh))
        if od is not None and edge_draw > edge_threshold:
            out["kelly_draw"] = _kelly_fraction(probs["D"], float(od))
        if oa is not None and edge_away > edge_threshold:
            out["kelly_away"] = _kelly_fraction(probs["A"], float(oa))

    return out


class HealthResponse(BaseModel):
    status: str
    model: str
    features: int
    labels: List[str]


# -----------------------------------------------------------------------------
# Endpunkte
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="loaded" if model is not None else "missing",
        features=len(FEATURE_COLS),
        labels=LABEL_ORDER,
    )


@app.get("/feature-schema")
def get_feature_schema() -> Dict[str, Any]:
    return schema


@app.post("/predict")
def predict(
    fixtures: List[Dict[str, Any]] = Body(..., description="Liste von Partien (Records)"),
    kelly: bool = Query(False, description="Kelly-Stakes berechnen"),
    edge_threshold: float = Query(0.0, ge=0.0, description="Edge-Schwelle (z.B. 0.02 = 2%)"),
) -> List[Dict[str, Any]]:
    # vollständiger Try/Except zur aussagekräftigen Fehlerrückgabe
    try:
        if not isinstance(fixtures, list) or len(fixtures) == 0:
            raise HTTPException(status_code=422, detail="Body muss eine nicht-leere Liste sein.")

        X = _coerce_dataframe(fixtures)

        # Debug-Infos ins Log
        log.info("Predict: X.shape=%s", X.shape)
        log.info("dtypes: %s", {c: str(t) for c, t in X.dtypes.items()})

        proba = model.predict_proba(X)

        try:
            model_labels = list(model.classes_)  # type: ignore[attr-defined]
        except Exception:
            model_labels = LABEL_ORDER

        idx = {lab: model_labels.index(lab) for lab in LABEL_ORDER}

        results: List[Dict[str, Any]] = []
        for i, row in enumerate(fixtures):
            probs = {
                "A": float(proba[i, idx["A"]]),
                "D": float(proba[i, idx["D"]]),
                "H": float(proba[i, idx["H"]]),
            }
            results.append(_one_result_row(probs, row, kelly=kelly, edge_threshold=edge_threshold))

        return results

    except HTTPException:
        # bereits korrekt formularisiert
        raise
    except Exception as e:
        # sehr ausführliche Fehlermeldung zurückgeben
        tb = traceback.format_exc()
        # versuche, etwas Debug-Context zu liefern
        debug: Dict[str, Any] = {}
        try:
            debug["received_keys"] = sorted({k for rec in fixtures for k in rec.keys()})
        except Exception:
            pass
        try:
            X_dbg = _coerce_dataframe(fixtures[:1])
            debug["x_head"] = {c: float(X_dbg.iloc[0][c]) for c in X_dbg.columns}
            debug["x_dtypes"] = {c: str(t) for c, t in X_dbg.dtypes.items()}
        except Exception:
            pass

        log.error("Predict failed: %s\n%s", e, tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "trace": tb, "x_debug": debug},
        )
