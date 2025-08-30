# train_optuna.py
# End-to-end Training mit Optuna-Tuning (HistGradientBoosting),
# sauberer Label-Order für LogLoss + optionalem Blending mit Odds.
#
# Aufrufbeispiele:
#   python train_optuna.py --trials 60
#   python train_optuna.py --trials 60 --blend
#
# Annahmen zu Daten:
#   - CSV: data/processed/features_lagged.csv (Default via --features)
#   - Zielspalte: 'y_outcome' ∈ {'H','D','A'}
#   - Odds-Features vorhanden: p_home, p_draw, p_away  (oder ersatzweise qh,qd,qa)
#
# Wichtiger Fix (Punkt A):
#   LogLoss wird mit labels=model.classes_ gerechnet,
#   Odds-Probas werden in exakt diese Reihenfolge gebracht.

import argparse
import json
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--features", type=str, default="data/processed/features_lagged.csv",
                    help="Pfad zur Feature-Datei")
parser.add_argument("--trials", type=int, default=50, help="Anzahl Optuna-Trials")
parser.add_argument("--blend", action="store_true", help="Blending (Model + Odds) aktivieren")
parser.add_argument("--test_frac", type=float, default=0.125, help="Anteil für Test (chronologisch)")
parser.add_argument("--val_frac", type=float, default=0.125, help="Anteil für Val (vor Test, chronologisch)")
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

features_path = Path(args.features)
if not features_path.exists():
    raise FileNotFoundError(f"Features CSV nicht gefunden: {features_path}")

# ---------- Daten laden ----------
df = pd.read_csv(features_path)
# Wir erwarten eine Datumsspalte 'match_date' zur chronologischen Sortierung
if "match_date" in df.columns:
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["match_date"]).reset_index(drop=True)

# Zielspalte
target_col = "y_outcome"
if target_col not in df.columns:
    # Fallbacks
    if "FTResult" in df.columns:
        target_col = "FTResult"
    else:
        raise ValueError("Keine Zielspalte gefunden (erwartet: 'y_outcome' oder 'FTResult').")

y = df[target_col].astype("category")

# Spalten, die NICHT als Features genutzt werden sollen
drop_cols = {
    target_col,
    "match_date", "match_time_sort",
    "FTHome","FTAway","HTHome","HTAway","HTResult",
    "p_home","p_draw","p_away","qh","qd","qa","qsum","overround",
    "Division","HomeTeam","AwayTeam","cfmd_match_id","MatchDate","MatchTime",
    "HomeTeam_1","AwayTeam_1","FTHome_1","FTAway_1","HomeElo_1","AwayElo_1"
}

X_cols = [c for c in df.columns if c not in drop_cols]
X = df[X_cols].copy()

# numerische / kategoriale Spalten identifizieren
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------- Chronologisches Splitten ----------
n = len(df)
n_test = int(n * args.test_frac)
n_val = int(n * args.val_frac)
n_train = n - n_val - n_test
if n_train <= 0:
    raise ValueError("Train-Split ist leer — passe val/test Fraktionen an.")

X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_val   = X.iloc[n_train:n_train+n_val]
y_val   = y.iloc[n_train:n_train+n_val]
X_test  = X.iloc[n_train+n_val:]
y_test  = y.iloc[n_train+n_val:]

print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------- Preprocessing ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop"
)

# ---------- Helper: sichere LogLoss mit Modell-Label-Reihenfolge ----------
def log_loss_with_model_labels(y_true, proba, model_classes):
    """
    y_true: Pandas Series mit Labels ('H','D','A')
    proba:  ndarray/DataFrame shape (n_samples, n_classes) in Reihenfolge model_classes
    model_classes: array-like, z.B. ['A','D','H']
    """
    # Stelle sicher, dass y_true die gleichen Labels enthält (keine unbekannten)
    y_true = y_true.astype("category")
    # log_loss erwartet labels in der Reihenfolge, wie y_pred-Spalten angeordnet sind:
    return log_loss(y_true, proba, labels=list(model_classes))

# ---------- Odds-Probs extrahieren in Modellreihenfolge ----------
def get_odds_probs_in_order(frame_index, model_classes):
    """
    Liefert eine (n,3)-Matrix mit odds-basierten Wahrscheinlichkeiten in der Reihenfolge model_classes.
    Bevorzugt p_home/p_draw/p_away, fällt zurück auf 1/qh usw.
    """
    if set(["p_home","p_draw","p_away"]).issubset(df.columns):
        P = pd.DataFrame({
            "H": df.loc[frame_index, "p_home"].astype(float),
            "D": df.loc[frame_index, "p_draw"].astype(float),
            "A": df.loc[frame_index, "p_away"].astype(float),
        })
    elif set(["qh","qd","qa"]).issubset(df.columns):
        # rohe Quoten -> Wahrscheinlichkeiten (renormalisieren)
        inv = pd.DataFrame({
            "H": 1.0 / df.loc[frame_index, "qh"].astype(float),
            "D": 1.0 / df.loc[frame_index, "qd"].astype(float),
            "A": 1.0 / df.loc[frame_index, "qa"].astype(float),
        })
        P = inv.div(inv.sum(axis=1), axis=0)
    else:
        raise ValueError("Weder p_home/p_draw/p_away noch qh/qd/qa im DataFrame gefunden.")

    # Auf Modellreihenfolge bringen:
    order = list(model_classes)  # z.B. ['A','D','H']
    return P[order].to_numpy()

# ---------- Pipeline-Builder ----------
def make_model(trial=None):
    if trial is None:
        params = dict(
            learning_rate=0.06,
            max_depth=4,
            max_leaf_nodes=62,
            l2_regularization=0.0,
            min_samples_leaf=40
        )
    else:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=False),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [31, 63, 127, 255]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 18, 80),
            l2_regularization=trial.suggest_float("l2_regularization", 0.0, 0.5)
        )

    clf = HistGradientBoostingClassifier(
        **params,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=args.random_state
    )
    pipe = Pipeline(steps=[
        ("pre", preprocess),
        ("clf", clf)
    ])
    return pipe

# ---------- Optuna Objective ----------
def objective(trial):
    model = make_model(trial)
    model.fit(X_train, y_train)
    # Reihenfolge der Klassen vom Modell holen
    model_classes = model.named_steps["clf"].classes_
    # Val-Probabilitäten
    P_val = model.predict_proba(X_val)
    return log_loss_with_model_labels(y_val, P_val, model_classes)

study = optuna.create_study(direction="minimize")
for _ in range(args.trials):
    study.optimize(objective, n_trials=1, show_progress_bar=False)

best_params = study.best_trial.params
print("\n=== Optuna best params ===")
for k, v in best_params.items():
    print(f"{k}: {v}")

# ---------- Bestes Modell auf Train fitten ----------
best_model = make_model()
# Override mit Optuna-Params
best_model.named_steps["clf"].set_params(**best_params)
best_model.fit(X_train, y_train)

# Klassenreihenfolge festhalten
best_classes = best_model.named_steps["clf"].classes_

# ---------- Validierung ----------
P_val = best_model.predict_proba(X_val)
val_ll = log_loss_with_model_labels(y_val, P_val, best_classes)
val_pred = best_model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("\n=== Best HGB (Val) ===")
print("Accuracy:", val_acc)
print("LogLoss :", val_ll)
print(classification_report(y_val, val_pred))

# ---------- Test mit Kalibrierung ----------
cal = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
cal.fit(X_val, y_val)
P_test = cal.predict_proba(X_test)
cal_classes = cal.classes_  # sollte identisch zu best_classes sein

test_ll = log_loss_with_model_labels(y_test, P_test, cal_classes)
test_acc = accuracy_score(y_test, cal.predict(X_test))
print("\n=== Calibrated Best HGB (Test) ===")
print("Accuracy:", test_acc)
print("LogLoss :", test_ll)

# ---------- Odds-only Benchmark ----------
try:
    P_val_odds = get_odds_probs_in_order(X_val.index, cal_classes)  # in Klassen-Reihenfolge!
    P_test_odds = get_odds_probs_in_order(X_test.index, cal_classes)
    # LogLoss (Labels exakt in derselben Reihenfolge)
    odds_val_ll = log_loss_with_model_labels(y_val, P_val_odds, cal_classes)
    odds_test_ll = log_loss_with_model_labels(y_test, P_test_odds, cal_classes)

    # Accuracy via argmax
    idx2label = {i: c for i, c in enumerate(list(cal_classes))}
    val_argmax = pd.Series(P_val_odds.argmax(axis=1), index=X_val.index).map(idx2label)
    test_argmax = pd.Series(P_test_odds.argmax(axis=1), index=X_test.index).map(idx2label)

    print("\n=== Odds-only Benchmark ===")
    print(f"Coverage Val/Test: {P_val_odds.shape[0]}/{len(X_val)} | {P_test_odds.shape[0]}/{len(X_test)}")
    print("Val LogLoss:", odds_val_ll)
    print("Test LogLoss:", odds_test_ll)
    print("Val Acc (argmax odds): ", accuracy_score(y_val, val_argmax))
    print("Test Acc (argmax odds):", accuracy_score(y_test, test_argmax))
except Exception as e:
    print("\n[Odds Benchmark] Hinweis:", e)

# ---------- Optional: Blending ----------
if args.blend:
    try:
        # α via simple grid auf Val
        P_val_model = P_val  # vom unkalibrierten best_model auf Val
        P_val_odds = get_odds_probs_in_order(X_val.index, best_classes)

        alphas = np.linspace(0.0, 1.0, 21)
        best_alpha, best_blend_ll = None, 1e9
        for a in alphas:
            P_blend = a * P_val_model + (1 - a) * P_val_odds
            ll = log_loss_with_model_labels(y_val, P_blend, best_classes)
            if ll < best_blend_ll:
                best_blend_ll = ll
                best_alpha = a

        print(f"\n=== Blending (Val) ===\nBest alpha: {best_alpha:.2f} | LogLoss: {best_blend_ll:.6f}")

        # Auf Test anwenden (mit kalibriertem Modell ist üblicherweise stabiler):
        P_test_model = P_test  # kalibriert
        P_test_odds = get_odds_probs_in_order(X_test.index, cal_classes)
        P_test_blend = best_alpha * P_test_model + (1 - best_alpha) * P_test_odds
        blend_test_ll = log_loss_with_model_labels(y_test, P_test_blend, cal_classes)
        blend_test_acc = accuracy_score(y_test, [cal_classes[i] for i in P_test_blend.argmax(axis=1)])
        print("\n=== Blending (Test) ===")
        print("Accuracy:", blend_test_acc)
        print("LogLoss :", blend_test_ll)
    except Exception as e:
        print("\n[Blending] Hinweis:", e)

# ---------- Artefakte speichern ----------
out_dir = Path("models")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)
print(f"\n✅ Gespeichert: {out_dir/'best_params.json'}")
