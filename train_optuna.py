import argparse, json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
# Standard jetzt auf features_base.csv, fällt zurück auf features_lagged.csv
parser.add_argument("--features", type=str, default="data/processed/features_base.csv")
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--blend", action="store_true")
parser.add_argument("--test_frac", type=float, default=0.125)
parser.add_argument("--val_frac", type=float, default=0.125)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ---------------- Load ----------------
features_path = Path(args.features)
if not features_path.exists():
    fallback = Path("data/processed/features_lagged.csv")
    if fallback.exists():
        print(f"ℹ️ {features_path} nicht gefunden — verwende Fallback: {fallback}")
        features_path = fallback
    else:
        raise FileNotFoundError(f"Features CSV nicht gefunden: {features_path}")

df = pd.read_csv(features_path, low_memory=False)

# Chronologische Sortierung (falls vorhanden)
if "match_date" in df.columns:
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

# Zielspalte
target_col = "y_outcome" if "y_outcome" in df.columns else ("FTResult" if "FTResult" in df.columns else None)
if target_col is None:
    raise ValueError("Keine Zielspalte gefunden (erwartet: 'y_outcome' oder 'FTResult').")
y = df[target_col].astype("category")

# ---------------- Feature-Auswahl ohne Leakage ----------------
hard_drop = {
    target_col,
    "FTHome","FTAway","FTResult","HTHome","HTAway","HTResult",
    "cfmd_match_id","Division","MatchDate","MatchTime","match_time_sort",
    "HomeTeam","AwayTeam","HomeTeam_1","AwayTeam_1","FTHome_1","FTAway_1","HomeElo_1","AwayElo_1",
    "shots_diff","shots_on_target_diff","fouls_diff","yellow_diff","red_diff","corners_diff","dominance_index",
}
hard_drop.update([c for c in df.columns if c.startswith("y_")])

post_suffixes = ("Shots","Target","Fouls","Yellow","Red","Corners")
for c in df.columns:
    if c in hard_drop:
        continue
    if c.startswith("FT") or c.startswith("HT"):
        hard_drop.add(c); continue
    if (c.startswith("Home") or c.startswith("Away")) and c.endswith(post_suffixes):
        hard_drop.add(c)

X_cols = [c for c in df.columns if c not in hard_drop]
X = df[X_cols].copy()

# numerisch/kategorisch trennen
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------------- Zeit-Splits ----------------
n = len(df)
n_test = int(n * args.test_frac)
n_val = int(n * args.val_frac)
n_train = n - n_val - n_test
if n_train <= 0:
    raise ValueError("Train-Split ist leer — val/test Fraktionen anpassen.")

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_val,   y_val   = X.iloc[n_train:n_train+n_val], y.iloc[n_train:n_train+n_val]
X_test,  y_test  = X.iloc[n_train+n_val:], y.iloc[n_train+n_val:]
print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ---------------- Preprocess ----------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# ---------------- Utils ----------------
def log_loss_with_model_labels(y_true, proba, model_classes):
    return log_loss(y_true.astype("category"), proba, labels=list(model_classes))

def odds_probs_for_index(idx, cls_order):
    """
    Liefert (n,3)-Array in Reihenfolge cls_order.
    Falls p_* NaN -> Fallback auf qh/qd/qa (schon invers = implizite Wahrscheinlichkeiten),
    also **nicht** erneut invertieren.
    """
    out = np.zeros((len(idx), 3), dtype=float)
    for i, ridx in enumerate(idx):
        row = df.loc[ridx]
        P = None
        # Bevorzugt p_*
        if all(k in df.columns for k in ["p_home","p_draw","p_away"]) \
           and pd.notna(row.get("p_home")) and pd.notna(row.get("p_draw")) and pd.notna(row.get("p_away")):
            Pmap = {"H": float(row["p_home"]), "D": float(row["p_draw"]), "A": float(row["p_away"])}
            P = [Pmap.get(c, np.nan) for c in cls_order]
        # Fallback: qh/qd/qa sind bereits 1/odds ⇒ einfach normalisieren
        elif all(k in df.columns for k in ["qh","qd","qa"]) \
             and pd.notna(row.get("qh")) and pd.notna(row.get("qd")) and pd.notna(row.get("qa")):
            qh, qd, qa = float(row["qh"]), float(row["qd"]), float(row["qa"])
            s = qh + qd + qa
            if s > 0:
                Pmap = {"H": qh/s, "D": qd/s, "A": qa/s}
                P = [Pmap.get(c, np.nan) for c in cls_order]
        if P is None or any(pd.isna(P)):
            out[i, :] = np.nan
        else:
            out[i, :] = P
    return out

# ---------------- Model Factory ----------------
def make_model(params=None):
    if params is None:
        params = dict(learning_rate=0.06, max_depth=4, max_leaf_nodes=62,
                      l2_regularization=0.0, min_samples_leaf=40)
    clf = HistGradientBoostingClassifier(
        **params, early_stopping=True, validation_fraction=0.15, random_state=args.random_state
    )
    return Pipeline([("pre", preprocess), ("clf", clf)])

# ---------------- Optuna Objective ----------------
def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [31, 63, 127, 255]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 18, 80),
        l2_regularization=trial.suggest_float("l2_regularization", 0.0, 0.5),
    )
    model = make_model(params)
    model.fit(X_train, y_train)
    classes_ = model.named_steps["clf"].classes_
    P_val = model.predict_proba(X_val)
    return log_loss_with_model_labels(y_val, P_val, classes_)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

best_params = study.best_trial.params
print("\n=== Optuna best params ===")
for k, v in best_params.items():
    print(f"{k}: {v}")

# ---------------- Fit best & validate ----------------
best_model = make_model(best_params)
best_model.fit(X_train, y_train)
classes_best = best_model.named_steps["clf"].classes_

P_val = best_model.predict_proba(X_val)
val_ll = log_loss_with_model_labels(y_val, P_val, classes_best)
val_pred = best_model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("\n=== Best HGB (Val) ===")
print("Accuracy:", val_acc)
print("LogLoss :", val_ll)
print(classification_report(y_val, val_pred))

# ---------------- Calibrate & Test ----------------
# Hinweis: cv='prefit' ist ab sklearn 1.6 deprec. Funktioniert aktuell,
# später ggf. CalibratedClassifierCV(FrozenEstimator(best_model), ...)
cal = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
cal.fit(X_val, y_val)
P_test = cal.predict_proba(X_test)
classes_cal = cal.classes_
test_ll = log_loss_with_model_labels(y_test, P_test, classes_cal)
test_acc = accuracy_score(y_test, cal.predict(X_test))
print("\n=== Calibrated Best HGB (Test) ===")
print("Accuracy:", test_acc)
print("LogLoss :", test_ll)

# ---------------- Odds Benchmark ----------------
try:
    P_val_odds = odds_probs_for_index(X_val.index, classes_cal)
    P_test_odds = odds_probs_for_index(X_test.index, classes_cal)
    m_val = ~np.isnan(P_val_odds).any(axis=1)
    m_test = ~np.isnan(P_test_odds).any(axis=1)
    print(f"\n=== Odds-only Benchmark ===")
    print(f"Coverage Val/Test: {m_val.sum()}/{len(m_val)} | {m_test.sum()}/{len(m_test)}")
    if m_val.any():
        print("Val LogLoss:", log_loss_with_model_labels(y_val.iloc[m_val.nonzero()[0]], P_val_odds[m_val], classes_cal))
        val_arg = classes_cal[P_val_odds[m_val].argmax(axis=1)]
        print("Val Acc (argmax odds): ", accuracy_score(y_val.iloc[m_val.nonzero()[0]], val_arg))
    if m_test.any():
        print("Test LogLoss:", log_loss_with_model_labels(y_test.iloc[m_test.nonzero()[0]], P_test_odds[m_test], classes_cal))
        test_arg = classes_cal[P_test_odds[m_test].argmax(axis=1)]
        print("Test Acc (argmax odds):", accuracy_score(y_test.iloc[m_test.nonzero()[0]], test_arg))
except Exception as e:
    print("\n[Odds Benchmark] Hinweis:", e)

# ---------------- Blending ----------------
if args.blend:
    try:
        P_val_model = P_val  # unkalibriert auf Val
        P_val_odds = odds_probs_for_index(X_val.index, classes_best)
        m = ~np.isnan(P_val_odds).any(axis=1)
        alphas = np.linspace(0.0, 1.0, 21)
        best_alpha, best_ll = None, 1e9
        for a in alphas:
            ll = log_loss_with_model_labels(y_val.iloc[m.nonzero()[0]], a*P_val_model[m] + (1-a)*P_val_odds[m], classes_best)
            if ll < best_ll:
                best_ll, best_alpha = ll, a
        print(f"\n=== Blending (Val) ===\nBest alpha: {best_alpha:.2f} | LogLoss: {best_ll:.6f}")

        # Test mit kalibriertem Modell
        P_test_odds = odds_probs_for_index(X_test.index, classes_cal)
        m2 = ~np.isnan(P_test_odds).any(axis=1)
        P_blend = best_alpha*P_test[m2] + (1-best_alpha)*P_test_odds[m2]
        print("\n=== Blending (Test) ===")
        print("Accuracy:", accuracy_score(y_test.iloc[m2.nonzero()[0]], classes_cal[P_blend.argmax(axis=1)]))
        print("LogLoss :", log_loss_with_model_labels(y_test.iloc[m2.nonzero()[0]], P_blend, classes_cal))
    except Exception as e:
        print("\n[Blending] Hinweis:", e)

# ---------------- Save ----------------
meta_dir = Path("models"); meta_dir.mkdir(parents=True, exist_ok=True)

# Welche Features hat das Modell gesehen?
feature_schema = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "all_feature_cols_in_training_order": X_cols,  # vor dem ColumnTransformer
    "classes": list(classes_cal)
}

dump(cal, meta_dir / "model_calibrated.joblib")
with open(meta_dir / "feature_schema.json", "w", encoding="utf-8") as f:
    json.dump(feature_schema, f, indent=2)

print(f"💾 Modell: {meta_dir/'model_calibrated.joblib'}")
print(f"💾 Schema: {meta_dir/'feature_schema.json'}")

out_dir = Path("models"); out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)
print(f"\n✅ Gespeichert: {out_dir/'best_params.json'}")
