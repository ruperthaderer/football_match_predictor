# train_boosted.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import train_test_split

# === 1) Load ===
DATA = Path("data/processed/features_lagged.csv")
df = pd.read_csv(DATA, low_memory=False)

TARGET = "y_outcome"  # 'H','D','A'
DATE   = "match_date"

if TARGET not in df.columns: raise SystemExit(f"{TARGET} fehlt!")
if DATE   not in df.columns: raise SystemExit(f"{DATE} fehlt!")

df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

# === 2) Drop Leakage & unnötiges ===
drop_always = [
    "Division","HomeTeam","AwayTeam","MatchDate","MatchTime","match_time_sort",
    "Division_1","HomeTeam_1","AwayTeam_1","HomeElo_1","AwayElo_1","FTHome_1","FTAway_1",
    "y_home_win", "cfmd_match_id", "log_odds_away_vs_home"  # optional Meta
]
leakage = [
    "FTHome","FTAway","HTHome","HTAway","HTResult",
    "HomeShots","AwayShots","HomeTarget","AwayTarget",
    "HomeFouls","AwayFouls","HomeCorners","AwayCorners",
    "HomeYellow","AwayYellow","HomeRed","AwayRed",
    "shots_diff","shots_on_target_diff","shot_acc_home","shot_acc_away","shot_acc_diff",
    "fouls_diff","yellow_diff","red_diff","corners_diff","dominance_index",
]
to_drop = [c for c in (drop_always+leakage) if c in df.columns]
X = df.drop(columns=to_drop+[TARGET], errors="ignore")
y = df[TARGET]

# === 3) Chrono split ===
cut_train = pd.Timestamp("2019-06-30")
cut_val   = pd.Timestamp("2022-06-30")

idx_train = df[DATE] <= cut_train
idx_val   = (df[DATE] > cut_train) & (df[DATE] <= cut_val)
idx_test  = df[DATE] > cut_val

X_train, y_train = X.loc[idx_train], y.loc[idx_train]
X_val,   y_val   = X.loc[idx_val],   y.loc[idx_val]
X_test,  y_test  = X.loc[idx_test],  y.loc[idx_test]

# === 4) Nur numerische + Missing-Indicator für kritische Felder ===
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# kritische Felder, die oft NaN haben könnten
miss_keys = [c for c in [
    "home_rest_days","away_rest_days",
    "home_form_momentum","away_form_momentum",
    "home_elo_trend3","away_elo_trend3","home_elo_trend5","away_elo_trend5"
] if c in num_cols]

def add_missing_indicators(df_in, keys):
    df_out = df_in.copy()
    for c in keys:
        ind_name = f"{c}__is_na"
        df_out[ind_name] = df_out[c].isna().astype("int8")
    return df_out

def impute_group_median(df_in, cols):
    df_out = df_in.copy()
    med = df_out[cols].median(numeric_only=True)
    df_out[cols] = df_out[cols].fillna(med)
    return df_out

X_train_num = X_train[num_cols].copy()
X_val_num   = X_val[num_cols].copy()
X_test_num  = X_test[num_cols].copy()

X_train_num = add_missing_indicators(X_train_num, miss_keys)
X_val_num   = add_missing_indicators(X_val_num,   miss_keys)
X_test_num  = add_missing_indicators(X_test_num,  miss_keys)

# nach Indicators neu bestimmen (sie sind ints)
num_cols_2 = X_train_num.columns.tolist()

# Median-Imputation (global)
X_train_num = impute_group_median(X_train_num, num_cols_2)
X_val_num   = impute_group_median(X_val_num,   num_cols_2)
X_test_num  = impute_group_median(X_test_num,  num_cols_2)

# === 5) HistGradientBoosting mit Early Stopping ===
hgb = HistGradientBoostingClassifier(
    learning_rate=0.06,
    max_depth=6,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    min_samples_leaf=30,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)
hgb.fit(X_train_num, y_train)

# Val
proba_val = hgb.predict_proba(X_val_num)
pred_val  = hgb.predict(X_val_num)
print("\n=== HGB (Val) ===")
print("Accuracy:", accuracy_score(y_val, pred_val))
print("LogLoss :", log_loss(y_val, proba_val))
print(classification_report(y_val, pred_val))

# === 6) Kalibrierung auf Val (hold-out kalibrieren, dann auf Test anwenden) ===
cal = CalibratedClassifierCV(hgb, method="isotonic", cv="prefit")
cal.fit(X_val_num, y_val)

proba_test_cal = cal.predict_proba(X_test_num)
pred_test_cal  = cal.predict(X_test_num)

print("\n=== HGB + Isotonic (Test) ===")
print("Accuracy:", accuracy_score(y_test, pred_test_cal))
print("LogLoss :", log_loss(y_test, proba_test_cal))

# === 7) Odds-only Benchmark (robust, mit Fallback & NaN-Handling) ===
def build_probs(df_part: pd.DataFrame) -> pd.DataFrame | None:
    cols_p = ["p_home", "p_draw", "p_away"]
    cols_q = ["qh", "qd", "qa"]  # rohe implied probs aus Quoten (noch evtl. mit Overround)
    if all(c in df_part.columns for c in cols_p):
        P = df_part[cols_p].copy()
    elif all(c in df_part.columns for c in cols_q):
        P = df_part[cols_q].copy()
        # Negative/absurde Werte absichern
        P = P.clip(lower=0)
    else:
        return None
    # Normalisieren (Zeilensumme = 1); NaNs bleiben NaNs
    s = P.sum(axis=1)
    # Zeilen mit Summe 0 oder NaNs rausschmeißen
    valid = s.replace(0, np.nan).notna()
    P = P[valid].div(s[valid], axis=0)
    return P

def eval_odds_block(X_val, y_val, X_test, y_test):
    P_val = build_probs(X_val)
    P_test = build_probs(X_test)
    if P_val is None or P_test is None:
        print("\n(Odds-only Benchmark übersprungen: keine passenden Spalten gefunden)")
        return

    map_idx = {"H":0,"D":1,"A":2}
    # Align y auf gültige (nicht-NaN) Zeilen
    valid_val = P_val.index
    valid_test = P_test.index
    y_val_idx = y_val.loc[valid_val].map(map_idx)
    y_test_idx = y_test.loc[valid_test].map(map_idx)

    # Sicherheit: keine NaNs im y
    msk_val = y_val_idx.notna()
    msk_test = y_test_idx.notna()
    P_val = P_val.loc[valid_val[msk_val]]
    P_test = P_test.loc[valid_test[msk_test]]
    y_val_idx = y_val_idx.loc[msk_val]
    y_test_idx = y_test_idx.loc[msk_test]

    print(f"\n=== Odds-only Benchmark ===")
    print(f"Coverage Val/Test: {len(P_val)}/{len(X_val)} | {len(P_test)}/{len(X_test)}")
    print("Val LogLoss :", log_loss(y_val_idx, P_val.values, labels=[0,1,2]))
    print("Test LogLoss:", log_loss(y_test_idx, P_test.values, labels=[0,1,2]))

    # naive Accuracy (argmax)
    inv_map = {0:"H",1:"D",2:"A"}
    pred_val_lbl  = pd.Series(P_val.values.argmax(axis=1)).map(inv_map)
    pred_test_lbl = pd.Series(P_test.values.argmax(axis=1)).map(inv_map)
    print("Val Acc (argmax odds): ", accuracy_score(y_val_idx.map(inv_map), pred_val_lbl))
    print("Test Acc (argmax odds):", accuracy_score(y_test_idx.map(inv_map), pred_test_lbl))

eval_odds_block(X_val, y_val, X_test, y_test)