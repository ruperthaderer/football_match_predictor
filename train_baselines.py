# train_baselines.py
# Baselines für 3-Klassen (H/D/A) auf features_lagged.csv – ohne Leakage

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

# --- 1) Daten laden ---
DATA = Path("data/processed/features_lagged.csv")
df = pd.read_csv(DATA, low_memory=False)
print(df.head())

# --- 2) Ziel & Datum vorbereiten ---
# Zielvariable in deinen Daten:
TARGET_COL = "y_outcome"   # Werte: 'H','D','A'
if TARGET_COL not in df.columns:
    raise ValueError(f"Zielspalte {TARGET_COL} nicht gefunden. Verfügbare Spalten: {list(df.columns)}")

# Datum: wir nehmen die lower-case Spalte 'match_date'
DATE_COL = "match_date"
if DATE_COL not in df.columns:
    raise ValueError(f"Datums-Spalte {DATE_COL} nicht gefunden. Verfügbare Spalten: {list(df.columns)}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# --- 3) Offensichtliche Duplikate + Leakage-Spalten droppen ---
drop_always = [
    # Metadaten/duplizierte Strings
    "Division", "HomeTeam", "AwayTeam", "MatchDate", "MatchTime", "match_time_sort",
    "Division_1", "HomeTeam_1", "AwayTeam_1",
    # Duplikate/Artefakte
    "HomeElo_1", "AwayElo_1", "FTHome_1", "FTAway_1", "HTHome", "HTAway", "HTResult",
    # Zielspalten (binary, falls später separat genutzt)
    "y_home_win",
]

# Leakage: post-match Metriken, die das Ergebnis verraten/aufnehmen
leakage_like = [
    "FTHome", "FTAway",  # Endtore
    "HomeShots","AwayShots","HomeTarget","AwayTarget",
    "HomeFouls","AwayFouls","HomeCorners","AwayCorners",
    "HomeYellow","AwayYellow","HomeRed","AwayRed",
    "shots_diff","shots_on_target_diff","shot_acc_home","shot_acc_away","shot_acc_diff",
    "fouls_diff","yellow_diff","red_diff","corners_diff","dominance_index",
]

# Falls Spalten nicht vorhanden sind: errors='ignore' unten
to_drop = [c for c in (drop_always + leakage_like) if c in df.columns]
X_all = df.drop(columns=to_drop + [TARGET_COL], errors="ignore")
y_all = df[TARGET_COL]

print(f"Features nach Drop: {X_all.shape[1]}  |  Zielverteilung: {y_all.value_counts(normalize=True).round(3).to_dict()}")

# --- 4) Zeitlicher Split (realistisch) ---
cut_train = pd.Timestamp("2019-06-30")
cut_val   = pd.Timestamp("2022-06-30")

idx_train = df[DATE_COL] <= cut_train
idx_val   = (df[DATE_COL] > cut_train) & (df[DATE_COL] <= cut_val)
idx_test  = df[DATE_COL] > cut_val

X_train = X_all.loc[idx_train]
y_train = y_all.loc[idx_train]
X_val   = X_all.loc[idx_val]
y_val   = y_all.loc[idx_val]
X_test  = X_all.loc[idx_test]
y_test  = y_all.loc[idx_test]

print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# --- 5) Nur numerische Features; Missing füllen ---
num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
X_train_num = X_train[num_cols].fillna(0)
X_val_num   = X_val[num_cols].fillna(0)
X_test_num  = X_test[num_cols].fillna(0)

# --- 6) Logistic Regression (multinomial, L1) ---
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="saga",
        penalty="l1",
        C=0.5,
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe_lr.fit(X_train_num, y_train)
proba_val_lr = pipe_lr.predict_proba(X_val_num)
pred_val_lr  = pipe_lr.predict(X_val_num)

print("\n--- Logistic Regression (Val) ---")
print("Accuracy:", accuracy_score(y_val, pred_val_lr))
print("LogLoss :", log_loss(y_val, proba_val_lr))
print(classification_report(y_val, pred_val_lr))

# --- 7) Random Forest ---
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf.fit(X_train_num, y_train)
proba_val_rf = rf.predict_proba(X_val_num)
pred_val_rf  = rf.predict(X_val_num)

print("\n--- Random Forest (Val) ---")
print("Accuracy:", accuracy_score(y_val, pred_val_rf))
print("LogLoss :", log_loss(y_val, proba_val_rf))
print(classification_report(y_val, pred_val_rf))

# --- 8) Finaler Test ---
proba_test_lr = pipe_lr.predict_proba(X_test_num)
pred_test_lr  = pipe_lr.predict(X_test_num)
print("\n=== TEST (LogReg) ===")
print("Accuracy:", accuracy_score(y_test, pred_test_lr))
print("LogLoss :", log_loss(y_test, proba_test_lr))

proba_test_rf = rf.predict_proba(X_test_num)
pred_test_rf  = rf.predict(X_test_num)
print("\n=== TEST (RandomForest) ===")
print("Accuracy:", accuracy_score(y_test, pred_test_rf))
print("LogLoss :", log_loss(y_test, proba_test_rf))
