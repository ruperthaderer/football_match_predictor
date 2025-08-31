import argparse, json, math
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path

def kelly_fraction(p, odds_dec):
    # Kelly bei Dezimalquote: f* = (b*p - (1-p))/b, b=odds-1
    b = odds_dec - 1.0
    return max(0.0, (b*p - (1-p)) / b) if b > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="CSV mit Features für kommende Spiele (gleiches Schema wie Training)")
    ap.add_argument("--out", default="predictions.csv", help="Output-CSV")
    ap.add_argument("--edge-threshold", type=float, default=0.02, help="Mindest-Edge vs. impliziter Odds-Prob für Bet-Empfehlung")
    ap.add_argument("--kelly", action="store_true", help="Kelly-Fraktion berechnen (falls Odds vorhanden)")
    args = ap.parse_args()

    model = load("models/model_calibrated.joblib")
    schema = json.loads(Path("models/feature_schema.json").read_text(encoding="utf-8"))
    feature_cols = schema["all_feature_cols_in_training_order"]
    label_order = schema["classes"]

    df = pd.read_csv(args.fixtures)
    # Falls ID/Meta-Spalten vorhanden sind, bleiben sie erhalten – wir ziehen nur die Feature-Spalten zum Modell
    X = df.reindex(columns=feature_cols, fill_value=0.0)  # robuste Alignierung
    proba = model.predict_proba(X)  # Reihenfolge entspricht label_order

    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in label_order])
    out = pd.concat([df.reset_index(drop=True), proba_df], axis=1)

    # Odds-Support (optional): erwarte Spalten 'odd_H','odd_D','odd_A' (Dezimalquoten)
    has_odds = all(c in out.columns for c in ["odd_H","odd_D","odd_A"])
    if has_odds:
        # implizite Wahrscheinlichkeiten (ohne/mit Overround-Korrektur)
        inv = 1.0/out[["odd_H","odd_D","odd_A"]]
        overround = inv.sum(axis=1)
        odds_imp = inv.div(overround, axis=0)
        out[["q_H","q_D","q_A"]] = odds_imp[["odd_H","odd_D","odd_A"]].rename(columns={"odd_H":"q_H","odd_D":"q_D","odd_A":"q_A"})

        # Edge je Ausgang
        for lab, col_p, col_q in zip(label_order, [f"p_{c}" for c in label_order], [f"q_{c}" for c in label_order]):
            out[f"edge_{lab}"] = out[col_p] - out[col_q]

        # Beste Wette pro Spiel (größte positive Edge)
        edges = out[[f"edge_{c}" for c in label_order]].values
        best_idx = np.argmax(edges, axis=1)
        best_lab = [label_order[i] for i in best_idx]
        best_edge = edges[np.arange(edges.shape[0]), best_idx]
        out["best_pick"] = best_lab
        out["best_edge"] = best_edge

        # Kelly optional
        if args.kelly:
            pick_odds = []
            for i, lab in enumerate(best_lab):
                col_odds = {"H":"odd_H", "D":"odd_D", "A":"odd_A"}[lab]
                pick_odds.append(out.loc[out.index[i], col_odds])
            pick_odds = np.array(pick_odds, dtype=float)

            pick_prob = out.lookup(out.index, [f"p_{lab}" for lab in best_lab])  # pandas <2.0; für 2.x anders lösen
            # Fallback für pandas 2.x:
            try:
                pass
            except:
                pick_prob = np.array([out.loc[i, f"p_{lab}"] for i, lab in enumerate(best_lab)], dtype=float)

            out["kelly_frac"] = [kelly_fraction(p, o) for p, o in zip(pick_prob, pick_odds)]
            out["stake_suggestion"] = out["kelly_frac"]  # skaliere extern mit Bankroll

        # Bet-Empfehlung basierend auf Edge
        out["bet_recommendation"] = np.where(out["best_edge"] >= args.edge_threshold, out["best_pick"], "")

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"gespeichert: {args.out}")

if __name__ == "__main__":
    main()
