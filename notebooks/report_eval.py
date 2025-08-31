import argparse, json
import numpy as np, pandas as pd
from joblib import load
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV mit Features + y")
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    import os; os.makedirs(args.outdir, exist_ok=True)

    model = load("../models/model_calibrated.joblib")
    cols = json.load(open("models/feature_columns.json","r",encoding="utf-8"))
    labels = json.load(open("models/label_order.json","r",encoding="utf-8"))

    df = pd.read_csv(args.data)
    y = df["result"].values  # passe an deine Zielspalte an
    X = df.reindex(columns=cols, fill_value=0.0)
    proba = model.predict_proba(X)
    y_hat = np.array([labels[i] for i in np.argmax(proba, axis=1)])

    # LogLoss
    y_idx = np.array([labels.index(t) for t in y])
    ll = log_loss(y_idx, proba, labels=list(range(len(labels))))
    print(f"LogLoss: {ll:.3f}")

    # Confusion
    cm = confusion_matrix(y, y_hat, labels=labels)
    pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]) \
        .to_csv(f"{args.outdir}/confusion_{args.split}.csv")

    # Calibration plot (für Klasse H als Beispiel)
    prob_true, prob_pred = calibration_curve( (y=="H").astype(int), proba[:, labels.index("H")], n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"Calibration {args.split} (class H)")
    plt.xlabel("Predicted prob")
    plt.ylabel("Observed freq")
    plt.savefig(f"{args.outdir}/calibration_{args.split}_H.png", dpi=160, bbox_inches="tight")
    print(f"Reports in {args.outdir}")

if __name__ == "__main__":
    main()
