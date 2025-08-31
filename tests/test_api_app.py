from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_single():
    payload = [{
        "match_date": "2025-09-01",
        "HomeElo": 1500, "AwayElo": 1450, "elo_diff": 50, "elo_ratio": 1.03,
        "Form3Home": 2, "Form3Away": 1, "form3_diff": 1,
        "Form5Home": 3, "Form5Away": 2, "form5_diff": 1,
        "OddHome": 2.1, "OddDraw": 3.3, "OddAway": 3.5,
        "qh": 0.33, "qd": 0.29, "qa": 0.28, "qsum": 0.9,
        "p_home": 0.42, "p_draw": 0.29, "p_away": 0.29,
        "overround": 1.04,
        "MaxHome": 2.2, "MaxDraw": 3.4, "MaxAway": 3.6,
        "log_odds_away_vs_home": -0.15,
        "shot_acc_home": 0.45, "shot_acc_away": 0.38, "shot_acc_diff": 0.07,
        "form_momentum_home": 0.10, "form_momentum_away": -0.05
    }]
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()[0]
    for k in ["proba_home","proba_draw","proba_away"]:
        assert k in out
        assert 0 <= out[k] <= 1
