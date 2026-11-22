import json
import config
from app import app

def test_flask_app():
    client = app.test_client()

    print("[*] Testing GET / (HTML Interface)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "SpamShield AI" in res.get_data(as_text=True)
    print("    GET / returned 200 OK with UI HTML")

    print("[*] Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200
    print(f"    Health: {res.get_json()}")

    print("[*] Testing GET /api/status...")
    res = client.get("/api/status")
    assert res.status_code == 200
    status_data = res.get_json()
    assert status_data["trained"] is True
    print(f"    Status: Trained={status_data['trained']}, BestModel={status_data['best_model']}")

    print("[*] Testing GET /api/stats...")
    res = client.get("/api/stats")
    assert res.status_code == 200
    stats_data = res.get_json()
    assert stats_data["success"] is True
    print(f"    Stats: Total Raw={stats_data['stats']['total_raw']}, Clean Unique={stats_data['stats']['total_clean']}, Spam={stats_data['stats']['spam_count']}")

    print("[*] Testing GET /api/results...")
    res = client.get("/api/results")
    assert res.status_code == 200
    results_data = res.get_json()
    assert len(results_data["leaderboard"]) == 4
    print(f"    Results: {len(results_data['leaderboard'])} models evaluated, Best={results_data['best_model_name']}")

    print("[*] Testing POST /api/predict (Spam sample)...")
    res = client.post("/api/predict", json={
        "text": "FREE ENTRY to win £1,000 cash! Text WIN to 87121 now. Terms apply."
    })
    assert res.status_code == 200
    pred = res.get_json()
    print(f"    Prediction: Verdict={pred['verdict']} ({pred['label']}), Conf={pred['confidence_percent']}, Triggers={pred['suspicious_terms']}")
    assert pred["is_spam"] is True

    print("[*] Testing POST /api/predict (Ham sample)...")
    res = client.post("/api/predict", json={
        "text": "Hey are we still meeting up for coffee tomorrow afternoon?"
    })
    assert res.status_code == 200
    pred_ham = res.get_json()
    print(f"    Prediction: Verdict={pred_ham['verdict']} ({pred_ham['label']}), Conf={pred_ham['confidence_percent']}")
    assert pred_ham["is_spam"] is False

    print("\n[V] ALL FLASK API ENDPOINTS AND ROUTES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_flask_app()
