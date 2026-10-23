"""Automated integration and end-to-end tests for the GlucoTwin FastAPI backend."""
import pytest
from fastapi.testclient import TestClient
from gluco_twin.service.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["patients_count"] == 100
    print("Test passed: Health check endpoint.")


def test_list_patients():
    response = client.get("/api/v1/patients")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 100
    p = data[0]
    assert "id" in p
    assert "name" in p
    assert "current_glucose_mgdl" in p
    assert "trend" in p
    assert "risk_severity" in p
    print(f"Test passed: List patients ({len(data)} patients returned).")


def test_get_patient_detail():
    response = client.get("/api/v1/patients/P001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "P001"
    assert "vitals" in data
    assert "labs" in data
    assert "active_conditions" in data
    assert "current_medications" in data
    assert "target_range" in data
    print("Test passed: Get patient detail.")


def test_patient_not_found():
    response = client.get("/api/v1/patients/P999_NON_EXISTENT")
    assert response.status_code == 404
    print("Test passed: 404 on missing patient.")


def test_predict_endpoint():
    response = client.get("/api/v1/patients/P001/predict")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "P001"
    assert len(data["forecast"]) == 8
    assert "event_risk" in data
    assert "probability" in data["event_risk"]
    assert "severity" in data["event_risk"]
    print("Test passed: Live trajectory prediction & spike risk.")


def test_history_endpoint():
    response = client.get("/api/v1/patients/P001/history?points=48")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 48
    assert "glucose" in data[0]
    assert "timestamp" in data[0]
    print("Test passed: CGM history chart data.")


def test_explain_endpoint():
    response = client.get("/api/v1/patients/P001/explain")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "P001"
    assert len(data["attributions"]) > 0
    assert "summary" in data
    print("Test passed: Clinical explainability.")


def test_whatif_endpoint():
    payload = {
        "carbs_g": 40.0,
        "walk_duration_min": 25.0,
        "stress_level": "none",
    }
    response = client.post("/api/v1/patients/P001/whatif", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "scenario" in data
    assert "delta" in data
    assert len(data["forecast_scenario"]) == 8
    print("Test passed: Counterfactual What-If simulation.")


def test_model_info():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "auroc" in data
    print("Test passed: Model metadata info.")


def test_websocket_stream():
    with client.websocket_connect("/api/v1/patients/P001/stream") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "tick"
        assert data["patient_id"] == "P001"
        assert "glucose_mgdl" in data
        assert "trend" in data
        print(f"Test passed: WebSocket telemetry tick received ({data['glucose_mgdl']} mg/dL).")


if __name__ == "__main__":
    test_health_check()
    test_list_patients()
    test_get_patient_detail()
    test_patient_not_found()
    test_predict_endpoint()
    test_history_endpoint()
    test_explain_endpoint()
    test_whatif_endpoint()
    test_model_info()
    test_websocket_stream()
    print("\nALL BACKEND API INTEGRATION & WEBSOCKET TESTS PASSED!")
