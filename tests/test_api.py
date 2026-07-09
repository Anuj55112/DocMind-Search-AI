import pytest
from fastapi.testclient import TestClient
from app.api import app, startup_event

# Initialize client and trigger startup event
client = TestClient(app)
startup_event()

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_search():
    response = client.get("/search", params={"query": "vision image", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert len(data["results"]) == 1

def test_api_classify():
    payload = {
        "text": "forecasted stocks are rising rapidly",
        "candidate_labels": ["Time Series", "Computer Vision"]
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_label" in data
    assert data["top_label"] == "Time Series"

def test_api_explain():
    payload = {
        "text": "forecasted stocks are rising rapidly",
        "target_label": "Time Series",
        "candidate_labels": ["Time Series", "Computer Vision"]
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "attributions" in data
    assert len(data["attributions"]) == 5
