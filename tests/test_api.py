"""
API Endpoint Verification Script
Internship Track: Week 4 - Task 2
Author: Sayantani Adhikary
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    print("Health Check Response:", resp.json())


def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    assert resp.status_code == 200
    print("Prediction Response:", resp.json())


if __name__ == "__main__":
    test_health()
    test_predict()
