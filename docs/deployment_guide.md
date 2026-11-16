# ML Model Deployment & API Documentation

## 1. Architecture Overview
This project containerizes a trained Random Forest Classifier (`model.joblib`) inside a **Docker** container and serves real-time predictions over HTTP using **FastAPI** and **Uvicorn**.

```text
[Client / App] ──POST /predict──▶ [Docker Container: Uvicorn + FastAPI] ──▶ [model.joblib]
