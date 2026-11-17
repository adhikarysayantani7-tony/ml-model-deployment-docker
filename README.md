# Deploying a Machine Learning Model with Docker & FastAPI

## Project Overview
This repository packages a trained Random Forest Classifier into a production-ready **Docker** container and serves real-time predictions over a **FastAPI** REST endpoint.

## Repository Structure
- `README.md`: Project overview and quickstart guide
- `Dockerfile`: Docker configuration to install dependencies, train the model, and start Uvicorn
- `requirements.txt`: Python dependencies (`fastapi`, `uvicorn`, `scikit-learn`, `joblib`)
- `src/train_model.py`: Trains the Random Forest model and saves `model.joblib`
- `src/app.py`: FastAPI application exposing `/health` and `/predict` endpoints
- `tests/test_api.py`: Automated endpoint verification script
- `docs/deployment_guide.md`: Docker build/run commands, Cloud Run / AWS deployment steps, and API reference

## Quickstart (Docker)
1. Build the image:
   `docker build -t iris-ml-api:v1 .`
2. Run the container:
   `docker run -d -p 8000:8000 iris-ml-api:v1`
