# Lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Train and serialize the ML model during image build
RUN python src/train_model.py

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
