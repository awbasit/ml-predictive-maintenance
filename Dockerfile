FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY utils/ ./utils/
COPY dashboard/ ./dashboard/
COPY data/processed/ ./data/processed/

ENV MODELS_DIR=models
ENV PROCESSED_DIR=data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
