FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install only API dependencies (keeps image small)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api/   ./api/
COPY src/   ./src/
COPY utils/ ./utils/
COPY models/ ./models/
COPY data/processed/feature_cols.pkl ./data/processed/feature_cols.pkl

ENV MODELS_DIR=models
ENV PROCESSED_DIR=data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
