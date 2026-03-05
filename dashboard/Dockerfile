FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy dashboard code
COPY backtest_dashboard.py .
COPY backfill_predictions.py .

# Copy runtime modules
COPY core/ core/
COPY data/ data/

# Copy ML pipeline (needed for backfill)
COPY model_training/__init__.py model_training/
COPY model_training/live_predict.py model_training/
COPY model_training/encode_v3.py model_training/
COPY model_training/encode_v5.py model_training/
COPY model_training/encode_v10.py model_training/
COPY model_training/build_labels.py model_training/

# Copy production models
COPY model_training/results_v10/production/ model_training/results_v10/production/

# Create dirs
RUN mkdir -p trading_logs trading_state

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "backtest_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
