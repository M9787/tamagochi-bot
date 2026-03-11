FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy runtime code
COPY trading_bot.py .
COPY backfill_predictions.py .
COPY core/ core/
COPY data/ data/
COPY trading/ trading/

# Copy ML pipeline (only files needed for live prediction)
COPY model_training/__init__.py model_training/
COPY model_training/live_predict.py model_training/
COPY model_training/encode_v3.py model_training/
COPY model_training/encode_v5.py model_training/
COPY model_training/encode_v10.py model_training/
COPY model_training/build_labels.py model_training/

# Copy production models (~15MB)
COPY model_training/results_v10/production/ model_training/results_v10/production/

# Create logs directory
RUN mkdir -p trading_logs

# SIGTERM handler for graceful Docker stop
STOPSIGNAL SIGTERM

# Default: testnet, reads TRADING_THRESHOLD/TRADING_LEVERAGE/TRADING_AMOUNT from env vars
CMD ["python", "-u", "trading_bot.py", "--testnet"]
