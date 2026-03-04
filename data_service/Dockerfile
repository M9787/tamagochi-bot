FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy runtime code
COPY core/ core/
COPY data/ data/
COPY data_service/ data_service/

# Copy ML pipeline (only files needed for live prediction + encoding)
COPY model_training/__init__.py model_training/
COPY model_training/live_predict.py model_training/
COPY model_training/encode_v3.py model_training/
COPY model_training/encode_v5.py model_training/
COPY model_training/encode_v10.py model_training/
COPY model_training/build_labels.py model_training/

# Copy production models (~15MB)
COPY model_training/results_v10/production/ model_training/results_v10/production/

# SIGTERM handler for graceful Docker stop
STOPSIGNAL SIGTERM

# Default: 5-minute cycles with 0.75 threshold
CMD ["python", "-u", "-m", "data_service.service", "--data-dir", "/data", "--threshold", "0.75"]
