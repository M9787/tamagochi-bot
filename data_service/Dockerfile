FROM python:3.11-slim

WORKDIR /app

# Build-arg gates: GCE master defaults to V3 only; the Contabo
# multi-target stack passes COPY_V3_MODELS=0 + COPY_MT_MODELS=1.
ARG COPY_V3_MODELS=1
ARG COPY_MT_MODELS=0

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
COPY model_training/multitarget_config.py model_training/

# Conditional V3 production models (~15MB)
COPY model_training/results_v10/production/ /tmp/v3_models/
RUN if [ "$COPY_V3_MODELS" = "1" ]; then mkdir -p model_training/results_v10/production && cp -r /tmp/v3_models/. model_training/results_v10/production/; fi && rm -rf /tmp/v3_models

# Conditional multi-target models (~517MB: 24 base + 8 stacking + JSONs)
COPY model_training/results_v10/multitarget/ /tmp/mt_models/
RUN if [ "$COPY_MT_MODELS" = "1" ]; then mkdir -p model_training/results_v10/multitarget && cp -r /tmp/mt_models/base_models model_training/results_v10/multitarget/ 2>/dev/null || true && cp -r /tmp/mt_models/stacking model_training/results_v10/multitarget/ 2>/dev/null || true && cp /tmp/mt_models/multitarget_base_results.json model_training/results_v10/multitarget/ 2>/dev/null || true ; fi && rm -rf /tmp/mt_models

# feature_matrix_v10.parquet (encoder auto-init + multi-target cold-start
# backfill source) is NOT baked into the image -- it is too large for
# .dockerignore overrides and changes per training run. The Contabo
# compose file bind-mounts a host directory containing the parquet over
# /app/model_training/encoded_data/, populated by deploy/contabo_bootstrap.sh.
RUN mkdir -p model_training/encoded_data

# SIGTERM handler for graceful Docker stop
STOPSIGNAL SIGTERM

# Default: 5-minute cycles with 0.75 threshold
CMD ["python", "-u", "-m", "data_service.service", "--data-dir", "/data", "--threshold", "0.75"]
