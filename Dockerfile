FROM python:3.11-slim

WORKDIR /app

# Build-arg gate for V3 production models. Default ON for the GCE
# stack. The Contabo multi-target stack passes COPY_V3_MODELS=0 so
# the V3 5-seed bundle is not required on the bot image (the bot
# reads predictions_multitarget.csv from the shared /data volume and
# never imports the multi-target predictor code).
ARG COPY_V3_MODELS=1

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
COPY model_training/multitarget_config.py model_training/

# Conditional V3 production models (~15MB). Stage to /tmp then copy
# only if the build arg is set; lets one Dockerfile drive both stacks.
COPY model_training/results_v10/production/ /tmp/v3_models/
RUN if [ "$COPY_V3_MODELS" = "1" ]; then mkdir -p model_training/results_v10/production && cp -r /tmp/v3_models/. model_training/results_v10/production/; fi && rm -rf /tmp/v3_models

# Create logs directory
RUN mkdir -p trading_logs

# SIGTERM handler for graceful Docker stop
STOPSIGNAL SIGTERM

# Default: testnet, reads TRADING_THRESHOLD/TRADING_LEVERAGE/TRADING_AMOUNT from env vars
CMD ["python", "-u", "trading_bot.py", "--testnet"]
