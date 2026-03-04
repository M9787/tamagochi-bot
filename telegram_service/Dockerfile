FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (lightweight — no catboost/scipy/binance)
COPY telegram_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the telegram service code
COPY telegram_service/ telegram_service/

STOPSIGNAL SIGTERM

CMD ["python", "-u", "-m", "telegram_service.service"]
