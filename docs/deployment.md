# Deployment & Operations

## GCE Instance

- **VM**: instance-20260303-232149, c3-standard-4, asia-southeast1-b
- **Project**: project-4d1ee130-e5dc-4495-a47
- **Account**: martinproject.varzhapetyan@gmail.com
- **Code**: `/opt/tamagochi` (cloned from GitHub)
- **Config**: V10, threshold=0.70, $10, 20x leverage, SL=2%/TP=4%

## Docker Architecture

3 services in `docker-compose.yml`, shared via Docker named volume `persistent_data`:

| Service | Container | Role |
|---------|-----------|------|
| `tamagochi-data` | `data_service/Dockerfile` | L1 (klines) -> L2 (regression) -> L3 (predictions), every 5 min |
| `tamagochi-bot` | `Dockerfile` (root) | Reads predictions, places orders, manages SL/TP + safety |
| `tamagochi-telegram` | `telegram_service/Dockerfile` | 13 commands + push notifications (read-only) |

## GitHub Repository

- **URL**: https://github.com/M9787/tamagochi-bot.git (private, master branch)
- **Workflow**: Local edit -> push to GitHub -> GCE pull + rebuild

## Deployment Commands

```bash
# SSH into GCE VM
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47

# Deploy latest code
cd /opt/tamagochi && git pull origin master && docker compose up -d --build

# Remote deploy (one-liner)
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47 --command="cd /opt/tamagochi && git pull origin master && docker compose up -d --build"

# View logs
gcloud compute ssh instance-20260303-232149 --zone=asia-southeast1-b --project=project-4d1ee130-e5dc-4495-a47 --command="cd /opt/tamagochi && docker compose logs --tail=50 tamagochi-bot"
```

## Docker Commands (on GCE)

```bash
docker compose up -d --build          # Build + start all 3 services
docker compose logs -f                # Follow all logs
docker compose logs --tail=50 tamagochi-bot      # Bot logs
docker compose logs --tail=50 tamagochi-telegram  # Telegram logs
docker compose stop                   # Graceful stop (SIGTERM)
docker compose restart tamagochi-bot  # Restart bot only
docker compose down                   # Stop + remove containers
```

## Git Commands

```bash
git add <files> && git commit -m "message" && git push origin master  # Push changes
git pull origin master                                                 # Pull on GCE
```

## Environment Variables (.env on GCE)

```
BINANCE_TESTNET_KEY=...
BINANCE_TESTNET_SECRET=...
BINANCE_KEY=...
BINANCE_SECRET=...
TELEGRAM_BOT_TOKEN=...
TRADING_LEVERAGE=20       # Bot leverage
TRADING_AMOUNT=10         # Bot position size in USDT
TRADING_THRESHOLD=0.70    # Bot + Telegram default threshold
```

Located at `/opt/tamagochi/.env`, loaded via `docker-compose.yml` `env_file`. Never commit to git.
Bot reads `TRADING_*` env vars as argparse defaults -- no CLI args needed in docker-compose.
