#!/bin/bash
# Bootstrap script for Google Cloud Engine e2-small (2 vCPU, 2GB RAM, ~$15/mo)
#
# Usage: Run on a fresh Debian/Ubuntu GCE instance
#   chmod +x gce-setup.sh && ./gce-setup.sh

set -euo pipefail

echo "=== Tamagochi Bot — GCE Setup ==="

# 1. Install Docker
echo "[1/5] Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker "$USER"
echo "Docker installed."

# 2. Setup working directory
echo "[2/5] Setting up working directory..."
WORKDIR=/opt/tamagochi
sudo mkdir -p "$WORKDIR"
sudo chown "$USER":"$USER" "$WORKDIR"
cd "$WORKDIR"

echo "Place your code in $WORKDIR or clone your repo here."
echo "Example: git clone <your-repo-url> ."

# 3. Create .env file
echo "[3/5] Setting up environment..."
if [ ! -f ".env" ]; then
    cat > .env << 'ENVEOF'
# Binance API Keys — fill these in before starting the bot
BINANCE_TESTNET_KEY=your-testnet-key-here
BINANCE_TESTNET_SECRET=your-testnet-secret-here
# For production (uncomment when ready):
# BINANCE_KEY=your-production-key
# BINANCE_SECRET=your-production-secret
ENVEOF
    echo "Created .env — EDIT THIS FILE with your API keys!"
else
    echo ".env already exists, skipping."
fi

# 4. Create persistent state file and logs dir
echo "[4/5] Creating persistent storage..."
mkdir -p trading_state
mkdir -p trading_logs

# 5. Instructions
echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Clone or copy your code to $WORKDIR"
echo "  2. Edit .env with your Binance API keys:"
echo "     nano $WORKDIR/.env"
echo "  3. Build and start the bot:"
echo "     cd $WORKDIR"
echo "     docker compose up -d --build"
echo "  4. Watch logs:"
echo "     docker compose logs -f"
echo ""
echo "Useful commands:"
echo "  docker compose stop            # Stop bot (saves state via SIGTERM)"
echo "  docker compose restart         # Restart bot"
echo "  docker compose down            # Stop and remove container"
echo "  docker compose logs --tail=50  # View recent logs"
echo ""
echo "=== Setup complete ==="
