# Running Instructions - CryptoTradingBot

Complete guide for running the CryptoTradingBot system locally or on a VPS.

---

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [VPS Production Deployment](#vps-production-deployment)
3. [Running Modes](#running-modes)
4. [Docker Deployment](#docker-deployment)
5. [Service Management](#service-management)
6. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- **Python 3.11+** (Check with `python3 --version`)
- **pip** (Python package manager)
- **Git**
- **(Optional)** Alpaca API account for live trading data

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone <repository-url>
cd CryptoTradingBot
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment (Optional)

```bash
# Create .env file from template
cp .env.example .env

# Edit .env file with your API keys (if using Alpaca)
nano .env  # or use your preferred editor
```

**Example `.env` file:**
```env
# Alpaca API Configuration (Optional)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_PAPER_TRADING=true

# Flask Configuration
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key-here
```

#### 5. Initialize Database

```bash
python main.py --init-db
```

This creates the SQLite database in `data/db/trading_bot.db`.

#### 6. Test Installation

```bash
# Run demo mode to verify everything works
python main.py --mode demo --symbol BTC/USDT
```

You should see output showing:
- Data fetched successfully
- Indicators calculated
- Trading signals generated

---

## Running Modes

### Mode 1: Demo Mode (Quick Test)

Test data fetching and indicator calculations without trading:

```bash
python main.py --mode demo --symbol BTC/USDT
```

**Options:**
- `--symbol`: Trading pair (e.g., `BTC/USDT`, `ETH/USD`)
- `--exchange`: Exchange name (default: `binance`)
- `--timeframe`: Data timeframe (default: `1m`)

**Example:**
```bash
python main.py --mode demo --symbol ETH/USDT --exchange binance --timeframe 5m
```

### Mode 2: Simulation Mode (Backtesting)

Run a complete backtest on historical data:

```bash
python main.py --mode simulate --symbol BTC/USDT --cash 10000
```

**Options:**
- `--symbol`: Trading pair
- `--cash`: Starting cash balance (default: 10000)
- `--exchange`: Exchange name
- `--timeframe`: Data timeframe

**Example:**
```bash
python main.py --mode simulate \
  --symbol BTC/USDT \
  --cash 50000 \
  --exchange binance \
  --timeframe 1m
```

### Mode 3: Continuous Trading (24/7)

Run continuous trading service:

```bash
python main.py --mode run \
  --symbol BTC/USDT \
  --provider ccxt \
  --interval 60
```

**Options:**
- `--symbol`: Trading pair
- `--provider`: Data provider (`ccxt` or `alpaca`)
- `--interval`: Update interval in seconds (default: 60)
- `--cash`: Starting cash balance
- `--enable-ml`: Enable ML predictions (requires trained model)
- `--model-path`: Path to ML model file

**Examples:**

```bash
# Basic continuous trading (CCXT - no API keys needed)
python main.py --mode run --symbol BTC/USDT --provider ccxt --interval 60

# With Alpaca (requires API keys in .env)
python main.py --mode run --symbol BTC/USD --provider alpaca --interval 60

# With ML predictions
python main.py --mode run \
  --symbol BTC/USDT \
  --provider alpaca \
  --interval 60 \
  --enable-ml \
  --model-path ml_models/trained_model.pkl
```

**Stopping the Service:**
Press `Ctrl+C` for graceful shutdown.

---

## VPS Production Deployment

### Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Root or sudo access
- Alpaca API account (recommended for production)
- Domain name (optional, for frontend access)

### Step-by-Step VPS Setup

#### 1. Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  git \
  nginx \
  supervisor \
  sqlite3 \
  curl
```

#### 2. Create Application User

```bash
# Create dedicated user for trading bot
sudo useradd -m -s /bin/bash crypto-trader
sudo passwd crypto-trader  # Set password or use SSH keys
```

#### 3. Clone and Setup Application

```bash
# Switch to application user
sudo su - crypto-trader

# Clone repository
cd ~
git clone <repository-url> crypto-trading-bot
cd crypto-trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
# Create .env file
nano .env
```

Add your configuration:

```env
# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER_TRADING=true

# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=$(openssl rand -hex 32)

# Database
DATABASE_URL=sqlite:///home/crypto-trader/crypto-trading-bot/data/db/trading_bot.db
```

#### 5. Initialize Database

```bash
python main.py --init-db
```

#### 6. Test Run

```bash
# Test demo mode
python main.py --mode demo --symbol BTC/USDT

# If successful, proceed with service setup
```

### Setup as System Service (systemd)

#### 1. Create Service File

```bash
sudo nano /etc/systemd/system/crypto-trading-bot.service
```

Add the following:

```ini
[Unit]
Description=CryptoTradingBot Continuous Trading Service
After=network.target

[Service]
Type=simple
User=crypto-trader
WorkingDirectory=/home/crypto-trader/crypto-trading-bot
Environment="PATH=/home/crypto-trader/crypto-trading-bot/venv/bin"
Environment="PYTHONPATH=/home/crypto-trader/crypto-trading-bot"
EnvironmentFile=/home/crypto-trader/crypto-trading-bot/.env
ExecStart=/home/crypto-trader/crypto-trading-bot/venv/bin/python /home/crypto-trader/crypto-trading-bot/main.py --mode run --symbol BTC/USDT --provider alpaca --interval 60 --enable-ml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading-bot

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ReadWritePaths=/home/crypto-trader/crypto-trading-bot/data /home/crypto-trader/crypto-trading-bot/logs

[Install]
WantedBy=multi-user.target
```

#### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable crypto-trading-bot

# Start service
sudo systemctl start crypto-trading-bot

# Check status
sudo systemctl status crypto-trading-bot
```

#### 3. View Logs

```bash
# View logs
sudo journalctl -u crypto-trading-bot -f

# View last 100 lines
sudo journalctl -u crypto-trading-bot -n 100
```

### Setup with Supervisor (Alternative)

#### 1. Create Supervisor Config

```bash
sudo nano /etc/supervisor/conf.d/crypto-trading-bot.conf
```

Add:

```ini
[program:crypto-trading-bot]
command=/home/crypto-trader/crypto-trading-bot/venv/bin/python /home/crypto-trader/crypto-trading-bot/main.py --mode run --symbol BTC/USDT --provider alpaca --interval 60 --enable-ml
directory=/home/crypto-trader/crypto-trading-bot
user=crypto-trader
autostart=true
autorestart=true
startretries=3
environment=PYTHONPATH="/home/crypto-trader/crypto-trading-bot"
stderr_logfile=/home/crypto-trader/crypto-trading-bot/logs/crypto-trading-bot.err.log
stdout_logfile=/home/crypto-trader/crypto-trading-bot/logs/crypto-trading-bot.out.log
stopwaitsecs=30
stopasgroup=true
killasgroup=true
```

#### 2. Update Supervisor

```bash
# Create logs directory
mkdir -p /home/crypto-trader/crypto-trading-bot/logs

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start crypto-trading-bot

# Check status
sudo supervisorctl status crypto-trading-bot
```

### Setup Model Retraining (Cron)

#### 1. Create Retraining Script

```bash
nano /home/crypto-trader/crypto-trading-bot/scripts/retrain-scheduled.sh
```

Add:

```bash
#!/bin/bash
cd /home/crypto-trader/crypto-trading-bot
source venv/bin/activate
python scripts/retrain_model.py --force >> logs/retraining.log 2>&1
```

Make executable:

```bash
chmod +x /home/crypto-trader/crypto-trading-bot/scripts/retrain-scheduled.sh
```

#### 2. Setup Cron Job

```bash
crontab -e
```

Add (runs daily at 2 AM):

```
0 2 * * * /home/crypto-trader/crypto-trading-bot/scripts/retrain-scheduled.sh
```

---

## Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose installed

### Quick Start

```bash
# Build and start all services
cd deployment
./scripts/deploy.sh
```

### Manual Docker Commands

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.yml down

# Rebuild and restart
docker-compose -f deployment/docker-compose.yml up -d --build
```

### Access Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

---

## Service Management

### systemd Commands

```bash
# Start service
sudo systemctl start crypto-trading-bot

# Stop service
sudo systemctl stop crypto-trading-bot

# Restart service
sudo systemctl restart crypto-trading-bot

# Check status
sudo systemctl status crypto-trading-bot

# View logs
sudo journalctl -u crypto-trading-bot -f

# Disable auto-start
sudo systemctl disable crypto-trading-bot

# Enable auto-start
sudo systemctl enable crypto-trading-bot
```

### Supervisor Commands

```bash
# Start service
sudo supervisorctl start crypto-trading-bot

# Stop service
sudo supervisorctl stop crypto-trading-bot

# Restart service
sudo supervisorctl restart crypto-trading-bot

# View status
sudo supervisorctl status crypto-trading-bot

# View logs
sudo tail -f /home/crypto-trader/crypto-trading-bot/logs/crypto-trading-bot.out.log
```

---

## Running Different Components

### Flask API Server

```bash
# Start Flask API
python backend/app.py

# Or with Flask CLI
export FLASK_APP=backend/app.py
flask run --host=0.0.0.0 --port=5000
```

Access API at: http://localhost:5000

### React Frontend

```bash
cd frontend
npm install
npm start
```

Access frontend at: http://localhost:3000

### Model Retraining

#### Standard Retraining

```bash
# Check if retraining needed
python scripts/retrain_model.py

# Force retraining
python scripts/retrain_model.py --force
```

#### Prefect-based Retraining

```bash
# Start Prefect server (if not running)
prefect server start

# Run Prefect workflow
python scripts/retrain_model_prefect.py --symbol BTC/USDT --days 30

# Deploy workflows
python deployment/prefect/deploy.py
```

### Data Collection

```bash
# Collect training data manually
python -c "
from ml_models.prefect_flows import data_collection_flow
data = data_collection_flow(symbol='BTC/USDT', hours=24)
print(f'Collected {len(data)} samples')
"
```

---

## Configuration Files

### Main Configuration

**Location**: `.env`

```env
# API Keys
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Flask
FLASK_ENV=production
FLASK_SECRET_KEY=your-secret-key

# Database
DATABASE_URL=sqlite:///path/to/db/trading_bot.db
```

### Trading Parameters

Edit trading parameters in:
- `main.py` - Default configuration
- `trading_engine/continuous_service.py` - Service configuration
- Systemd service file - Runtime configuration

**Key Parameters:**
- `max_position_size`: Maximum position size (default: 1% of equity)
- `stop_loss_pct`: Stop loss percentage (default: 5%)
- `interval`: Update interval in seconds (default: 60)

---

## Troubleshooting

### Common Issues

#### 1. Python Version Error

**Error**: `python: command not found`

**Solution**:
```bash
# Use python3 explicitly
python3 main.py --mode demo

# Or create alias
alias python=python3
```

#### 2. Module Not Found

**Error**: `ModuleNotFoundError: No module named 'ccxt'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Database Locked

**Error**: `database is locked`

**Solution**:
```bash
# Stop any running instances
sudo systemctl stop crypto-trading-bot

# Check for lock files
ls -la data/db/

# Remove lock if safe (backup first!)
# SQLite should handle this automatically
```

#### 4. API Connection Failed

**Error**: `Connection failed` or `Rate limit exceeded`

**Solution**:
- Check internet connection
- Verify API keys in `.env`
- Wait for rate limit reset
- Check exchange status

#### 5. Service Won't Start

**Check logs**:
```bash
sudo journalctl -u crypto-trading-bot -n 50
```

**Common causes**:
- Wrong Python path in service file
- Missing environment variables
- Permission issues
- Database path incorrect

**Fix permissions**:
```bash
sudo chown -R crypto-trader:crypto-trader /home/crypto-trader/crypto-trading-bot
```

#### 6. Prefect Not Working

**Error**: `Prefect not available`

**Solution**:
```bash
# Install Prefect
pip install prefect>=2.14.0

# Verify installation
python -c "import prefect; print(prefect.__version__)"
```

#### 7. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :5000  # For Flask
sudo lsof -i :3000  # For React

# Kill process
sudo kill -9 <PID>

# Or use different port
flask run --port 5001
```

### Debug Mode

Run with debug logging:

```bash
# Set log level
export PYTHONPATH=.
export LOG_LEVEL=DEBUG

# Run with verbose output
python main.py --mode run --symbol BTC/USDT --provider ccxt --interval 60 -v
```

### Health Checks

```bash
# Check Flask API health
curl http://localhost:5000/health

# Check service status
sudo systemctl status crypto-trading-bot

# Check database
sqlite3 data/db/trading_bot.db ".tables"
```

### Performance Monitoring

```bash
# Check system resources
htop

# Check disk space
df -h

# Check Python processes
ps aux | grep python

# Monitor logs in real-time
tail -f logs/crypto-trading-bot.out.log
```

---

## Quick Reference Commands

### Local Development

```bash
# Activate environment
source venv/bin/activate

# Run demo
python main.py --mode demo --symbol BTC/USDT

# Run simulation
python main.py --mode simulate --symbol BTC/USDT --cash 10000

# Run continuous trading
python main.py --mode run --symbol BTC/USDT --provider ccxt --interval 60
```

### VPS Production

```bash
# Start service
sudo systemctl start crypto-trading-bot

# View logs
sudo journalctl -u crypto-trading-bot -f

# Restart service
sudo systemctl restart crypto-trading-bot

# Check status
sudo systemctl status crypto-trading-bot
```

### Model Retraining

```bash
# Standard retraining
python scripts/retrain_model.py --force

# Prefect retraining
python scripts/retrain_model_prefect.py --symbol BTC/USDT --days 30
```

---

## Next Steps

After getting the system running:

1. **Monitor Performance**: Check logs and dashboard regularly
2. **Tune Parameters**: Adjust trading parameters based on performance
3. **Setup Alerts**: Configure email/SMS alerts for critical events
4. **Backup Database**: Regularly backup `data/db/trading_bot.db`
5. **Review Metrics**: Use Flask API to query performance metrics
6. **Update Models**: Retrain models regularly (use cron or Prefect)

---

## Additional Resources

- [24/7 Deployment Guide](docs/DEPLOYMENT_24_7.md) - Detailed production deployment
- [Prefect Integration Guide](docs/PREFECT_INTEGRATION.md) - Workflow orchestration
- [README.md](README.md) - Project overview and features
- [Quick Start Guide](QUICK_START.md) - Quick setup instructions

---

**Note**: Always use paper trading for testing. Never risk real money without thorough testing!

