# Quick Start Guide - CryptoTradingBot

Get up and running quickly with the CryptoTradingBot in minutes!

---

## 🚀 Option 1: Local Development (Recommended for Testing)

### Prerequisites
- Python 3.11+
- pip
- (Optional) Alpaca API account

### Steps

1. **Clone and Setup**
```bash
cd CryptoTradingBot
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment (Optional)**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (if using Alpaca)
nano .env
```

3. **Initialize Database**
```bash
python main.py --init-db
```

4. **Run Demo**
```bash
# Try the demo mode
python main.py --mode demo --symbol BTC/USDT

# Or run a full simulation
python main.py --mode simulate --symbol BTC/USDT --cash 10000
```

**That's it! You're trading! 🎉**

---

## 🚀 Option 2: Docker Compose Deployment

### Prerequisites
- Docker and Docker Compose installed
- Alpaca API keys (optional, for live trading)

### Quick Start

1. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (if using Alpaca)
nano .env
```

2. **Deploy with Docker Compose**
```bash
# Navigate to deployment directory
cd deployment

# Start all services
docker compose up -d --build

# Or use the deploy script
./scripts/deploy.sh
```

3. **Access Services**
- Frontend Dashboard: http://localhost:3000
- Backend API: http://localhost:5000
- Health Check: http://localhost:5000/health

### Docker Compose Services

The deployment includes:
- **backend**: Flask API server (port 5000)
- **frontend**: React dashboard (port 3000)
- **trader**: Continuous trading service (runs 24/7)
- **nginx**: Reverse proxy (optional, ports 80/443)
- **redis**: Caching service (optional, port 6379)

### Managing Services

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f trader
docker compose logs -f backend
docker compose logs -f frontend

# Check service status
docker compose ps

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Restart a specific service
docker compose restart trader

# Rebuild and restart
docker compose up -d --build
```

### Environment Variables

Key variables for Docker Compose (set in `.env`):

```env
# Required for Alpaca trading
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

# ML Configuration
ENABLE_ML=true
ML_MODEL_PATH=ml_models/trained_model.pkl

# Trading Configuration
TRADING_STRATEGY=decision_support
RSI_BUY=55
RSI_SELL=45
```

### Troubleshooting Docker

**Issue: Port already in use**
```bash
# Check what's using the port
lsof -i :5000
lsof -i :3000

# Stop conflicting services or change ports in docker-compose.yml
```

**Issue: Services won't start**
```bash
# Check logs
docker compose logs

# Verify environment file
cat .env | grep ALPACA

# Rebuild containers
docker compose build --no-cache
docker compose up -d
```

**Issue: Database not initializing**
```bash
# Initialize database manually
docker compose exec backend python main.py --init-db
```

**For detailed Docker guide:** See `docs/deployment/DOCKER_DEPLOYMENT.md`

---

## 🚀 Option 3: VPS Deployment with Alpaca Paper Trading

### Prerequisites
- VPS (Ubuntu 20.04+)
- Alpaca paper trading account
- SSH access

### Quick Deploy (5 Minutes)

1. **On Your VPS**
```bash
# Install dependencies
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx supervisor

# Clone repository
cd /var/www
sudo git clone <your-repo> crypto-trading-bot
cd crypto-trading-bot

# Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure Alpaca**
```bash
# Copy environment file
cp .env.example .env
nano .env

# Add your Alpaca API keys:
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER_TRADING=true
```

3. **Initialize Database**
```bash
python main.py --init-db
```

4. **Run with Supervisor**
```bash
# Create supervisor config
sudo nano /etc/supervisor/conf.d/crypto-trading-bot.conf

# Add:
[program:crypto-trading-bot]
command=/var/www/crypto-trading-bot/venv/bin/python /var/www/crypto-trading-bot/main.py --mode simulate --symbol BTC/USD --cash 10000 --run
directory=/var/www/crypto-trading-bot
autostart=true
autorestart=true

# Start service
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start crypto-trading-bot
```

5. **Check Status**
```bash
# View logs
sudo supervisorctl tail -f crypto-trading-bot

# Check health
curl http://localhost:5000/health
```

**Full VPS deployment guide: `deployment/VPS_DEPLOYMENT.md`**

---

## 📊 Option 4: Run Flask API + Frontend

### Backend
```bash
# Terminal 1: Start Flask API
source venv/bin/activate
python backend/app.py

# API running at http://localhost:5000
```

### Frontend
```bash
# Terminal 2: Start React frontend
cd frontend
npm install
npm start

# Frontend running at http://localhost:3000
```

Access dashboard at: **http://localhost:3000**

---

## 🔍 What Each Mode Does

### `--mode demo`
- Fetches latest market data
- Calculates technical indicators
- Displays recent data with indicators
- **Good for:** Testing data collection and indicators

### `--mode simulate`
- Fetches historical data
- Runs trading strategy on historical data
- Executes simulated trades
- Tracks portfolio performance
- **Good for:** Backtesting and strategy testing

### `--mode run`
- Connects to live data feed
- Executes trades in real-time (simulated)
- Tracks portfolio in real-time
- **Good for:** Paper trading and live testing

## 🤖 PPO (Reinforcement Learning) Trading

The system includes a PPO-based reinforcement learning agent for advanced trading decisions.

### Quick PPO Setup

1. **Train PPO Agent**
```bash
python scripts/train_ppo.py \
    --symbol BTC/USD \
    --timeframe 1h \
    --days 365 \
    --timesteps 100000 \
    --model-path models/ppo_trading_agent \
    --ml-model-path ml_models/trained_model.pkl \
    --use-ml
```

2. **Deploy PPO Trading**
```bash
python scripts/deploy_ppo.py \
    --ppo-model models/ppo_trading_agent \
    --ml-model ml_models/trained_model.pkl \
    --symbol BTC/USD \
    --timeframe 1h \
    --paper-trading \
    --interval 3600
```

**For detailed PPO guide:** See `docs/ml/PPO_QUICK_START.md`

---

## 📝 Common Commands

```bash
# Initialize database
python main.py --init-db

# Demo mode
python main.py --mode demo --symbol BTC/USDT --provider alpaca

# Simulation with custom cash
python main.py --mode simulate --symbol ETH/USD --cash 50000 --provider alpaca

# Continuous trading with Alpaca
python main.py --mode run --symbol BTC/USD --provider alpaca --interval 60

# Enable ML predictions
python main.py --mode run --symbol BTC/USD --provider alpaca --enable-ml --model-path ml_models/trained_model.pkl

# Enable ML with auto-retraining
python main.py --mode run --symbol BTC/USD --provider alpaca --enable-ml --enable-auto-retraining --interval 60

# Run with different timeframe
python main.py --mode simulate --symbol BTC/USD --provider alpaca --timeframe 5m --cash 10000

# Train PPO agent
python scripts/train_ppo.py --symbol BTC/USD --timeframe 1h --days 365 --timesteps 100000

# Deploy PPO trading system
python scripts/deploy_ppo.py --ppo-model models/ppo_trading_agent --ml-model ml_models/trained_model.pkl --symbol BTC/USD --paper-trading

# Check Flask API health
curl http://localhost:5000/health

# View portfolio metrics
curl http://localhost:5000/api/metrics/performance

# Docker Compose commands
cd deployment
docker compose up -d                    # Start all services
docker compose logs -f                  # View all logs
docker compose logs -f trader          # View trader logs
docker compose ps                       # Check service status
docker compose restart trader           # Restart trading service
docker compose down                     # Stop all services
```

---

## 🆘 Troubleshooting

### Issue: "No module named X"
```bash
pip install -r requirements.txt
```

### Issue: "Database locked"
```bash
# Kill any running processes
pkill -f trading_bot
# Then restart
```

### Issue: "Alpaca API error"
```bash
# Check your .env file has correct keys
cat .env | grep ALPACA

# Test connection
python -c "from data.collector import AlpacaDataCollector; collector = AlpacaDataCollector(api_key='YOUR_KEY', secret_key='YOUR_SECRET'); print(collector.test_connection())"
```

### Issue: "Port 5000 already in use"
```bash
# Find process using port
lsof -i :5000

# Kill it or use different port
# Update FLASK_PORT in .env
```

### Issue: "XGBoost Library (libxgboost.dylib) could not be loaded"
```bash
# macOS: Install OpenMP runtime library
brew install libomp

# Verify installation
ls -la /opt/homebrew/opt/libomp/lib/libomp.dylib

# Test XGBoost import
python -c "import xgboost; print('XGBoost imported successfully')"
```

**Note**: This is required for macOS users. XGBoost needs the OpenMP runtime library to function properly.

---

## 🎓 Next Steps

1. **Customize Strategy**: Edit `trading_engine/simple_strategy.py`
2. **Add Indicators**: Edit `trading_engine/indicators.py`
3. **Train ML Model**: Run Jupyter notebooks
4. **Deploy Frontend**: Build and serve React app
5. **Add Monitoring**: Configure health checks

---

## 📚 Learn More

- Full documentation: `README.md`
- Deployment guide: `docs/deployment/VPS_DEPLOYMENT.md`
- PPO Quick Start: `docs/ml/PPO_QUICK_START.md`
- Running Instructions: `docs/guides/RUNNING_INSTRUCTIONS.md`
- 24/7 Deployment: `docs/deployment/DEPLOYMENT_24_7.md`

---

**Happy Trading! 🚀**
