# Docker Deployment Guide

## Quick Start

### 1. Create `.env` file

Create a `.env` file in the `deployment/` directory (or project root):

```env
# Alpaca API Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# ML Configuration (Optional)
ENABLE_ML=true
ENABLE_AUTO_RETRAINING=true
ML_MODEL_PATH=ml_models/trained_model.pkl

# Retraining Configuration (Optional)
RETRAIN_INTERVAL_DAYS=7
MIN_ACCURACY_THRESHOLD=0.48
RETRAINING_CHECK_HOURS=24

# Trading Configuration (Optional)
TRADING_STRATEGY=decision_support
RSI_BUY=55
RSI_SELL=45
PROTECT_PROFITS=true
MIN_PROFIT_TARGET=0.002
```

### 2. Train Model (if using ML)

Before deploying, train your model:

```bash
cd FINAL/CryptoTradingBot
python train_model.py --provider alpaca --symbol BTC/USD --algorithm lightgbm
```

The model will be saved to `ml_models/trained_model.pkl` and will be available in the Docker container.

### 3. Deploy with Docker Compose

```bash
cd deployment
docker compose up -d
```

### 4. View Logs

```bash
# View all logs
docker compose logs -f

# View trader logs (continuous trading service)
docker compose logs -f trader

# View backend logs
docker compose logs -f backend
```

### 5. Stop Services

```bash
docker compose down
```

## Services

### Trader Service

The main continuous trading service:

- **Container**: `crypto-trading-trader`
- **Command**: Runs `main.py --mode run` with auto-restart
- **Volumes**: 
  - `../data:/app/data` - Database and logs
  - `../logs:/app/logs` - Log files
  - `../ml_models:/app/ml_models` - ML models (for auto-retraining)

### Backend API

- **Container**: `crypto-trading-backend`
- **Port**: 5000
- **Health Check**: `/health` endpoint

### Frontend

- **Container**: `crypto-trading-frontend`
- **Port**: 3000

### Nginx (Optional)

- **Container**: `crypto-trading-nginx`
- **Ports**: 80, 443
- **SSL**: Configured via `deployment/ssl/`

## Configuration

### Enable ML

Set in `.env` file:
```env
ENABLE_ML=true
```

Or via environment variable:
```bash
ENABLE_ML=true docker compose up -d
```

### Enable Auto-Retraining

Set in `.env` file:
```env
ENABLE_ML=true
ENABLE_AUTO_RETRAINING=true
```

### Model Path

Default: `ml_models/trained_model.pkl`

To use a different model:
```env
ML_MODEL_PATH=ml_models/my_custom_model.pkl
```

## Volume Mounts

The Docker setup mounts:
- `../data` → `/app/data` - Database and data files
- `../logs` → `/app/logs` - Log files  
- `../ml_models` → `/app/ml_models` - ML models

This means:
- **Models persist** between container restarts
- **Database persists** between restarts
- **Logs persist** and can be accessed from host

## Health Checks

All services include health checks:
- **Trader**: Checks if `main.py` process is running
- **Backend**: Checks `/health` endpoint
- **Frontend**: Checks if server responds

## Troubleshooting

### Model Not Found

If you see "ML model not found", ensure:

1. Model is trained before deployment:
   ```bash
   python train_model.py --provider alpaca --algorithm lightgbm
   ```

2. Model file exists in `ml_models/trained_model.pkl`

3. Volume mount is correct (should mount `../ml_models`)

### Auto-Retraining Not Working

1. **Check ML is enabled**:
   ```bash
   docker compose logs trader | grep "ML enabled"
   ```

2. **Check auto-retraining is enabled**:
   ```bash
   docker compose logs trader | grep "Auto-retraining"
   ```

3. **Check environment variables**:
   ```bash
   docker compose exec trader env | grep ENABLE
   ```

### Container Keeps Restarting

Check logs:
```bash
docker compose logs trader
```

Common issues:
- Missing API keys (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)
- Model file not found (if ML enabled)
- Port conflicts

### Rebuild Containers

If you make code changes:
```bash
docker compose build
docker compose up -d
```

### View Container Status

```bash
docker compose ps
```

### Execute Commands in Container

```bash
# Enter trader container
docker compose exec trader bash

# Run Python commands
docker compose exec trader python -c "import ml_models; print('ML models available')"

# Check model file
docker compose exec trader ls -lh ml_models/
```

## Production Deployment

### 1. Use Docker Secrets (Recommended)

For production, use Docker secrets instead of environment variables:

```yaml
secrets:
  alpaca_api_key:
    file: ./secrets/alpaca_api_key.txt
  alpaca_secret_key:
    file: ./secrets/alpaca_secret_key.txt
```

### 2. Set Resource Limits

Add to `docker-compose.yml`:
```yaml
services:
  trader:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### 3. Enable Logging

Configure Docker logging:
```yaml
services:
  trader:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Updates

### Update Code

```bash
git pull
docker compose build
docker compose up -d
```

### Update Model

Train new model on host:
```bash
python train_model.py --algorithm lightgbm
```

Model will be available in container via volume mount. Restart trader to use:
```bash
docker compose restart trader
```

Or enable auto-retraining to update automatically.

## Monitoring

### Check Service Status

```bash
docker compose ps
```

### Monitor Logs

```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f trader
```

### Check Resource Usage

```bash
docker stats
```

## Clean Up

### Stop and Remove Containers

```bash
docker compose down
```

### Remove Volumes (⚠️ Deletes Data)

```bash
docker compose down -v
```

### Remove Images

```bash
docker compose down --rmi all
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ML` | `false` | Enable ML predictions |
| `ENABLE_AUTO_RETRAINING` | `false` | Enable automatic retraining |
| `ML_MODEL_PATH` | `ml_models/trained_model.pkl` | Path to ML model |
| `RETRAIN_INTERVAL_DAYS` | `7` | Days between scheduled retrains |
| `MIN_ACCURACY_THRESHOLD` | `0.48` | Minimum accuracy before retraining |
| `RETRAINING_CHECK_HOURS` | `24` | Hours between retraining checks |
| `TRADING_STRATEGY` | `ma_crossover` | Trading strategy |
| `ALPACA_API_KEY` | - | Alpaca API key (required) |
| `ALPACA_SECRET_KEY` | - | Alpaca secret key (required) |

