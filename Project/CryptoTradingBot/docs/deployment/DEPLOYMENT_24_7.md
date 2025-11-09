# 24/7 Continuous Trading Deployment Guide

This guide explains how to deploy the CryptoTradingBot for continuous 24/7 operation.

## Overview

The continuous trading system consists of two main components:

1. **Continuous Trading Service**: Runs 24/7, fetches market data, generates signals, and executes trades
2. **Model Retraining Service**: Periodically retrains the ML model to maintain performance

## Prerequisites

- Python 3.11+
- Virtual environment
- API keys for your chosen data provider (CCXT or Alpaca)
- System service manager (systemd or supervisor)

## Quick Start

### 1. Basic Continuous Trading (No ML)

```bash
python main.py --mode run --symbol BTC/USDT --provider ccxt --interval 60
```

### 2. Continuous Trading with ML

```bash
python main.py --mode run \
  --symbol BTC/USDT \
  --provider alpaca \
  --interval 60 \
  --enable-ml \
  --model-path ml_models/trained_model.pkl
```

## Configuration

### Environment Variables

Set these in your environment or `.env` file:

```bash
# For Alpaca provider
export ALPACA_API_KEY=your_api_key_here
export ALPACA_SECRET_KEY=your_secret_key_here

# For CCXT (most exchanges don't require API keys for data)
# Some exchanges may require keys for rate limits
```

### Trading Parameters

Default parameters (can be adjusted in code):

- **Update Interval**: 60 seconds (configurable with `--interval`)
- **Max Position Size**: 1% of equity per trade
- **Stop Loss**: 5% from entry price
- **Starting Cash**: $10,000 (configurable with `--cash`)

## Deployment Options

### Option 1: Systemd Service (Linux)

1. **Create service file**

   Copy `deployment/systemd/crypto-trading-bot.service` to `/etc/systemd/system/`

2. **Edit service file**

   Update paths and environment variables:
   ```ini
   WorkingDirectory=/opt/crypto-trading-bot
   ExecStart=/opt/crypto-trading-bot/venv/bin/python /opt/crypto-trading-bot/main.py --mode run --symbol BTC/USDT --provider alpaca --interval 60 --enable-ml
   ```

3. **Install service**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable crypto-trading-bot
   sudo systemctl start crypto-trading-bot
   ```

4. **Monitor service**

   ```bash
   # Check status
   sudo systemctl status crypto-trading-bot
   
   # View logs
   sudo journalctl -u crypto-trading-bot -f
   
   # Stop service
   sudo systemctl stop crypto-trading-bot
   ```

### Option 2: Supervisor (Linux/Mac)

1. **Install supervisor** (if not installed)

   ```bash
   # Ubuntu/Debian
   sudo apt-get install supervisor
   
   # macOS
   pip install supervisor
   ```

2. **Create config**

   Copy `deployment/supervisor/crypto-trading-bot.conf` to `/etc/supervisor/conf.d/`

3. **Update supervisor**

   ```bash
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start crypto-trading-bot
   ```

4. **Monitor**

   ```bash
   # View logs
   sudo supervisorctl tail -f crypto-trading-bot
   
   # Check status
   sudo supervisorctl status crypto-trading-bot
   ```

### Option 3: Docker Container

1. **Build Docker image**

   ```bash
   docker build -f deployment/docker/backend.Dockerfile -t crypto-trading-bot .
   ```

2. **Run container**

   ```bash
   docker run -d \
     --name crypto-trading-bot \
     --restart unless-stopped \
     -e ALPACA_API_KEY=your_key \
     -e ALPACA_SECRET_KEY=your_secret \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/logs:/app/logs \
     crypto-trading-bot \
     python main.py --mode run --provider alpaca --enable-ml
   ```

3. **Monitor**

   ```bash
   docker logs -f crypto-trading-bot
   ```

### Option 4: Docker Compose

1. **Update docker-compose.yml**

   Add a trading service section:
   ```yaml
   trading-service:
     build:
       context: ..
       dockerfile: deployment/docker/backend.Dockerfile
     container_name: crypto-trading-bot
     environment:
       - ALPACA_API_KEY=${ALPACA_API_KEY}
       - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
     command: python main.py --mode run --provider alpaca --enable-ml --interval 60
     volumes:
       - ../data:/app/data
       - ../logs:/app/logs
     restart: unless-stopped
   ```

2. **Start service**

   ```bash
   docker-compose -f deployment/docker-compose.yml up -d trading-service
   ```

## Model Retraining

### Automatic Retraining

The model retraining service monitors model performance and retrains when:

1. **Scheduled**: After 7 days (configurable)
2. **Performance-based**: When accuracy drops below threshold (default: 0.55)
3. **Manual**: When forced with `--force` flag

### Setup Automatic Retraining

1. **Using Cron**

   ```bash
   # Edit crontab
   crontab -e
   
   # Add daily retraining check (runs at 2 AM)
   0 2 * * * cd /opt/crypto-trading-bot && /opt/crypto-trading-bot/venv/bin/python scripts/retrain_model.py >> logs/retraining.log 2>&1
   ```

2. **Using systemd timer**

   Create `/etc/systemd/system/crypto-retrain.timer`:
   ```ini
   [Unit]
   Description=Daily model retraining
   
   [Timer]
   OnCalendar=daily
   Persistent=true
   
   [Install]
   WantedBy=timers.target
   ```

   Create `/etc/systemd/system/crypto-retrain.service`:
   ```ini
   [Unit]
   Description=Crypto Trading Bot Model Retraining
   
   [Service]
   Type=oneshot
   WorkingDirectory=/opt/crypto-trading-bot
   ExecStart=/opt/crypto-trading-bot/venv/bin/python scripts/retrain_model.py
   ```

   Enable:
   ```bash
   sudo systemctl enable crypto-retrain.timer
   sudo systemctl start crypto-retrain.timer
   ```

### Manual Retraining

```bash
# Check if retraining is needed (doesn't force)
python scripts/retrain_model.py

# Force retraining regardless of conditions
python scripts/retrain_model.py --force
```

## Monitoring

### Log Files

- **Application logs**: `logs/crypto-trading-bot.out.log`
- **Error logs**: `logs/crypto-trading-bot.err.log`
- **Trading logs**: Stored in database (`data/db/trading_bot.db`)

### Health Checks

The trading service logs portfolio snapshots to the database. You can query:

```python
# Check recent portfolio snapshots
from data.db import get_db_manager
db = get_db_manager()
snapshots = db.get_portfolio_metrics(limit=10)
```

### Performance Metrics

Monitor via the Flask API:

```bash
# Health check
curl http://localhost:5000/health

# Portfolio status
curl http://localhost:5000/api/portfolio

# Recent trades
curl http://localhost:5000/api/trades?limit=10
```

## Troubleshooting

### Service Won't Start

1. **Check logs**
   ```bash
   sudo journalctl -u crypto-trading-bot -n 100
   ```

2. **Verify Python path**
   ```bash
   which python
   /opt/crypto-trading-bot/venv/bin/python --version
   ```

3. **Check permissions**
   ```bash
   ls -la /opt/crypto-trading-bot
   chmod +x main.py
   ```

### No Trades Executing

1. **Check data feed**
   ```bash
   python main.py --mode demo --symbol BTC/USDT
   ```

2. **Check signals**
   - Review logs for signal generation
   - Verify indicators are calculating correctly

3. **Check risk management**
   - Risk manager may be blocking trades
   - Review risk settings in configuration

### High CPU/Memory Usage

1. **Adjust interval**
   - Increase `--interval` to reduce update frequency

2. **Disable ML** (if not needed)
   - Remove `--enable-ml` flag

3. **Reduce data window**
   - Modify limit in data feed configuration

### Model Not Loading

1. **Check model path**
   ```bash
   ls -la ml_models/trained_model.pkl
   ```

2. **Train initial model**
   ```bash
   python scripts/retrain_model.py --force
   ```

## Best Practices

1. **Start with Demo Mode**
   - Test your setup first with `--mode demo`
   - Verify data feed is working

2. **Use Paper Trading**
   - Always use paper/sandbox mode when available
   - Never risk real money without thorough testing

3. **Monitor Regularly**
   - Check logs daily
   - Review performance weekly
   - Adjust parameters as needed

4. **Backup Database**
   - Regularly backup `data/db/trading_bot.db`
   - Keep backups of trained models

5. **Gradual Scaling**
   - Start with small position sizes
   - Increase gradually as confidence grows

6. **Model Maintenance**
   - Review model performance monthly
   - Retrain when accuracy degrades
   - Keep model version history

## Security Considerations

1. **API Keys**
   - Never commit API keys to git
   - Use environment variables or secure key management
   - Rotate keys regularly

2. **File Permissions**
   - Restrict access to config files
   - Use separate user account for service

3. **Network Security**
   - Use HTTPS for API calls (when supported)
   - Monitor for suspicious activity

4. **Rate Limiting**
   - Be aware of exchange rate limits
   - Implement proper backoff strategies

## Production Checklist

- [ ] Service runs automatically on boot
- [ ] Logs are properly rotated
- [ ] Database is backed up regularly
- [ ] Monitoring/alerts are set up
- [ ] API keys are secured
- [ ] Model retraining is scheduled
- [ ] Performance metrics are tracked
- [ ] Error handling is comprehensive
- [ ] Documentation is up to date

## Support

For issues or questions:

1. Check logs first
2. Review this documentation
3. Test with demo mode
4. Check GitHub issues (if applicable)
5. Review code comments

## Model Retraining Strategy

### Why Retrain?

Financial markets exhibit **concept drift** - patterns change over time. Models trained on historical data may lose accuracy as market conditions evolve.

### When to Retrain

1. **Scheduled**: Every 7 days (configurable)
2. **Performance-based**: When accuracy < threshold
3. **Market events**: After major market shifts
4. **Manual**: When you want to update

### Retraining Process

1. Collect recent market data (30 days default)
2. Prepare features using same pipeline
3. Train new model with walk-forward validation
4. Compare new model with current model
5. Activate better model (if improvement > 1%)

### Model Versioning

Models are versioned with timestamps:
- `trained_model_v20241201_143022.pkl`
- Active model: `trained_model.pkl` (symlink)
- Version history: `ml_models/model_versions.json`

This allows:
- Rollback to previous models
- A/B testing between versions
- Performance tracking over time

---

**Note**: This is a simulation system. Always test thoroughly before using with real money.
