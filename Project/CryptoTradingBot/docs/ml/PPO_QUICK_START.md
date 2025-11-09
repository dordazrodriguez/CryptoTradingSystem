# PPO Hybrid Trading System - Quick Start Guide

## Overview

This system combines three powerful components:
1. **Trend Following**: EMA crossovers, ADX trend strength, ATR volatility
2. **Machine Learning**: LightGBM predicts price movement probability
3. **PPO Agent**: Reinforcement learning for optimal trading decisions

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install PPO-specific packages
pip install stable-baselines3[extra] gymnasium
```

## Quick Start

### Step 1: Train ML Model (LightGBM)

```python
from ml_models.predictor import CryptoPredictionModel
import pandas as pd

# Load your data
df = pd.read_csv('data/historical_data.csv')

# Create and train model
ml_model = CryptoPredictionModel(
    algorithm='lightgbm',
    model_type='classifier',
    n_estimators=200,
    learning_rate=0.05
)

# Prepare and train
X, y = ml_model.prepare_data(df)
results = ml_model.train(X, y, validation_split=0.2)

# Save model
ml_model.save_model('models/lightgbm_trading_model.pkl')
```

### Step 2: Train PPO Agent

```bash
python scripts/train_ppo.py \
    --symbol BTC/USD \
    --timeframe 1h \
    --days 365 \
    --timesteps 100000 \
    --model-path models/ppo_trading_agent \
    --ml-model-path models/lightgbm_trading_model.pkl \
    --use-ml
```

### Step 3: Deploy for Trading

**Paper Trading (Safe Testing):**
```bash
python scripts/deploy_ppo.py \
    --ppo-model models/ppo_trading_agent \
    --ml-model models/lightgbm_trading_model.pkl \
    --symbol BTC/USD \
    --timeframe 1h \
    --paper-trading \
    --once
```

**Continuous Trading:**
```bash
python scripts/deploy_ppo.py \
    --ppo-model models/ppo_trading_agent \
    --ml-model models/lightgbm_trading_model.pkl \
    --symbol BTC/USD \
    --timeframe 1h \
    --paper-trading \
    --interval 3600
```

## System Architecture

```
Market Data
    ↓
┌─────────────────────────────────────┐
│  Technical Indicators (EMA, ADX)    │
│  ML Predictor (LightGBM)           │
│  Position Info                     │
└─────────────────────────────────────┘
    ↓
    State Vector (to PPO Agent)
    ↓
┌─────────────────────────────────────┐
│  PPO Agent Decision                │
│  (Hold/Long/Short)                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Trend Filter (EMA, ADX check)     │
│  ML Filter (probability check)     │
└─────────────────────────────────────┘
    ↓
    Final Trading Decision
```

## Decision Flow

1. **Collect Market Data**: OHLCV bars with technical indicators
2. **ML Prediction**: Get probability of upward movement (0-1)
3. **PPO Decision**: Agent suggests action (Hold/Long/Short)
4. **Trend Filter**: Check if strong trend detected (ADX > 25)
5. **ML Filter**: Validate ML prediction aligns with action
6. **Final Decision**: Execute trade or hold

## Configuration

### Environment Config

```python
from trading_engine.ppo_env import TradingEnvConfig

config = TradingEnvConfig(
    initial_cash=100000.0,       # Starting capital
    commission_rate=0.001,       # 0.1% commission
    slippage=0.0005,             # 0.05% slippage
    position_size_limit=1.0,     # Max position size (100%)
    max_drawdown_penalty=0.1,    # Drawdown penalty weight
    transaction_penalty=0.01,   # Transaction cost weight
    trend_bonus=0.05            # Trend bonus weight
)
```

### PPO Agent Config

```python
from trading_engine.ppo_agent import PPOAgent

agent = PPOAgent(
    env=env,
    learning_rate=3e-4,        # Learning rate
    n_steps=2048,              # Steps per update
    batch_size=64,             # Batch size
    n_epochs=10,               # Epochs per update
    gamma=0.99,                # Discount factor
    clip_range=0.2,            # PPO clip range
    ent_coef=0.01              # Entropy coefficient
)
```

## Example Output

```
Decision: Long (Confidence: 0.85)
Reasoning: PPO action: Long | Trend: uptrend | ML: bullish
Price: $50,000.00
Portfolio Value: $102,500.00
```

## Troubleshooting

### Issue: "stable-baselines3 not available"
**Solution**: Install with `pip install stable-baselines3[extra]`

### Issue: "No data fetched"
**Solution**: Check API keys in `.env` file or use synthetic data for testing

### Issue: "Model file not found"
**Solution**: Train models first using `scripts/train_ppo.py`

## Next Steps

1. **Backtest**: Run on historical data to validate performance
2. **Optimize**: Tune hyperparameters (learning rate, reward weights)
3. **Monitor**: Track performance metrics (Sharpe ratio, max drawdown)
4. **Retrain**: Periodically retrain models with new data

## File Structure

```
trading_engine/
├── ppo_env.py          # Gymnasium trading environment
└── ppo_agent.py        # PPO agent and hybrid strategy

scripts/
├── train_ppo.py        # PPO training script
└── deploy_ppo.py       # Deployment script

ml_models/
├── predictor.py        # ML model (LightGBM)
└── features.py        # Feature engineering

models/
├── ppo_trading_agent/  # Trained PPO model
└── lightgbm_trading_model.pkl  # Trained ML model
```

## Resources

- [Full Documentation](PPO_HYBRID_SYSTEM.md)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

