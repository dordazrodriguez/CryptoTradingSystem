# PPO Hybrid Trading System

A sophisticated trading system that combines **Trend Following**, **Machine Learning**, and **Reinforcement Learning** (PPO) for optimal trading decisions.

## Architecture Overview

The system integrates three key components:

### 1. Trend Following (Baseline Signal)
- **EMA Crossovers**: Fast EMA (12) vs Slow EMA (26)
- **ADX**: Trend strength indicator (ADX > 25 = strong trend)
- **ATR**: Volatility filter for position sizing

### 2. Machine Learning (Prediction Layer)
- **LightGBM Model**: Predicts probability of upward price movement
- **Feature Engineering**: Technical indicators, price momentum, volatility patterns
- **Probability Output**: 0-1 scale indicating bullish/bearish probability

### 3. PPO Agent (Decision Layer)
- **Reinforcement Learning**: Learns optimal trading policy from experience
- **State Space**: Technical indicators, ML probabilities, position info, price features
- **Action Space**: Hold (0), Long (1), Short (2)
- **Reward Function**: Portfolio change - drawdown penalty - transaction cost + trend bonus

## System Components

### Trading Environment (`ppo_env.py`)
- Gymnasium-compatible environment for RL training
- Simulates cryptocurrency trading with:
  - Commission (default: 0.1%)
  - Slippage (default: 0.05%)
  - Position sizing limits
  - Drawdown tracking
- State vector includes:
  - Technical indicators (EMA, ADX, ATR, RSI, MACD)
  - ML prediction probability
  - Current position and unrealized PnL
  - Price momentum and volatility features

### PPO Agent (`ppo_agent.py`)
- Wraps Stable-Baselines3 PPO algorithm
- Handles model training, saving, and loading
- Provides action predictions with probabilities
- Supports both discrete and continuous action spaces

### Hybrid Strategy (`ppo_agent.py` - HybridPPOStrategy)
- Combines all three components:
  1. **Trend Filter**: Prevents trades against strong trends
  2. **ML Filter**: Adjusts confidence based on ML predictions
  3. **PPO Decision**: Final action from trained RL agent
- Includes reasoning and confidence scoring

## Usage

### Phase 1: Train ML Model (LightGBM)

First, train your ML model:

```python
from ml_models.predictor import CryptoPredictionModel
import pandas as pd

# Load historical data
df = pd.read_csv('data/historical_data.csv')

# Create and train model
ml_model = CryptoPredictionModel(
    algorithm='lightgbm',
    model_type='classifier',
    n_estimators=200,
    learning_rate=0.05
)

# Prepare data
X, y = ml_model.prepare_data(df)

# Train
results = ml_model.train(X, y, validation_split=0.2)

# Save model
ml_model.save_model('models/lightgbm_trading_model.pkl')
```

### Phase 2: Train PPO Agent

Train the PPO agent on historical data:

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

Or programmatically:

```python
from scripts.train_ppo import train_ppo_agent
from trading_engine.ppo_env import TradingEnvConfig
from ml_models.predictor import CryptoPredictionModel
import pandas as pd

# Load data and ML model
df = pd.read_csv('data/historical_data.csv')
ml_model = CryptoPredictionModel(algorithm='lightgbm', model_type='classifier')
ml_model.load_model('models/lightgbm_trading_model.pkl')

# Configure environment
config = TradingEnvConfig(
    initial_cash=100000.0,
    commission_rate=0.001,
    slippage=0.0005,
    position_size_limit=1.0
)

# Train
results = train_ppo_agent(
    df=df,
    ml_model=ml_model,
    config=config,
    total_timesteps=100000,
    model_save_path='models/ppo_trading_agent'
)
```

### Phase 3: Deploy Hybrid System

Deploy for live trading:

```bash
python scripts/deploy_ppo.py \
    --ppo-model models/ppo_trading_agent \
    --ml-model models/lightgbm_trading_model.pkl \
    --symbol BTC/USD \
    --timeframe 1h \
    --paper-trading \
    --interval 3600
```

Or run once:

```bash
python scripts/deploy_ppo.py \
    --ppo-model models/ppo_trading_agent \
    --ml-model models/lightgbm_trading_model.pkl \
    --symbol BTC/USD \
    --once
```

### Phase 4: Continuous Training (Optional)

The system can be continuously retrained as new data arrives:

```python
from scripts.train_ppo import train_ppo_agent
from ml_models.predictor import CryptoPredictionModel

# Reload latest data
df = load_historical_data(symbol='BTC/USD', days=365)

# Retrain ML model
ml_model = CryptoPredictionModel(algorithm='lightgbm', model_type='classifier')
X, y = ml_model.prepare_data(df)
ml_model.train(X, y)
ml_model.save_model('models/lightgbm_trading_model.pkl')

# Retrain PPO agent
results = train_ppo_agent(
    df=df,
    ml_model=ml_model,
    total_timesteps=50000,  # Additional training
    model_save_path='models/ppo_trading_agent'
)
```

## Decision Flow

The hybrid system makes decisions through the following flow:

```
Market Data
    ↓
[Technical Indicators] → EMA, ADX, ATR, RSI, MACD
    ↓
[ML Model] → Probability of upward movement (0-1)
    ↓
[PPO Agent] → Action (Hold/Long/Short) with confidence
    ↓
[Trend Filter] → Override if no strong trend detected
    ↓
[ML Filter] → Adjust confidence based on ML prediction
    ↓
Final Decision → Execute trade or hold
```

### Example Decision

```python
{
    'action': 1,  # Long
    'action_name': 'Long',
    'confidence': 0.85,
    'ppo_action': 1,
    'ppo_probability': 0.92,
    'trend_signal': 'uptrend',
    'ml_signal': 'bullish',
    'reasoning': 'PPO action: Long | Trend: uptrend | ML: bullish',
    'price': 50000.0,
    'timestamp': '2024-01-01T12:00:00'
}
```

## Reward Function

The PPO agent is trained using this reward function:

```
Reward = Δ_equity - λ1 * max_drawdown - λ2 * transaction_costs + λ3 * trend_bonus
```

Where:
- **Δ_equity**: Change in portfolio value (%)
- **max_drawdown**: Maximum drawdown penalty (λ1 = 0.1)
- **transaction_costs**: Trading penalty to reduce overtrading (λ2 = 0.01)
- **trend_bonus**: Bonus for holding during strong trends (λ3 = 0.05)

## Configuration

### Environment Configuration

```python
from trading_engine.ppo_env import TradingEnvConfig

config = TradingEnvConfig(
    initial_cash=100000.0,           # Starting capital
    commission_rate=0.001,            # 0.1% commission
    slippage=0.0005,                  # 0.05% slippage
    position_size_limit=1.0,          # Max position size (100%)
    max_drawdown_penalty=0.1,         # Drawdown penalty weight
    transaction_penalty=0.01,         # Transaction cost weight
    trend_bonus=0.05,                 # Trend bonus weight
    lookback_window=50                # State lookback window
)
```

### PPO Agent Configuration

```python
from trading_engine.ppo_agent import PPOAgent

agent = PPOAgent(
    env=env,
    learning_rate=3e-4,               # Learning rate
    n_steps=2048,                     # Steps per update
    batch_size=64,                    # Batch size
    n_epochs=10,                      # Epochs per update
    gamma=0.99,                       # Discount factor
    gae_lambda=0.95,                  # GAE lambda
    clip_range=0.2,                   # PPO clip range
    ent_coef=0.01,                    # Entropy coefficient
    vf_coef=0.5                       # Value function coefficient
)
```

## Performance Monitoring

The system tracks:
- **Episode Rewards**: Total reward per episode
- **Episode Lengths**: Number of steps per episode
- **Final Equity**: Portfolio value at episode end
- **Max Drawdown**: Maximum drawdown during trading
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns

## Advantages

| Component | Role | Strength |
|-----------|------|----------|
| Trend Following | Baseline signal | Simple, explainable, filters noise |
| ML (LightGBM) | Prediction layer | Captures nonlinear patterns, probabilistic |
| PPO Agent | Decision layer | Adapts dynamically, learns optimal policy |
| Risk Manager | Safety net | Limits drawdown, ensures capital survival |

## File Structure

```
trading_engine/
├── ppo_env.py          # Gymnasium trading environment
├── ppo_agent.py        # PPO agent and hybrid strategy
├── indicators.py       # Technical indicators
├── portfolio.py        # Portfolio management
└── simple_executor.py  # Trade execution

scripts/
├── train_ppo.py        # PPO training script
└── deploy_ppo.py        # Deployment script

ml_models/
├── predictor.py        # ML model (LightGBM)
└── features.py         # Feature engineering

models/
├── ppo_trading_agent/  # Trained PPO model
└── lightgbm_trading_model.pkl  # Trained ML model
```

## Next Steps

1. **Backtest**: Run comprehensive backtesting on historical data
2. **Optimize**: Tune hyperparameters (learning rate, reward weights)
3. **Validate**: Walk-forward validation on unseen data
4. **Deploy**: Start with paper trading, then move to live
5. **Monitor**: Track performance and retrain periodically

## Dependencies

Install required packages:

```bash
pip install stable-baselines3[extra] gymnasium
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

