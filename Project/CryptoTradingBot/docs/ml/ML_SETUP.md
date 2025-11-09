# ML Model Setup Guide

This guide will help you train the ML model and configure it for use in continuous trading.

## Step 1: Train the ML Model

First, you need to train the ML model and save it as a `.pkl` file.

### Option A: Using the training script (Recommended)

```bash
# Using Alpaca (recommended for paper trading)
python train_model.py --provider alpaca --symbol BTC/USD --limit 2000

# Or using CCXT with Binance
python train_model.py --provider ccxt --exchange binance --symbol BTC/USDT --limit 2000
```

This will:
- Fetch 2000 candles of historical data
- Train a Random Forest Classifier
- Save the model to `ml_models/trained_model.pkl`
- Display training metrics

### Option B: Using the retraining service

```bash
python scripts/retrain_model.py --provider alpaca --symbol BTC/USD --force
```

## Step 2: Update Environment Variables

Since `.env` is in `.gitignore`, you need to manually add these lines to your `.env` file:

### Required ML Settings

Add these lines to your `.env` file:

```env
# ==============================================================================
# MACHINE LEARNING CONFIGURATION
# ==============================================================================

# Enable ML predictions (set to true to use ML in trading decisions)
ML_PREDICTIONS_ENABLED=true
ENABLE_ML=true

# Model path (relative to project root)
ML_MODEL_PATH=ml_models/trained_model.pkl

# Feature engineering window size
FEATURE_WINDOW=60

# ML Model Configuration
ML_MODEL_TYPE=classifier
ML_N_ESTIMATORS=100
ML_MAX_DEPTH=10
ML_RANDOM_STATE=42

# Trading Strategy (must be 'decision_support' to use ML)
TRADING_STRATEGY=decision_support
```

### Important Notes:

1. **TRADING_STRATEGY**: Must be set to `decision_support` for ML to be used. Options:
   - `ma_crossover`: Simple moving average crossover (no ML)
   - `multi_indicator`: Multiple indicators voting (no ML)
   - `decision_support`: Full analysis with ML integration (use this for ML)

2. **ENABLE_ML**: Set to `true` to enable ML predictions

3. **ML_MODEL_PATH**: Path to your trained model file (default: `ml_models/trained_model.pkl`)

## Step 3: Verify Model File Exists

Check that the model file was created:

```bash
# Windows PowerShell
Test-Path ml_models/trained_model.pkl

# Linux/Mac
test -f ml_models/trained_model.pkl && echo "Model exists" || echo "Model not found"
```

If the file doesn't exist, run the training script from Step 1.

## Step 4: Run Continuous Trading with ML

Once the model is trained and environment variables are set:

### Option A: Using command-line flags

```bash
python main.py --mode run --provider alpaca --symbol BTC/USD --enable-ml --model-path ml_models/trained_model.pkl
```

### Option B: Using environment variables (already set in .env)

```bash
# Just run with decision_support strategy (ML will be enabled automatically)
python main.py --mode run --provider alpaca --symbol BTC/USD
```

The system will automatically:
1. Read `ENABLE_ML=true` from `.env`
2. Read `TRADING_STRATEGY=decision_support` from `.env`
3. Load the model from `ML_MODEL_PATH` (or `ml_models/trained_model.pkl`)

## Step 5: Verify ML is Active

When the continuous trader starts, look for these log messages:

```
✅ ML model loaded from ml_models/trained_model.pkl
✅ ML enabled: True
✅ Trading strategy: decision_support
```

If you see:
```
⚠️ ML model not found at ml_models/trained_model.pkl, running without ML
```

Then:
1. Make sure you trained the model (Step 1)
2. Check the model path is correct in `.env`
3. Verify the file exists: `Test-Path ml_models/trained_model.pkl`

## Troubleshooting

### Model Not Found

**Error**: `ML model not found at ml_models/trained_model.pkl`

**Solution**: 
1. Train the model first: `python train_model.py --provider alpaca`
2. Check file exists: `ls ml_models/trained_model.pkl` (Linux/Mac) or `dir ml_models\trained_model.pkl` (Windows)
3. Verify path in `.env`: `ML_MODEL_PATH=ml_models/trained_model.pkl`

### ML Not Enabled

**Symptom**: Logs show `ML enabled: False` even though you set `ENABLE_ML=true`

**Solution**:
1. Check `.env` file has `ENABLE_ML=true` (case-insensitive)
2. Restart the application (environment variables are loaded at startup)
3. Use command-line flag: `--enable-ml` to force enable

### Wrong Strategy

**Symptom**: ML is enabled but not being used in trading decisions

**Solution**:
1. Set `TRADING_STRATEGY=decision_support` in `.env`
2. The `decision_support` strategy is the only one that uses ML
3. Other strategies (`ma_crossover`, `multi_indicator`) ignore ML

### Model Training Failed

**Error**: Training script fails or produces poor accuracy

**Solution**:
1. Increase data limit: `--limit 3000` (more data = better training)
2. Check you have API keys set: `echo $ALPACA_API_KEY` (Linux/Mac) or `$env:ALPACA_API_KEY` (PowerShell)
3. Try different symbol: `--symbol ETH/USD`
4. Check internet connection (needs to fetch live data)

## Quick Start Checklist

- [ ] Train model: `python train_model.py --provider alpaca`
- [ ] Verify model exists: `Test-Path ml_models/trained_model.pkl`
- [ ] Update `.env` with ML settings (see Step 2)
- [ ] Set `TRADING_STRATEGY=decision_support` in `.env`
- [ ] Run continuous trader: `python main.py --mode run --provider alpaca`
- [ ] Verify ML is loaded in logs

## Next Steps

Once ML is working:

1. **Monitor Performance**: Check trading logs for ML predictions
2. **Retrain Regularly**: Run `python scripts/retrain_model.py` weekly
3. **Tune Parameters**: Adjust `ML_N_ESTIMATORS`, `ML_MAX_DEPTH` in `.env`
4. **Backtest**: Test ML strategy before live trading

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ML` | `false` | Enable ML predictions |
| `ML_MODEL_PATH` | `ml_models/trained_model.pkl` | Path to trained model |
| `TRADING_STRATEGY` | `ma_crossover` | Trading strategy (`decision_support` for ML) |
| `ML_MODEL_TYPE` | `classifier` | Model type (`classifier` or `regressor`) |
| `ML_N_ESTIMATORS` | `100` | Number of trees in Random Forest |
| `ML_MAX_DEPTH` | `10` | Maximum depth of trees |

