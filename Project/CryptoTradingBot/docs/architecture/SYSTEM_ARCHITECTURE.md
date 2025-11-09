# CryptoTradingBot - System Architecture & Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Technical Indicators](#technical-indicators)
3. [Trading Strategy](#trading-strategy)
4. [Signal Generation](#signal-generation)
5. [Trade Execution](#trade-execution)
6. [Risk Management](#risk-management)
7. [Architecture Components](#architecture-components)
8. [Configuration](#configuration)

---

## System Overview

The CryptoTradingBot is an automated cryptocurrency trading system that:
- Fetches real-time market data from Alpaca Markets (paper trading)
- Calculates 26 technical indicators for market analysis
- Generates trading signals based on moving average crossovers and trend following
- Executes trades automatically with risk management
- Tracks portfolio performance and logs all trades to a SQLite database

### Key Features
- **Continuous Trading**: Runs 24/7 with configurable update intervals
- **Real-time Data**: Fetches live market data from Alpaca API
- **Paper Trading**: Safe testing environment before live trading
- **Risk Management**: ATR-based position sizing and stop-loss protection
- **Comprehensive Indicators**: 26 technical indicators for market analysis
- **Trade Logging**: All trades and portfolio snapshots stored in database

---

## Technical Indicators

The system calculates **26 technical indicators** to analyze market conditions:

### Moving Averages (10 indicators)

**Simple Moving Averages (SMA)**
- `sma_5` - 5-period Simple Moving Average
- `sma_10` - 10-period Simple Moving Average
- `sma_20` - 20-period Simple Moving Average
- `sma_50` - 50-period Simple Moving Average
- `sma_200` - 200-period Simple Moving Average

**Exponential Moving Averages (EMA)**
- `ema_5` - 5-period Exponential Moving Average
- `ema_10` - 10-period Exponential Moving Average
- `ema_20` - 20-period Exponential Moving Average
- `ema_50` - 50-period Exponential Moving Average
- `ema_200` - 200-period Exponential Moving Average

**Purpose**: Identify trend direction and support/resistance levels. Fast periods (5-20) react quickly to price changes, while slow periods (50-200) smooth out noise and show long-term trends.

### Momentum Indicators (1 indicator)

**RSI - Relative Strength Index**
- `rsi` - 14-period RSI (0-100 scale)
  - Values < 30: Oversold (potential buy signal)
  - Values > 70: Overbought (potential sell signal)
  - Values 40-60: Neutral zone

**Purpose**: Measures momentum and identifies overbought/oversold conditions. Used for signal confirmation (buy signals when RSI is low, sell signals when RSI is high).

### Trend Indicators (3 indicators)

**MACD - Moving Average Convergence Divergence**
- `macd` - MACD line (12-period EMA - 26-period EMA)
- `macd_signal` - Signal line (9-period EMA of MACD)
- `macd_histogram` - Histogram (MACD - Signal)

**Purpose**: Identifies trend changes and momentum. Bullish when MACD crosses above signal, bearish when it crosses below.

### Volatility Indicators (5 indicators)

**Bollinger Bands**
- `bb_upper` - Upper band (SMA 20 + 2 standard deviations)
- `bb_middle` - Middle band (SMA 20)
- `bb_lower` - Lower band (SMA 20 - 2 standard deviations)
- `bb_width` - Band width ((upper - lower) / middle)

**Purpose**: Measures volatility and identifies potential breakouts or mean reversion opportunities. Price touching lower band may indicate oversold, upper band indicates overbought.

**ATR - Average True Range**
- `atr` - 14-period Average True Range

**Purpose**: Measures market volatility. Used for:
- Position sizing (risk-based quantity calculation)
- Stop-loss placement (ATR-based stop distance)
- Trailing stop management

### Oscillators (5 indicators)

**Stochastic Oscillator**
- `stoch_k` - %K (14-period)
- `stoch_d` - %D (3-period moving average of %K)

**Purpose**: Compares closing price to price range over a period. Values:
- < 20: Oversold (potential buy)
- > 80: Overbought (potential sell)

**Williams %R**
- `williams_r` - 14-period Williams %R

**Purpose**: Momentum indicator similar to Stochastic but inverted scale (-100 to 0). Values:
- < -80: Oversold
- > -20: Overbought

**CCI - Commodity Channel Index**
- `cci` - 20-period CCI

**Purpose**: Identifies cyclical trends. Values:
- > +100: Strong uptrend (potential sell)
- < -100: Strong downtrend (potential buy)

### Trend Strength Indicators (3 indicators)

**ADX - Average Directional Index**
- `adx` - 14-period ADX (trend strength, 0-100)
- `plus_di` - +DI (upward trend strength)
- `minus_di` - -DI (downward trend strength)

**Purpose**: Measures trend strength (not direction). ADX > 25 indicates strong trend, ADX < 20 indicates weak/choppy market.

---

## Trading Strategy

### Strategy Selection (Configurable)

The bot supports **three configurable trading strategies** selectable via `TRADING_STRATEGY` environment variable:

1. **`ma_crossover`** (Default) - Simple Moving Average Crossover
2. **`multi_indicator`** - Multi-Indicator Voting System
3. **`decision_support`** - Full Decision Support System with ML Integration

### Strategy 1: MA Crossover (Default)

The default strategy uses a **Moving Average Crossover** with trend following:

- **Fast MA**: 3-period SMA
- **Slow MA**: 9-period SMA

**Signal Generation Rules:**

1. **Buy Signal (Signal = 1)**
   - Fast MA crosses above Slow MA, OR
   - Fast MA is > 0.01% above Slow MA (trend following mode)
   - RSI confirmation: RSI <= RSI_BUY threshold (default: 100, effectively disabled)

2. **Sell Signal (Signal = -1)**
   - Fast MA crosses below Slow MA, OR
   - Fast MA is > 0.01% below Slow MA (trend following mode)
   - RSI confirmation: RSI >= RSI_SELL threshold (default: 0, effectively disabled)
   - Closes existing long position if one exists

### Trading Mode

**Current Mode: Long-Only (Alpaca Limitation)**
- Alpaca does not support crypto shorting
- Sell signals close long positions only
- Buy signals open long positions

**Position Management:**
- ATR-based position sizing
- Stop-loss protection
- Trailing stop management
- Full position closure on sell signals

### Strategy 2: Multi-Indicator

Uses a **voting system** that combines multiple technical indicators:

**Indicators Used:**
- RSI (oversold/overbought)
- MACD (bullish/bearish crossovers)
- Bollinger Bands (squeeze, breakouts)
- Stochastic (oversold/overbought)
- Williams %R (oversold/overbought)
- SMA crossovers (price vs SMA 20)

**Signal Generation:**
- Counts bullish signals from all indicators
- Counts bearish signals from all indicators
- Requires at least **2 indicators** to agree (configurable)
- Generates buy signal if `bullish_signals >= 2`
- Generates sell signal if `bearish_signals >= 2`

**Advantages:**
- More robust signals (consensus-based)
- Uses all calculated indicators
- Reduces false signals from single indicator noise

**Configuration:**
```env
TRADING_STRATEGY=multi_indicator
```

### Strategy 3: Decision Support

Full **Decision Support System** with ML integration (requires ML model):

**Components:**
- Technical indicators analysis
- ML predictions (if model available)
- Market regime analysis
- Risk assessment
- Confidence scoring

**Signal Generation:**
- Combines ML prediction, technical signals, market regime, and risk assessment
- Weighted scoring system
- Confidence-based signal generation
- Detailed reasoning and recommendations

**Requirements:**
- `enable_ml=true` in configuration
- Trained ML model available

**Configuration:**
```env
TRADING_STRATEGY=decision_support
ENABLE_ML=true
```

---

## Signal Generation

### Process Flow

1. **Data Fetching**
   - Fetches OHLCV data from Alpaca API
   - Default: 1000 candles (configurable via `OHLCV_LIMIT`)
   - Timeframe: 1 minute

2. **Indicator Calculation**
   - Calculates all 26 technical indicators
   - Uses latest price from live ticker (not stale OHLCV close)

3. **Signal Generation**
   - Calculates SMA(3) and SMA(9)
   - Determines crossover or trend following signal
   - Applies RSI filtering (if enabled)
   - Returns signal: 1 (buy), -1 (sell), or 0 (hold)

4. **Signal Execution**
   - Checks risk management constraints
   - Enforces cooldown period between trades
   - Calculates position size based on ATR and risk percentage
   - Executes trade via Alpaca API

### Signal Filtering

**RSI Filtering** (configurable via `DISABLE_RSI_FILTER`):
- When enabled: Buy signals require RSI <= threshold, sell signals require RSI >= threshold
- When disabled: All signals pass through

**Trend Following** (configurable via `TREND_FOLLOWING`):
- When enabled: Generates signals when MAs are separated (not just on crossovers)
- Threshold: 0.01% difference between Fast and Slow MA

---

## Trade Execution

### Buy Signal Execution

1. **Position Size Calculation**
   - Uses ATR-based risk management
   - Formula: `dollar_risk = equity × RISK_PER_TRADE`
   - Quantity: `qty = dollar_risk / (ATR_STOP_K × ATR)`
   - Capped at 98% of available cash (2% buffer for fees/slippage)

2. **Order Placement**
   - Market order via Alpaca API
   - Symbol normalization (handles BTC/USD vs BTCUSD)
   - Updates portfolio: cash decreases, position_qty increases

3. **Stop-Loss Initialization**
   - Initial stop: `entry_price - (ATR_STOP_K × ATR)`
   - Trailing stop managed during each iteration

### Sell Signal Execution

1. **Position Closure**
   - Fetches exact position quantity from Alpaca API
   - Attempts to use Alpaca's close position API (DELETE /v2/positions/{symbol})
   - Fallback: Market sell with exact quantity
   - Updates portfolio: position_qty = 0, cash increases

2. **No Position Handling**
   - If no long position exists, sell signal is skipped
   - Logs: "Skipping SELL signal: no position"

### Stop-Loss Management

**Trailing Stop Logic:**
- Long positions: Stop trails upward, never moves down
- Formula: `new_trail = current_price - (ATR_TRAIL_K × ATR)`
- Stop hit: Triggers immediate position closure

**Stop-Out Execution:**
- Fetches actual position from Alpaca
- Closes entire position via API or market order
- Logs trade as 'stop_out' type

---

## Risk Management

### Position Sizing

**ATR-Based Risk Management:**
- Risk per trade: Configurable percentage of equity (default: 1%)
- Stop distance: `ATR_STOP_K × ATR` (default: 2.0)
- Position size: Calculated to risk fixed dollar amount

**Cash Management:**
- Maximum position size: 98% of available cash
- 2% buffer for fees and slippage
- Additional 5% safety factor if needed

### Trade Limits

**Cooldown Period:**
- Minimum time between trades (default: 5 seconds)
- Configurable via `COOLDOWN_SEC` environment variable
- Prevents over-trading

**Daily Limits:**
- Monitored via RiskManager
- Tracks daily trade count and P&L
- Can halt trading if limits exceeded

---

## Architecture Components

### Core Services

**ContinuousTradingService** (`trading_engine/continuous_service.py`)
- Main trading loop
- Data fetching and signal generation
- Trade execution orchestration
- Portfolio management integration

**TechnicalIndicators** (`trading_engine/indicators.py`)
- Calculates all 26 technical indicators
- Provides indicator summaries
- Stores indicators in database

**DataFeeder** (`data/data_feeder.py`)
- Fetches OHLCV data from Alpaca API
- Handles symbol normalization
- Manages API rate limiting

**SimplePortfolio** (`trading_engine/simple_portfolio.py`)
- Tracks cash balance and positions
- Calculates equity and P&L
- Applies trade fills

**AlpacaExecutor** (`trading_engine/simple_executor.py`)
- Executes trades via Alpaca API
- Handles order placement and fills
- Manages symbol format conversions

### Database

**SQLite Database** (`data/db/trading_bot.db`)
- `trades` table: All executed trades
- `portfolio_metrics` table: Portfolio snapshots
- `indicators` table: Historical indicator values

### Backend API

**Flask REST API** (`backend/app.py`)
- Portfolio status endpoint (`/api/portfolio`)
- Positions endpoint (`/api/portfolio/positions`)
- Syncs with Alpaca API for real-time data
- Rate limiting and security

### Frontend

**React Dashboard**
- Real-time portfolio visualization
- Trade history display
- Performance metrics
- Chart visualizations

---

## Configuration

### Environment Variables

**Trading Strategy:**
- `RSI_BUY` - RSI threshold for buy signal confirmation (default: 100)
- `RSI_SELL` - RSI threshold for sell signal confirmation (default: 0)
- `RISK_PER_TRADE` - Risk percentage per trade (default: 0.01 = 1%)
- `ATR_STOP_K` - ATR multiplier for stop distance (default: 2.0)
- `ATR_TRAIL_K` - ATR multiplier for trailing stop (default: 2.5)
- `COOLDOWN_SEC` - Seconds between trades (default: 5)
- `OHLCV_LIMIT` - Number of candles to fetch (default: 1000)

**Trading Mode:**
- `SELL_MODE` - long_only (default) or long_short
- `ALLOW_SHORTING` - Enable shorting (default: false)
- `IMMEDIATE_POSITION_FLIP` - Close and open opposite position in same iteration (default: true)

**Strategy Selection:**
- `TRADING_STRATEGY` - ma_crossover (default), multi_indicator, or decision_support

**Data & Execution:**
- `USE_ALPACA_EXECUTOR` - Use real Alpaca executor (default: false)
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key

**Testing/Debug:**
- `DISABLE_RSI_FILTER` - Disable RSI filtering (default: false)
- `TREND_FOLLOWING` - Enable trend following mode (default: true)

### Configuration Files

- `.env` - Root environment variables
- `deployment/.env` - Docker deployment environment variables
- `deployment/docker-compose.yml` - Docker services configuration

---

## System Flow

```
┌─────────────────┐
│  Start Service  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fetch OHLCV    │
│  Data (Alpaca)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculate 26   │
│  Indicators     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │
│  Signal (MA)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RSI Filter?    │
│  (if enabled)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Risk Check     │
│  Cooldown Check │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculate      │
│  Position Size  │
│  (ATR-based)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execute Trade  │
│  (Alpaca API)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update         │
│  Portfolio      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Log Trade      │
│  (Database)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Check Stop-Loss│
│  (Trailing)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Wait Interval  │
│  (15 seconds)   │
└────────┬────────┘
         │
         └─────────────┐
                       │
                       ▼
              (Repeat Loop)
```

---

## Notes

- **Paper Trading**: All trades execute on Alpaca's paper trading environment
- **No Shorting**: Alpaca does not support crypto shorting; system is long-only
- **Real-time Sync**: Portfolio values sync directly from Alpaca API
- **Precision**: Uses 9 decimal places for BTC quantities to avoid rounding errors
- **Position Closure**: Attempts to close entire position using Alpaca's close position API

---

*Last Updated: 2025-10-31*
*System Version: CryptoTradingBot v1.0*

