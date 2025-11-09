-- Crypto Trading Bot Database Schema
-- SQLite database for storing trading data, portfolio state, and ML predictions

-- Historical price data table
CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '1m',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '1m',
    rsi REAL,
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    ema_12 REAL,
    ema_26 REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    bb_width REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
);

-- ML model predictions table
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction REAL NOT NULL,
    confidence REAL NOT NULL,
    features TEXT, -- JSON string of features used
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL,
    current_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    realized_pnl REAL DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol)
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop')),
    quantity REAL NOT NULL,
    price REAL,
    stop_price REAL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    filled_quantity REAL DEFAULT 0,
    filled_price REAL,
    commission REAL DEFAULT 0,
    strategy TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Trades table (filled orders)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    net_amount REAL NOT NULL,
    timestamp TEXT NOT NULL,
    strategy TEXT,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

-- Portfolio performance metrics
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash_balance REAL NOT NULL,
    invested_value REAL NOT NULL,
    total_pnl REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    total_return REAL NOT NULL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Strategy configuration
CREATE TABLE IF NOT EXISTS strategy_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT UNIQUE NOT NULL,
    config TEXT NOT NULL, -- JSON string of strategy parameters
    is_active BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    module TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    data TEXT -- JSON string for additional data
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_price_data_symbol_timestamp ON price_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_indicators_symbol_timestamp ON indicators(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_timestamp ON ml_predictions(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp ON portfolio_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_level_timestamp ON system_logs(level, timestamp);
