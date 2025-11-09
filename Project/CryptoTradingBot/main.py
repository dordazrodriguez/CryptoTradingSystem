"""Main entry point for the Cryptocurrency Trading Bot Simulator."""

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
from data.data_feeder import DataFeed, FeedConfig, AlpacaFeed, AlpacaConfig, sma, rsi, normalize_for_provider
from trading_engine.simple_strategy import MovingAverageCrossover, CrossoverConfig
from trading_engine.simple_portfolio import SimplePortfolio
from trading_engine.simple_executor import SimulatorExecutor, ExecutionResult, AlpacaExecutor
from data.db import get_db_manager
from trading_engine.indicators import TechnicalIndicators
from trading_engine.portfolio import PortfolioManager
from data.processor import DataProcessor
from trading_engine.continuous_service import ContinuousTradingService
import os
from dotenv import load_dotenv


def demo_strategy(
    symbol: str = "BTC/USD",
    exchange: str = "alpaca",
    timeframe: str = "1m",
    provider: str = "alpaca",
):
    """Demonstrate the trading strategy with live data."""
    logger.info("=" * 60)
    logger.info("CryptoTradingBot - Strategy Demonstration")
    logger.info("=" * 60)
    
    if provider == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            logger.error("ALPACA_API_KEY/ALPACA_SECRET_KEY not set. Create a .env with your keys or export them.")
            return
        feed_config = AlpacaConfig(symbol=symbol, api_key=api_key, api_secret=api_secret, timeframe=timeframe, limit=100)
        feed = AlpacaFeed(feed_config)
    else:
        feed_config = FeedConfig(exchange=exchange, symbol=symbol, timeframe=timeframe, limit=100)
        feed = DataFeed(feed_config)
    
    logger.info(f"Fetching data for {symbol} from {exchange}...")
    df = feed.fetch_ohlcv()
    
    if df.empty:
        logger.error("Failed to fetch market data")
        return
    
    logger.info(f"Fetched {len(df)} candles")
    
    logger.info("\nCalculating technical indicators...")
    df['sma_12'] = sma(df, 12)
    df['sma_26'] = sma(df, 26)
    df['rsi_14'] = rsi(df, 14)
    
    strategy = MovingAverageCrossover(CrossoverConfig(fast=12, slow=26))
    df['signal'] = strategy.generate(df, 'sma_12', 'sma_26')
    
    logger.info("\nStrategy Analysis:")
    logger.info(f"Buy signals: {(df['signal'] == 1).sum()}")
    logger.info(f"Sell signals: {(df['signal'] == -1).sum()}")
    
    display_cols = ['ts', 'close', 'sma_12', 'sma_26', 'rsi_14', 'signal']
    logger.info("\nLatest market data and signals:")
    print(df[display_cols].tail(10).to_string())
    
    logger.info("\n\nCalculating comprehensive technical indicators...")
    indicators = TechnicalIndicators()
    indicators_df = indicators.calculate_all_indicators(df)
    
    logger.info(f"\nCalculated {len(indicators_df.columns) - len(df.columns)} additional indicators")
    logger.info("\nIndicator summary:")
    logger.info(f"RSI: {df['rsi_14'].iloc[-1]:.2f}")
    logger.info(f"SMA 12: {df['sma_12'].iloc[-1]:.2f}")
    logger.info(f"SMA 26: {df['sma_26'].iloc[-1]:.2f}")
    logger.info(f"Current price: {df['close'].iloc[-1]:.2f}")
    
    signals_df = indicators.get_trading_signals(indicators_df)
    logger.info(f"\nBullish signals: {signals_df['bullish_signals'].iloc[-1]}")
    logger.info(f"Bearish signals: {signals_df['bearish_signals'].iloc[-1]}")
    logger.info(f"Signal strength: {signals_df['signal_strength'].iloc[-1]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


def run_simulation(
    symbol: str = "BTC/USD",
    exchange: str = "alpaca",
    timeframe: str = "1m",
    starting_cash: float = 10000.0,
    provider: str = "alpaca",
):
    """Run a complete trading simulation."""
    logger.info("=" * 60)
    logger.info("CryptoTradingBot - Trading Simulation")
    logger.info("=" * 60)
    
    if provider == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            logger.error("ALPACA_API_KEY/ALPACA_SECRET_KEY not set. Create a .env with your keys or export them.")
            return
        feed_config = AlpacaConfig(symbol=symbol, api_key=api_key, api_secret=api_secret, timeframe=timeframe, limit=100)
        feed = AlpacaFeed(feed_config)
    else:
        feed_config = FeedConfig(exchange=exchange, symbol=symbol, timeframe=timeframe, limit=100)
        feed = DataFeed(feed_config)
    portfolio = SimplePortfolio(cash_usd=starting_cash)
    executor = SimulatorExecutor(fee_bps=10.0)
    strategy = MovingAverageCrossover(CrossoverConfig(fast=12, slow=26))
    
    # Fetch data
    logger.info(f"Fetching market data for {symbol}...")
    df = feed.fetch_ohlcv()
    
    if df.empty:
        logger.error("Failed to fetch market data")
        return
    
    # Calculate indicators and signals
    df['sma_12'] = sma(df, 12)
    df['sma_26'] = sma(df, 26)
    df['signal'] = strategy.generate(df, 'sma_12', 'sma_26')
    
    # Track performance
    trades = []
    equity_history = []
    
    # Process signals
    for idx, row in df.iterrows():
        current_price = row['close']
        signal = row['signal']
        timestamp = row['ts']
        
        equity = portfolio.equity(current_price)
        equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'price': current_price
        })
        
        if signal == 1:
            if portfolio.cash_usd > 100:
                risk_amount = equity * 0.01
                fee_rate = executor.fee_bps / 10000.0
                notional = risk_amount / (1 + fee_rate)
                
                try:
                    result = executor.market('buy', current_price, notional)
                    if portfolio.can_afford(current_price, result.qty, result.fee):
                        portfolio.apply_fill('buy', current_price, result.qty, result.fee)
                        trades.append({
                            'timestamp': timestamp,
                            'side': 'buy',
                            'price': current_price,
                            'qty': result.qty,
                            'fee': result.fee
                        })
                        logger.info(f"BUY: {result.qty:.6f} @ ${current_price:.2f}")
                except Exception as e:
                    logger.error(f"Trade execution error: {e}")
        
        elif signal == -1:
            if portfolio.position_qty > 0:
                notional = portfolio.position_qty * current_price
                result = executor.market('sell', current_price, notional)
                portfolio.apply_fill('sell', current_price, result.qty, result.fee)
                trades.append({
                    'timestamp': timestamp,
                    'side': 'sell',
                    'price': current_price,
                    'qty': result.qty,
                    'fee': result.fee
                })
                logger.info(f"SELL: {result.qty:.6f} @ ${current_price:.2f}")
    
    final_price = df['close'].iloc[-1]
    final_equity = portfolio.equity(final_price)
    total_return = (final_equity - starting_cash) / starting_cash * 100
    
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Starting cash: ${starting_cash:,.2f}")
    logger.info(f"Final equity: ${final_equity:,.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    logger.info(f"Total trades: {len(trades)}")
    logger.info(f"Positions: {portfolio.position_qty:.6f}")
    logger.info(f"Cash remaining: ${portfolio.cash_usd:,.2f}")
    logger.info("=" * 60)


def run_continuous_trading(symbol: str = "BTC/USD", exchange: str = "alpaca",
                           timeframe: str = "1m", starting_cash: float = 10000.0,
                           provider: str = "alpaca", interval: int = 60,
                           enable_ml: bool = False, model_path: str = "ml_models/trained_model.pkl",
                           enable_auto_retraining: bool = False):
    """Run continuous 24/7 trading service."""
    trading_logger = None
    local_logger = logger
    try:
        from core.logger import TradingLogger
        trading_logger = TradingLogger(config={})
        local_logger = trading_logger.logger
    except ImportError:
        trading_logger = None
        pass
    except Exception as e:
        trading_logger = None
        local_logger = logger
        local_logger.warning(f"Could not initialize TradingLogger: {e}, using standard logging")
    
    logger_ref = local_logger
    
    logger_ref.info("=" * 60)
    logger_ref.info("CryptoTradingBot - Continuous Trading Service")
    logger_ref.info("=" * 60)
    
    # Get model path from environment variable if not provided
    if not model_path or model_path == "ml_models/trained_model.pkl":
        model_path = os.getenv('ML_MODEL_PATH', model_path)
    
    # Get enable_ml from environment if not provided via command line
    if not enable_ml:
        enable_ml = os.getenv('ENABLE_ML', '').lower() in ('true', '1', 'yes')
    
    # Get enable_auto_retraining from environment if not provided via command line
    if not enable_auto_retraining:
        enable_auto_retraining = os.getenv('ENABLE_AUTO_RETRAINING', '').lower() in ('true', '1', 'yes')
    
    # Load YAML config if available
    try:
        from config.config_loader import load_config
        yaml_config = load_config()
        logger_ref.info("Loaded configuration from config/config.yaml")
    except ImportError:
        yaml_config = {}
        logger_ref.info("YAML config loader not available, using environment variables and defaults")
    except Exception as e:
        yaml_config = {}
        logger_ref.warning(f"Could not load YAML config: {e}, using environment variables and defaults")
    
    trading_strategy = (
        yaml_config.get('strategy') or 
        os.getenv('TRADING_STRATEGY') or 
        yaml_config.get('trading_strategy', 'ma_crossover')
    ).lower()
    
    config = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'starting_cash': starting_cash,
        'provider': provider,
        'interval': interval,
        'enable_ml': enable_ml,
        'enable_auto_retraining': enable_auto_retraining,
        'model_path': model_path,
        'max_position_size': 0.01,  # 1% of equity per trade
        'stop_loss_pct': 0.05,  # 5% stop loss
        'retrain_interval_days': int(os.getenv('RETRAIN_INTERVAL_DAYS', '7')),
        'min_accuracy_threshold': float(os.getenv('MIN_ACCURACY_THRESHOLD', '0.48')),
        'retraining_check_hours': int(os.getenv('RETRAINING_CHECK_HOURS', '24'))
    }
    
    if yaml_config:
        for key in ['trading', 'risk', 'ppo', 'reward', 'features', 'ml', 'experience_storage', 'checkpoints', 'logging']:
            if key in yaml_config:
                if key not in config:
                    config[key] = {}
                if isinstance(config[key], dict) and isinstance(yaml_config[key], dict):
                    config[key].update(yaml_config[key])
                else:
                    config[key] = yaml_config[key]
    
    if trading_strategy == 'ppo_rl':
        ppo_defaults = yaml_config.get('ppo', {}) if yaml_config else {}
        config['ppo'] = {
            'gamma': float(os.getenv('PPO_GAMMA', str(ppo_defaults.get('gamma', 0.99)))),
            'clip_epsilon': float(os.getenv('PPO_CLIP_EPSILON', str(ppo_defaults.get('clip_epsilon', 0.2)))),
            'learning_rate': float(os.getenv('PPO_LEARNING_RATE', str(ppo_defaults.get('learning_rate', 3e-4)))),
            'update_epochs': int(os.getenv('PPO_UPDATE_EPOCHS', str(ppo_defaults.get('update_epochs', 10)))),
            'batch_size': int(os.getenv('PPO_BATCH_SIZE', str(ppo_defaults.get('batch_size', 64)))),
            'update_interval': int(os.getenv('PPO_UPDATE_INTERVAL', str(ppo_defaults.get('update_interval', 100)))),
            'value_coef': float(os.getenv('PPO_VALUE_COEF', str(ppo_defaults.get('value_coef', 0.5)))),
            'entropy_coef': float(os.getenv('PPO_ENTROPY_COEF', str(ppo_defaults.get('entropy_coef', 0.01)))),
            'use_gae': os.getenv('PPO_USE_GAE', str(ppo_defaults.get('use_gae', True))).lower() in ('true', '1', 'yes'),
            'gae_lambda': float(os.getenv('PPO_GAE_LAMBDA', str(ppo_defaults.get('gae_lambda', 0.95)))),
            'action_mode': os.getenv('PPO_ACTION_MODE', ppo_defaults.get('action_mode', 'discrete')),
            'network': {
                'hidden_layers': int(os.getenv('PPO_HIDDEN_LAYERS', str(ppo_defaults.get('network', {}).get('hidden_layers', 2)))),
                'hidden_size': int(os.getenv('PPO_HIDDEN_SIZE', str(ppo_defaults.get('network', {}).get('hidden_size', 128)))),
                'activation': os.getenv('PPO_ACTIVATION', ppo_defaults.get('network', {}).get('activation', 'relu'))
            }
        }
        config['ppo_model_path'] = os.getenv('PPO_MODEL_PATH')
        config['ppo_model_dir'] = os.getenv('PPO_MODEL_DIR', yaml_config.get('checkpoints', {}).get('save_dir', 'models') if yaml_config else 'models')
        reward_defaults = yaml_config.get('reward', {}) if yaml_config else {}
        config['reward'] = {
            'transaction_cost_pct': float(os.getenv('REWARD_TRANSACTION_COST', str(reward_defaults.get('transaction_cost_pct', 0.001)))),
            'volatility_penalty': os.getenv('REWARD_VOLATILITY_PENALTY', str(reward_defaults.get('volatility_penalty', False))).lower() in ('true', '1', 'yes')
        }
        features_defaults = yaml_config.get('features', {}) if yaml_config else {}
        config['features'] = {
            'lookback_window': int(os.getenv('FEATURE_LOOKBACK_WINDOW', str(features_defaults.get('lookback_window', 100)))),
            'normalization_method': os.getenv('FEATURE_NORMALIZATION', features_defaults.get('normalization_method', 'z-score'))
        }
        exp_storage_defaults = yaml_config.get('experience_storage', {}) if yaml_config else {}
        config['experience_storage'] = {
            'enabled': os.getenv('EXPERIENCE_STORAGE_ENABLED', str(exp_storage_defaults.get('enabled', False))).lower() in ('true', '1', 'yes'),
            'storage_dir': os.getenv('EXPERIENCE_STORAGE_DIR', exp_storage_defaults.get('storage_dir', 'data/experiences')),
            'max_experiences_to_load': int(os.getenv('MAX_EXPERIENCES_TO_LOAD', str(exp_storage_defaults.get('max_experiences_to_load', 50000))))
        }
        logger_ref.info("✅ PPO/RL configuration loaded")
    
    if provider == "alpaca":
        config['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
        config['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
    
    service = ContinuousTradingService(config, trading_logger=trading_logger)
    service.run()


def main():
    """Main entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="CryptoTradingBot - Capstone Project")
    parser.add_argument("--mode", choices=["demo", "simulate", "run"], default="demo",
                       help="Run mode: demo, simulate, or run (continuous)")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--exchange", default="alpaca", help="Exchange name")
    parser.add_argument("--timeframe", default="1m", help="Data timeframe")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash")
    parser.add_argument("--provider", choices=["ccxt", "alpaca"], default="alpaca",
                       help="Data provider (ccxt or alpaca)")
    parser.add_argument("--interval", type=int, default=60,
                       help="Update interval in seconds for continuous mode")
    parser.add_argument("--enable-ml", action="store_true",
                       help="Enable ML predictions (requires trained model)")
    parser.add_argument("--enable-auto-retraining", action="store_true",
                       help="Enable automatic model retraining (requires ML enabled)")
    parser.add_argument("--model-path", default="ml_models/trained_model.pkl",
                       help="Path to ML model file")
    # Test order options
    parser.add_argument("--test-order", action="store_true", help="Place a small Alpaca paper trade and exit")
    parser.add_argument("--test-side", choices=["buy", "sell"], default="buy", help="Test order side")
    parser.add_argument("--test-notional", type=float, default=10.0, help="USD notional for test order")
    
    args = parser.parse_args()
    
    if args.test_order:
        # Quick end-to-end test of Alpaca execution
        if args.provider != "alpaca":
            logger.error("--test-order requires --provider alpaca")
            return
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            logger.error("ALPACA_API_KEY/ALPACA_SECRET_KEY not set.")
            return
        # Normalize trading symbol for Alpaca
        trade_symbol = normalize_for_provider("alpaca", args.symbol, use_for="trading")
        executor = AlpacaExecutor(api_key=api_key, api_secret=api_secret, sandbox=True, fee_bps=10.0)
        # Fetch price to compute qty
        feed = AlpacaFeed(AlpacaConfig(symbol=normalize_for_provider("alpaca", args.symbol, use_for="data"), api_key=api_key, api_secret=api_secret, timeframe=args.timeframe, limit=1))
        df = feed.fetch_ohlcv()
        if df.empty:
            logger.error("Failed to fetch price for test order")
            return
        price = float(df['close'].iloc[-1])
        res = executor.market(args.test_side, price, args.test_notional, symbol=trade_symbol)
        logger.info(f"TEST {args.test_side.upper()} placed: qty={res.qty:.6f} {trade_symbol} @ ~${price:.2f}, fee=${res.fee:.4f}")
        return
    elif args.mode == "demo":
        demo_strategy(args.symbol, args.exchange, args.timeframe, provider=args.provider)
    elif args.mode == "simulate":
        run_simulation(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            starting_cash=args.cash,
            provider=args.provider,
        )
    elif args.mode == "run":
            run_continuous_trading(
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                starting_cash=args.cash,
                provider=args.provider,
                interval=args.interval,
                enable_ml=args.enable_ml,
                enable_auto_retraining=args.enable_auto_retraining,
                model_path=args.model_path
            )


if __name__ == "__main__":
    main()
