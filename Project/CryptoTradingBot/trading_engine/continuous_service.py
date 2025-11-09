"""
Continuous trading service for 24/7 operation.
Handles live trading loops, signal processing, and trade execution.
"""

import logging
import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
import pandas as pd
import random
import requests

from data.data_feeder import DataFeed, FeedConfig, AlpacaFeed, AlpacaConfig, normalize_for_provider
from trading_engine.simple_strategy import MovingAverageCrossover, CrossoverConfig
from trading_engine.simple_portfolio import SimplePortfolio
from trading_engine.simple_executor import SimulatorExecutor, AlpacaExecutor
from trading_engine.indicators import TechnicalIndicators
from trading_engine.decision_support import DecisionSupportSystem
from trading_engine.risk_manager import RiskManager
from trading_engine.portfolio import PortfolioManager
from data.db import get_db_manager
from ml_models.predictor import CryptoPredictionModel
from ml_models.retraining_service import ModelRetrainingService
from trading_engine.rl.ppo_agent_rl import PPOAgent
from trading_engine.rl.rewards.reward_calculator import RewardCalculator
from trading_engine.rl.experience.experience_storage import PersistentExperienceStorage
from trading_engine.rl.checkpoints.checkpoint_manager import CheckpointManager
from pathlib import Path
from ml_models.rl_feature_pipeline import RLFeaturePipeline
import os
import numpy as np

logger = logging.getLogger(__name__)


class ContinuousTradingService:
    """
    Continuous trading service that runs 24/7.
    Fetches data, generates signals, and executes trades automatically.
    """
    
    def __init__(self, config: Dict[str, Any], trading_logger=None):
        """
        Initialize continuous trading service.
        
        Args:
            config: Configuration dictionary
            trading_logger: Optional TradingLogger instance for enhanced logging
        """
        # Store trading logger for enhanced logging
        self.trading_logger = trading_logger
        
        """
        Initialize continuous trading service.
        
        Args:
            config: Configuration dictionary with:
                - symbol: Trading symbol (e.g., "BTC/USDT")
                - exchange: Exchange name (e.g., "binance")
                - timeframe: Data timeframe (e.g., "1m", "5m")
                - starting_cash: Starting cash balance
                - provider: Data provider ("ccxt" or "alpaca")
                - alpaca_api_key: Alpaca API key (if using Alpaca)
                - alpaca_secret_key: Alpaca secret key (if using Alpaca)
                - interval: Update interval in seconds
                - enable_ml: Whether to use ML predictions
                - model_path: Path to ML model file
                - max_position_size: Maximum position size percentage
                - stop_loss_pct: Stop loss percentage
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USD')
        self.exchange = config.get('exchange', 'binance')
        self.timeframe = config.get('timeframe', '1m')
        self.provider = config.get('provider', 'alpaca')
        # Normalized symbols for data vs trading (must be after provider is set)
        self.data_symbol = normalize_for_provider(self.provider, self.symbol, use_for="data")
        self.trade_symbol = normalize_for_provider(self.provider, self.symbol, use_for="trading")
        self.starting_cash = config.get('starting_cash', 10000.0)
        self.interval = config.get('interval', 60)  # Default 1 minute
        # Enable ML from config or environment variable (env takes precedence)
        enable_ml_env = os.getenv('ENABLE_ML', '').lower() in ('true', '1', 'yes')
        self.enable_ml = config.get('enable_ml', enable_ml_env)
        # OHLCV data fetch limit (default: 1000 candles)
        self.ohlcv_limit = int(os.getenv('OHLCV_LIMIT', str(config.get('ohlcv_limit', 1000))))

        # Strategy/risk parameters (env-overridable)
        self.rsi_buy = float(os.getenv('RSI_BUY', str(config.get('rsi_buy', 55))))
        self.rsi_sell = float(os.getenv('RSI_SELL', str(config.get('rsi_sell', 45))))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', str(config.get('risk_per_trade', 0.005))))
        self.atr_stop_k = float(os.getenv('ATR_STOP_K', str(config.get('atr_stop_k', 2.0))))
        self.atr_trail_k = float(os.getenv('ATR_TRAIL_K', str(config.get('atr_trail_k', 2.5))))
        self.cooldown_sec = int(os.getenv('COOLDOWN_SEC', str(config.get('cooldown_sec', 30))))
        # Test mode: disable RSI filtering (for debugging)
        self.disable_rsi_filter = os.getenv('DISABLE_RSI_FILTER', 'false').lower() in ('1', 'true', 'yes')
        # Trend following mode: generate signals when MAs are separated (not just crossovers)
        self.trend_following = os.getenv('TREND_FOLLOWING', 'false').lower() in ('1', 'true', 'yes')
        
        # Profit protection configuration
        self.protect_profits = os.getenv('PROTECT_PROFITS', 'false').lower() in ('1', 'true', 'yes')
        try:
            self.min_profit_target = float(os.getenv('MIN_PROFIT_TARGET', '0.0'))  # e.g., 0.002 = 0.2%
        except Exception:
            self.min_profit_target = 0.0
        if self.protect_profits:
            logger.info(f"Profit protection enabled: MIN_PROFIT_TARGET={self.min_profit_target * 100:.3f}%")
        
        # Trading strategy selection
        # Options: ma_crossover (default), multi_indicator, decision_support, ppo_rl
        # Check config dict first, then environment variable, then default
        self.trading_strategy = (
            config.get('strategy') or 
            config.get('trading_strategy') or 
            os.getenv('TRADING_STRATEGY', 'ma_crossover')
        ).lower()
        if self.trading_strategy not in ['ma_crossover', 'multi_indicator', 'decision_support', 'ppo_rl']:
            logger.warning(f"Invalid TRADING_STRATEGY={self.trading_strategy}, defaulting to 'ma_crossover'")
            self.trading_strategy = 'ma_crossover'
        logger.info(f"Trading strategy: {self.trading_strategy}")
        
        # Log PPO strategy banner if enabled
        if self.trading_strategy == 'ppo_rl':
            logger.info("=" * 60)
            logger.info("PPO REINFORCEMENT LEARNING STRATEGY ENABLED")
            logger.info("=" * 60)
        
        # Initialize components
        self.portfolio = SimplePortfolio(cash_usd=self.starting_cash)
        self.executor = SimulatorExecutor(fee_bps=10.0)
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager()
        self.db_manager = get_db_manager()
        
        # Initialize data feed
        feed_symbol = self.data_symbol
        if self.provider == "alpaca":
            api_key = config.get('alpaca_api_key')
            secret_key = config.get('alpaca_secret_key')
            if not api_key or not secret_key:
                raise ValueError("Alpaca API keys required when using Alpaca provider")
            self.data_feed = AlpacaFeed(AlpacaConfig(
                symbol=feed_symbol,
                timeframe=self.timeframe,
                limit=self.ohlcv_limit,
                api_key=api_key,
                api_secret=secret_key
            ))
            # Optionally use real Alpaca executor for paper trading
            use_alpaca_executor = bool(str(config.get('use_alpaca_executor', os.getenv('USE_ALPACA_EXECUTOR', 'false'))).lower() in ('1','true','yes'))
            if use_alpaca_executor:
                self.executor = AlpacaExecutor(api_key=api_key, api_secret=secret_key, sandbox=True, fee_bps=10.0)
            # Sync starting cash and positions from Alpaca paper account if available
            try:
                bal = self.data_feed.exchange.fetch_balance()
                # Prefer USD free cash; fall back to total
                usd = bal.get('USD') or {}
                cash = usd.get('free') or usd.get('total') or bal.get('free', {}).get('USD') or bal.get('total', {}).get('USD')
                if isinstance(cash, (int, float)) and cash > 0:
                    self.portfolio.cash_usd = float(cash)
                    # Align starting cash and performance baseline with Alpaca balance
                    self.starting_cash = self.portfolio.cash_usd
                    logger.info(f"Initialized portfolio cash from Alpaca: ${self.portfolio.cash_usd:,.2f}")
                
                # Sync positions from Alpaca using REST API (more direct than parsing balance)
                try:
                    # Use Alpaca's REST API directly to get positions
                    base_url = "https://paper-api.alpaca.markets"
                    headers = {
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": secret_key
                    }
                    
                    # Get all positions from Alpaca
                    response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        positions = response.json()
                        crypto_symbol_base = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.replace('USD', '')
                        
                        for pos in positions:
                            symbol = pos.get('symbol', '')
                            qty_str = pos.get('qty', '0')
                            qty = float(qty_str) if qty_str else 0.0
                            
                            # Check if this position matches our crypto symbol (e.g., BTCUSD, BTC/USD, etc.)
                            if qty != 0 and crypto_symbol_base.upper() in symbol.upper():
                                avg_price = float(pos.get('avg_entry_price', 0)) if pos.get('avg_entry_price') else None
                                
                                self.portfolio.position_qty = abs(qty)  # Use absolute value
                                self.portfolio.position_avg_price = avg_price
                                
                                # Format average price string (handle None case)
                                avg_price_str = f"${avg_price:.2f}" if avg_price is not None else "N/A"
                                logger.info(f"✅ Synced position from Alpaca API: {self.portfolio.position_qty:.9f} {symbol} @ avg {avg_price_str}")
                                break
                        else:
                            logger.info(f"No open positions found for {crypto_symbol_base}")
                    elif response.status_code == 401:
                        logger.warning("⚠️ Alpaca API authentication failed - check API keys")
                    else:
                        logger.warning(f"⚠️ Alpaca positions API returned status {response.status_code}: {response.text[:200]}")
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Could not fetch positions from Alpaca REST API: {e}")
                except Exception as e:
                    logger.warning(f"Error syncing positions from Alpaca: {e}")
            except Exception as e:
                logger.warning(f"Could not fetch Alpaca balance/positions: {e}")
        else:
            feed_config = FeedConfig(
                exchange=self.exchange,
                symbol=feed_symbol,
                timeframe=self.timeframe,
                limit=self.ohlcv_limit
            )
            self.data_feed = DataFeed(feed_config)
        
        # Initialize strategy
        # Use tighter windows to increase signal frequency
        self.strategy = MovingAverageCrossover(CrossoverConfig(fast=3, slow=9))
        
        # Initialize decision support system
        self.decision_support = DecisionSupportSystem(risk_manager=self.risk_manager)
        
        # Initialize PPO/RL components if using PPO strategy
        self.ppo_agent = None
        self.reward_calculator = None
        self.feature_pipeline = None
        self.experience_storage = None
        self.ppo_config = None
        self.ppo_step_count = 0
        self.last_action = None
        self.last_state = None
        self.last_portfolio_value = None
        self.last_position_size = 0.0
        self._last_action_log_prob = 0.0
        self.checkpoint_manager = None  # Initialized in PPO setup
        self.shutdown_requested = False
        
        if self.trading_strategy == 'ppo_rl':
            logger.info("Initializing PPO/RL components...")
            try:
                # Load PPO configuration from config dict or defaults
                self.ppo_config = config.get('ppo', {})
                
                # Initialize reward calculator
                self.reward_calculator = RewardCalculator(config={
                    'reward': config.get('reward', {
                        'transaction_cost_pct': 0.001,
                        'volatility_penalty': False
                    }),
                    'trading': config.get('trading', {
                        'short_selling_enabled': False  # Crypto is long-only
                    })
                })
                
                # Initialize feature pipeline
                self.feature_pipeline = RLFeaturePipeline(config={
                    'features': config.get('features', {
                        'lookback_window': 100,
                        'normalization_method': 'z-score'
                    })
                })
                
                # ML predictor will be set later after ML model loads
                # (can't check self.ml_model here as it's initialized after PPO)
                
                # Initialize experience storage (optional, for persistent memory)
                experience_storage_enabled = config.get('experience_storage', {}).get('enabled', False)
                if experience_storage_enabled:
                    storage_dir = config.get('experience_storage', {}).get('storage_dir', 'data/experiences')
                    self.experience_storage = PersistentExperienceStorage(storage_dir=storage_dir)
                    logger.info(f"Experience storage enabled: {storage_dir}")
                
                # Initialize checkpoint manager
                checkpoint_config = config.get('checkpoints', {})
                checkpoint_dir = Path(checkpoint_config.get('save_dir', 'models'))
                keep_last_n = checkpoint_config.get('keep_last_n', 5)
                self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir, keep_last_n=keep_last_n)
                logger.info(f"Checkpoint manager initialized: keeping last {keep_last_n} checkpoints in {checkpoint_dir}")
                
                # PPO agent will be initialized after first feature computation (need state_dim)
                logger.info("PPO components initialized (agent will be created on first signal)")
                
            except Exception as e:
                logger.error(f"Failed to initialize PPO components: {e}")
                logger.error("Falling back to ma_crossover strategy")
                self.trading_strategy = 'ma_crossover'
        
        # Initialize ML model if enabled
        self.ml_model = None
        if self.enable_ml:
            try:
                # Get model path from config, environment variable, or default
                model_path = config.get('model_path') or os.getenv('ML_MODEL_PATH', 'ml_models/trained_model.pkl')
                if os.path.exists(model_path):
                    # Suppress initialization log since we're loading an existing model
                    self.ml_model = CryptoPredictionModel(suppress_init_log=True)
                    self.ml_model.load_model(model_path)
                    self.decision_support.ml_model = self.ml_model
                    logger.info(f"ML model loaded from {model_path} (algorithm: {getattr(self.ml_model, 'algorithm', 'unknown')})")
                    
                    # Connect ML model to PPO feature pipeline if using PPO strategy
                    if self.trading_strategy == 'ppo_rl' and self.feature_pipeline:
                        self.feature_pipeline.set_ml_predictor(self.ml_model)
                        logger.info("✅ ML model connected to RL feature pipeline")
                else:
                    logger.warning(f"ML model not found at {model_path}, running without ML")
                    self.enable_ml = False
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}, running without ML")
                self.enable_ml = False
        
        # Auto-retraining configuration
        self.enable_auto_retraining = config.get('enable_auto_retraining', False)
        self.retraining_service = None
        if self.enable_ml and self.enable_auto_retraining:
            retraining_config = {
                'model_dir': config.get('model_dir', 'ml_models'),
                'min_accuracy_threshold': config.get('min_accuracy_threshold', 0.48),
                'performance_window_days': config.get('performance_window_days', 7),
                'retrain_interval_days': config.get('retrain_interval_days', 7),
                'min_training_samples': config.get('min_training_samples', 500),
                'symbol': self.symbol,
                'exchange': self.exchange,
                'provider': self.provider,
                'alpaca_api_key': config.get('alpaca_api_key') or os.getenv('ALPACA_API_KEY'),
                'alpaca_secret_key': config.get('alpaca_secret_key') or os.getenv('ALPACA_SECRET_KEY')
            }
            try:
                self.retraining_service = ModelRetrainingService(retraining_config)
                logger.info("✅ Auto-retraining service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize auto-retraining service: {e}")
                self.enable_auto_retraining = False
        
        # Track state
        self.last_signal_ts = None
        self.last_price = None
        self.backoff_seconds = self.interval
        self.iteration_count = 0
        self._active_stop: Optional[float] = None
        self._last_trade_ts: Optional[float] = None
        self._last_atr: Optional[float] = None
        self._last_retraining_check: Optional[datetime] = None
        self._retraining_check_interval = timedelta(hours=config.get('retraining_check_hours', 24))
        
        # Signal handlers will be registered in run() method
        
        logger.info(f"Continuous trading service initialized for {self.symbol}")
        logger.info(f"Provider: {self.provider}, Timeframe: {self.timeframe}, Interval: {self.interval}s")
        # Report actual portfolio cash rather than default starting cash
        logger.info(f"Starting cash: ${self.portfolio.cash_usd:,.2f}")
        logger.info(f"ML enabled: {self.enable_ml}")
        if self.enable_ml:
            logger.info(f"Auto-retraining enabled: {self.enable_auto_retraining}")
            if self.enable_auto_retraining and self.retraining_service:
                logger.info(f"  - Retrain interval: {self.retraining_service.retrain_interval_days} days")
                logger.info(f"  - Min accuracy threshold: {self.retraining_service.min_accuracy_threshold}")
        logger.info(f"OHLCV fetch limit: {self.ohlcv_limit} candles")
        logger.info(f"RSI thresholds: BUY<= {self.rsi_buy}, SELL>= {self.rsi_sell}")
        logger.info(f"Trend following: {self.trend_following}, RSI filter disabled: {self.disable_rsi_filter}")
    
    def fetch_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch latest market data with retry logic.
        
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = self.data_feed.fetch_ohlcv()
                if df.empty:
                    logger.warning(f"Empty data returned (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                
                # Warn if we got less data than expected
                if len(df) < self.ohlcv_limit * 0.5:  # Less than 50% of expected
                    logger.warning(f"⚠️  Only fetched {len(df)} candles (expected ~{self.ohlcv_limit}). This may affect indicator calculations.")
                
                # Reset backoff on success
                self.backoff_seconds = self.interval
                return df
                
            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signals from market data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Signal information dictionary
        """
        try:
            # Calculate technical indicators
            # Use strategy's configured MA periods (fast=3, slow=9)
            fast_period = self.strategy.cfg.fast
            slow_period = self.strategy.cfg.slow
            fast_col = f'sma_{fast_period}'
            slow_col = f'sma_{slow_period}'
            
            df[fast_col] = df['close'].rolling(window=fast_period).mean()
            df[slow_col] = df['close'].rolling(window=slow_period).mean()
            df['signal'] = self.strategy.generate(df, fast_col, slow_col)
            
            # Trend following mode: generate signals when MAs are separated (not just on crossovers)
            # ALWAYS enable trend following if we have valid MAs - this generates more signals
            fast_ma = df[fast_col].iloc[-1] if fast_col in df.columns else None
            slow_ma = df[slow_col].iloc[-1] if slow_col in df.columns else None
            
            # Check if values are valid
            if pd.notna(fast_ma) and pd.notna(slow_ma) and slow_ma > 0:
                ma_diff_pct = ((fast_ma - slow_ma) / slow_ma) * 100
                
                # Always use trend following for more signals (not just on crossovers)
                # Generate trend signals (only if crossover didn't already generate a signal)
                if df['signal'].iloc[-1] == 0:  # No crossover signal
                    # Use very low threshold (0.01% = $10 on $100k BTC) for more signals
                    if fast_ma > slow_ma and abs(ma_diff_pct) > 0.01:  # Fast MA above slow by >0.01%
                        df.at[df.index[-1], 'signal'] = 1
                        logger.info(f"📈 Trend following BUY: fast MA ({fast_ma:.2f}) > slow MA ({slow_ma:.2f}), diff={ma_diff_pct:.3f}%")
                    elif fast_ma < slow_ma and abs(ma_diff_pct) > 0.01:  # Fast MA below slow by >0.01%
                        df.at[df.index[-1], 'signal'] = -1
                        logger.info(f"📉 Trend following SELL: fast MA ({fast_ma:.2f}) < slow MA ({slow_ma:.2f}), diff={ma_diff_pct:.3f}%")
            else:
                logger.warning(f"⚠️  Invalid MA values: fast_ma={fast_ma}, slow_ma={slow_ma}")

            # RSI(14) - calculate and store as both 'rsi_14' and 'rsi' for compatibility
            delta = df['close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss.replace(0, pd.NA))
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi_14']  # Also store as 'rsi' for indicator compatibility

            # ATR(14)
            tr = pd.concat([
                (df['high'] - df['low']),
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
            
            # Calculate comprehensive indicators
            indicators_df = self.indicators.calculate_all_indicators(df)
            
            # Get latest row
            latest = indicators_df.iloc[-1]
            # Use ticker for live price instead of OHLCV close (which is up to 1min stale)
            try:
                ticker = self.data_feed.exchange.fetch_ticker(self.data_symbol)
                current_price = float(ticker['last']) if ticker.get('last') else float(latest['close'])
            except Exception as e:
                logger.warning(f"Could not fetch live ticker, using OHLCV close: {e}")
                current_price = float(latest['close'])
            
            # Strategy-specific signal generation
            signal = 0
            fast_ma_val = latest.get(fast_col, current_price)
            slow_ma_val = latest.get(slow_col, current_price)
            
            if self.trading_strategy == 'multi_indicator':
                # Multi-indicator strategy: Use voting system from get_trading_signals()
                try:
                    signals_df = self.indicators.get_trading_signals(indicators_df)
                    latest_signals = signals_df.iloc[-1]
                    
                    bullish_signals = int(latest_signals.get('bullish_signals', 0))
                    bearish_signals = int(latest_signals.get('bearish_signals', 0))
                    signal_strength = latest_signals.get('signal_strength', 0)
                    
                    # Generate signal based on vote count
                    # Require at least 2 indicators to agree (can be adjusted)
                    signal_threshold = 2
                    if bullish_signals >= signal_threshold:
                        signal = 1
                    elif bearish_signals >= signal_threshold:
                        signal = -1
                    else:
                        signal = 0
                    
                    # Get RSI for logging
                    rsi_val = float(latest.get('rsi_14', 50)) if 'rsi_14' in latest and not pd.isna(latest.get('rsi_14')) else 50.0
                    
                    logger.info(f"📊 Multi-indicator: bullish={bullish_signals}, bearish={bearish_signals}, signal_strength={signal_strength}, final_signal={signal}")
                    
                except Exception as e:
                    logger.error(f"Error in multi-indicator strategy: {e}, falling back to MA crossover")
                    # Fallback to MA crossover
                    signal_raw = latest.get('signal', 0)
                    signal = 0 if pd.isna(signal_raw) else int(signal_raw)
            
            elif self.trading_strategy == 'decision_support':
                # Decision support strategy: Full analysis with ML integration
                if self.enable_ml and self.ml_model:
                    try:
                        analysis = self.decision_support.analyze_market_data(indicators_df)
                        if 'error' not in analysis:
                            signal_result = self.decision_support.generate_trade_signal(analysis, current_price)
                            
                            # Convert decision support signal to -1/0/1
                            ds_signal = signal_result.get('signal', 'hold')
                            if ds_signal in ['strong_buy', 'buy']:
                                signal = 1
                            elif ds_signal in ['strong_sell', 'sell']:
                                signal = -1
                            else:
                                signal = 0
                            
                            confidence = signal_result.get('confidence', 0.5)
                            logger.info(f"🧠 Decision support: signal={ds_signal}, confidence={confidence:.3f}, final_signal={signal}")
                        else:
                            logger.warning("Decision support analysis failed, falling back to MA crossover")
                            signal_raw = latest.get('signal', 0)
                            signal = 0 if pd.isna(signal_raw) else int(signal_raw)
                    except Exception as e:
                        logger.error(f"Error in decision support strategy: {e}, falling back to MA crossover")
                        signal_raw = latest.get('signal', 0)
                        signal = 0 if pd.isna(signal_raw) else int(signal_raw)
                else:
                    logger.warning("Decision support requires ML model (enable_ml=true), falling back to MA crossover")
                    signal_raw = latest.get('signal', 0)
                    signal = 0 if pd.isna(signal_raw) else int(signal_raw)
            
            elif self.trading_strategy == 'ppo_rl':
                # PPO Reinforcement Learning strategy
                try:
                    # Initialize PPO agent if not already done (need state_dim from first feature computation)
                    if self.ppo_agent is None:
                        # Compute features to get state dimension
                        portfolio_value = self.portfolio.equity(current_price)
                        state = self.feature_pipeline.compute_features(
                            df=indicators_df,
                            portfolio_value=portfolio_value,
                            cash=self.portfolio.cash_usd,
                            position_size=self.portfolio.position_qty,
                            last_action=self.last_action,
                            ml_prediction=None  # Will be computed by pipeline if ML available
                        )
                        
                        state_dim = len(state)
                        action_dim = 3  # Buy=0, Hold=1, Sell=2 for discrete actions
                        action_mode = self.ppo_config.get('action_mode', 'discrete') if self.ppo_config else 'discrete'
                        
                        # Create PPO agent
                        self.ppo_agent = PPOAgent(
                            state_dim=state_dim,
                            action_dim=action_dim,
                            action_mode=action_mode,
                            config={
                                'ppo': self.ppo_config if self.ppo_config else {},
                                'action_mode': action_mode
                            }
                        )
                        
                        # Try to load existing model if path provided
                        ppo_model_path = config.get('ppo_model_path') or os.getenv('PPO_MODEL_PATH')
                        if ppo_model_path and os.path.exists(ppo_model_path):
                            try:
                                self.ppo_agent.load(ppo_model_path)
                                logger.info(f"✅ Loaded PPO model from {ppo_model_path}")
                            except Exception as e:
                                logger.warning(f"Could not load PPO model from {ppo_model_path}: {e}, starting fresh")
                        
                        # Load historical experiences if storage enabled
                        if self.experience_storage:
                            try:
                                max_experiences = config.get('experience_storage', {}).get('max_experiences_to_load', 50000)
                                historical_exp = self.experience_storage.load_experiences(num_files=10)
                                if historical_exp and len(historical_exp) > 0:
                                    # Limit and add to buffer
                                    if len(historical_exp) > max_experiences:
                                        historical_exp = historical_exp[-max_experiences:]
                                    for exp in historical_exp:
                                        state_arr = np.array(exp['state']) if not isinstance(exp['state'], np.ndarray) else exp['state']
                                        next_state_arr = np.array(exp['next_state']) if not isinstance(exp['next_state'], np.ndarray) else exp['next_state']
                                        self.ppo_agent.store_experience(
                                            state=state_arr,
                                            action=exp['action'],
                                            reward=exp['reward'],
                                            next_state=next_state_arr,
                                            done=exp['done'],
                                            action_log_prob=exp['action_log_prob']
                                        )
                                    logger.info(f"✅ Loaded {len(historical_exp)} historical experiences")
                            except Exception as e:
                                logger.warning(f"Could not load historical experiences: {e}")
                        
                        logger.info(f"✅ PPO agent initialized: state_dim={state_dim}, action_dim={action_dim}, mode={action_mode}")
                        try:
                            logger.info(f"   Policy network: {len(self.ppo_agent.policy_net.shared_layers)} layers")
                        except:
                            logger.info(f"   Policy network initialized")
                        logger.info(f"   Experience buffer: {len(self.ppo_agent.experience_buffer)}/{self.ppo_agent.experience_buffer.max_size} capacity")
                        logger.info(f"   Update interval: {self.ppo_config.get('update_interval', 100)} steps")
                    
                    # Compute current state
                    portfolio_value = self.portfolio.equity(current_price)
                    current_state = self.feature_pipeline.compute_features(
                        df=indicators_df,
                        portfolio_value=portfolio_value,
                        cash=self.portfolio.cash_usd,
                        position_size=self.portfolio.position_qty,
                        last_action=self.last_action,
                        ml_prediction=None
                    )
                    
                    # Get action from PPO agent
                    action, action_log_prob, action_info = self.ppo_agent.act(
                        current_state, 
                        deterministic=False,
                        return_probs=True
                    )
                    
                    # Store action log prob for experience storage
                    self._last_action_log_prob = action_log_prob
                    
                    # Convert PPO action to signal: 0=Buy, 1=Hold, 2=Sell -> signal: 1=Buy, 0=Hold, -1=Sell
                    if action == 0:
                        signal = 1  # Buy
                    elif action == 2:
                        signal = -1  # Sell
                    else:
                        signal = 0  # Hold
                    
                    # Store action for next iteration
                    self.last_action = action
                    self.last_state = current_state
                    self.last_portfolio_value = portfolio_value
                    
                    # Log PPO decision
                    action_names = ['Buy', 'Hold', 'Sell']
                    action_name = action_names[action] if action < len(action_names) else f'Action_{action}'
                    if action_info:
                        if 'action_probs' in action_info:
                            probs = action_info['action_probs']
                            prob_str = f"B:{probs[0]:.2f} H:{probs[1]:.2f} S:{probs[2]:.2f}"
                        else:
                            prob_str = "N/A"
                        entropy = action_info.get('entropy', 0.0)
                        logger.info(f"🤖 PPO decision: {action_name} (signal={signal}, probs={prob_str}, entropy={entropy:.3f})")
                        logger.debug(f"   State dim: {len(current_state)}, Portfolio: ${portfolio_value:,.2f}, Buffer: {len(self.ppo_agent.experience_buffer)}/{self.ppo_agent.experience_buffer.max_size}")
                    else:
                        logger.info(f"🤖 PPO decision: {action_name} (signal={signal})")
                        logger.debug(f"   State dim: {len(current_state)}, Portfolio: ${portfolio_value:,.2f}, Buffer: {len(self.ppo_agent.experience_buffer)}/{self.ppo_agent.experience_buffer.max_size}")
                    
                except Exception as e:
                    logger.error(f"Error in PPO strategy: {e}", exc_info=True)
                    # Fallback to MA crossover
                    signal_raw = latest.get('signal', 0)
                    signal = 0 if pd.isna(signal_raw) else int(signal_raw)
            
            else:  # ma_crossover (default)
                # Extract signal from MA crossover - try multiple column names for compatibility
                signal_raw = latest.get('signal', 0)
                if pd.isna(signal_raw):
                    signal = 0
                else:
                    signal = int(signal_raw)
            
            # Get RSI value for logging and filtering
            rsi_val = 50.0  # Default neutral
            if 'rsi_14' in latest and not pd.isna(latest['rsi_14']):
                rsi_val = float(latest['rsi_14'])
            elif 'rsi' in latest and not pd.isna(latest['rsi']):
                rsi_val = float(latest['rsi'])
            
            # Enhanced logging - always log when signal is generated
            if signal != 0:
                if self.trading_strategy == 'ma_crossover':
                    logger.info(f"🔔 Signal generated ({self.trading_strategy}): {signal} (fast MA={fast_ma_val:.2f}, slow MA={slow_ma_val:.2f}, RSI={rsi_val:.2f})")
                else:
                    logger.info(f"🔔 Signal generated ({self.trading_strategy}): {signal} (RSI={rsi_val:.2f})")
            else:
                # Log MA relationship for ma_crossover strategy
                if self.trading_strategy == 'ma_crossover' and pd.notna(fast_ma_val) and pd.notna(slow_ma_val) and slow_ma_val > 0:
                    ma_relationship = "above" if fast_ma_val > slow_ma_val else "below"
                    ma_diff = abs(fast_ma_val - slow_ma_val) / slow_ma_val * 100
                    logger.info(f"📊 No signal ({self.trading_strategy}): fast MA ({fast_ma_val:.2f}) {ma_relationship} slow MA ({slow_ma_val:.2f}) by {ma_diff:.3f}% (threshold: 0.01%)")
            
            # RSI filtering (can be disabled for testing, applies to all strategies)
            if not self.disable_rsi_filter:
                # Buy signals confirmed when RSI is low (oversold), sell when RSI is high (overbought)
                # Only execute buy if RSI confirms oversold condition (RSI <= threshold)
                if signal == 1 and rsi_val > self.rsi_buy:
                    logger.info(f"❌ Buy signal BLOCKED: RSI {rsi_val:.2f} > threshold {self.rsi_buy}")
                    signal = 0
                # Only execute sell if RSI confirms overbought condition (RSI >= threshold)
                if signal == -1 and rsi_val < self.rsi_sell:
                    logger.info(f"❌ Sell signal BLOCKED: RSI {rsi_val:.2f} < threshold {self.rsi_sell}")
                    signal = 0
            else:
                if self.trading_strategy == 'ma_crossover':
                    logger.debug(f"RSI filtering DISABLED (test mode)")
            
            return {
                'signal': signal,
                'price': current_price,
                'atr': float(latest.get('atr_14', 0)) if 'atr_14' in latest else 0.0,
                'timestamp': latest.get('ts', pd.Timestamp.now()),
                'indicators': {
                    'rsi': float(latest.get('rsi_14', 50)) if 'rsi_14' in latest else None,
                    'macd': float(latest.get('macd', 0)) if 'macd' in latest else None,
                    f'sma_{fast_period}': float(latest.get(fast_col, current_price)) if fast_col in latest else None,
                    f'sma_{slow_period}': float(latest.get(slow_col, current_price)) if slow_col in latest else None,
                    # Also include old column names for compatibility
                    'sma_12': float(latest.get('sma_12', current_price)) if 'sma_12' in latest else None,
                    'sma_26': float(latest.get('sma_26', current_price)) if 'sma_26' in latest else None,
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            return {'error': str(e)}
    
    def execute_trade(self, signal_info: Dict[str, Any]) -> bool:
        """
        Execute trade based on signal.
        
        Args:
            signal_info: Signal information dictionary
            
        Returns:
            True if trade executed, False otherwise
        """
        if 'error' in signal_info or 'signal' not in signal_info:
            return False
        
        signal = signal_info['signal']
        price = signal_info['price']
        atr = float(signal_info.get('atr', 0) or 0)
        timestamp_ms = int(signal_info['timestamp'].value // 1_000_000) if hasattr(signal_info['timestamp'], 'value') else int(time.time() * 1000)
        
        # Avoid duplicate trades for same timestamp
        if self.last_signal_ts == timestamp_ms:
            return False
        
        try:
            equity = self.portfolio.equity(price)
            
            # Risk management check
            # Note: RiskManager doesn't have can_trade method - check daily limits instead
            daily_limits = self.risk_manager.check_daily_limits()
            if not daily_limits.get('can_trade', True):
                logger.warning(f"🚫 Daily limits reached: trades={daily_limits.get('daily_trades')}, PnL={daily_limits.get('daily_pnl'):.2f}")
                return False

            # Enforce cooldown
            now = time.time()
            if self._last_trade_ts and (now - self._last_trade_ts) < self.cooldown_sec:
                remaining = self.cooldown_sec - (now - self._last_trade_ts)
                logger.debug(f"⏳ Cooldown active: {remaining:.1f}s remaining")
                return False
            
            # Execute buy signal
            if signal == 1:  # Buy
                # Check if position already exists - don't buy again if we have a position
                has_existing_position = self.portfolio.position_qty > 0
                
                # Also check Alpaca for actual position (more reliable)
                if self.provider == 'alpaca':
                    try:
                        api_key = os.getenv('ALPACA_API_KEY')
                        api_secret = os.getenv('ALPACA_SECRET_KEY')
                        if api_key and api_secret:
                            base_url = "https://paper-api.alpaca.markets"
                            headers = {
                                "APCA-API-KEY-ID": api_key,
                                "APCA-API-SECRET-KEY": api_secret
                            }
                            response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                            if response.status_code == 200:
                                positions = response.json()
                                crypto_symbol_base = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.replace('USD', '')
                                for pos in positions:
                                    symbol = pos.get('symbol', '')
                                    qty_str = pos.get('qty', '0')
                                    qty = float(qty_str) if qty_str else 0.0
                                    if qty > 0 and crypto_symbol_base.upper() in symbol.upper():
                                        has_existing_position = True
                                        logger.info(f"⏭️  Skipping BUY signal: position already open (Alpaca qty={qty:.9f} {symbol})")
                                        return False
                    except Exception as e:
                        logger.debug(f"Could not check Alpaca positions: {e}, using portfolio position_qty")
                
                # Check portfolio position as fallback
                if has_existing_position:
                    logger.info(f"⏭️  Skipping BUY signal: position already open (portfolio qty={self.portfolio.position_qty:.9f})")
                    return False
                
                # Sync cash from Alpaca before executing buy to get accurate balance
                actual_available_cash = self.portfolio.cash_usd
                if self.provider == 'alpaca':
                    try:
                        api_key = os.getenv('ALPACA_API_KEY')
                        api_secret = os.getenv('ALPACA_SECRET_KEY')
                        if api_key and api_secret:
                            bal = self.data_feed.exchange.fetch_balance()
                            usd = bal.get('USD') or {}
                            alpaca_cash = usd.get('free') or usd.get('total') or bal.get('free', {}).get('USD') or bal.get('total', {}).get('USD')
                            if isinstance(alpaca_cash, (int, float)) and alpaca_cash > 0:
                                actual_available_cash = float(alpaca_cash)
                                # Update portfolio cash to match Alpaca
                                if abs(actual_available_cash - self.portfolio.cash_usd) > 1.0:  # Only log if significant difference
                                    logger.info(f"💰 Synced cash from Alpaca: ${actual_available_cash:,.2f} (portfolio had ${self.portfolio.cash_usd:,.2f})")
                                    self.portfolio.cash_usd = actual_available_cash
                    except Exception as e:
                        logger.warning(f"Could not sync cash from Alpaca: {e}, using portfolio cash")
                
                if actual_available_cash <= 100:
                    logger.warning(f"🚫 Insufficient cash for buy: ${actual_available_cash:.2f} (need > $100)")
                    return False
                if atr <= 0:
                    logger.warning(f"🚫 ATR is 0 or invalid: {atr}")
                    return False
                
                # ATR-based position sizing
                stop_distance = self.atr_stop_k * atr
                if stop_distance <= 0:
                    logger.warning(f"🚫 Stop distance invalid: {stop_distance}")
                    return False
                    
                dollar_risk = equity * self.risk_per_trade
                qty = max(dollar_risk / stop_distance, 0)
                
                # Calculate notional (cost before fees)
                fee_rate = self.executor.fee_bps / 10000.0
                
                # Start with ideal qty and work backwards to ensure we can afford it
                # Total cost = qty * price * (1 + fee_rate)
                # We want: total_cost <= available_cash
                # So: qty * price * (1 + fee_rate) <= available_cash
                # Therefore: qty <= available_cash / (price * (1 + fee_rate))
                
                # Cap position size to available cash (with 3% buffer for safety and slippage)
                # Use actual Alpaca balance, not stale portfolio cash
                available_cash = actual_available_cash * 0.97  # 3% buffer for safety and price changes
                
                # Calculate max affordable quantity based on cash
                max_qty_by_cash = available_cash / (price * (1 + fee_rate))
                
                # Use the smaller of risk-based qty or cash-limited qty
                qty = min(qty, max_qty_by_cash)
                
                # Calculate notional from the final qty
                # IMPORTANT: We need to account for the fact that Alpaca will charge fees ON TOP
                # So if we want total_cost <= available_cash, we need:
                # notional * (1 + fee_rate) <= available_cash
                # Therefore: notional <= available_cash / (1 + fee_rate)
                
                # Calculate maximum notional that keeps total cost within budget
                max_notional = available_cash / (1 + fee_rate)
                
                # Recalculate qty and notional to ensure we stay within budget
                qty = min(qty, max_notional / price)
                notional = qty * price
                
                # Double-check: ensure notional doesn't exceed max
                if notional > max_notional:
                    notional = max_notional
                    qty = notional / price
                
                total_cost = notional * (1 + fee_rate)
                
                # Final safety check - if still over, be even more conservative
                if total_cost > available_cash:
                    # Reduce by 5% more for safety margin (slippage, price changes)
                    safety_factor = 0.95
                    max_notional_safe = (available_cash * safety_factor) / (1 + fee_rate)
                    qty = max_notional_safe / price
                    notional = max_notional_safe
                    total_cost = notional * (1 + fee_rate)
                    logger.info(f"💰 Position size reduced with safety margin (95%): ${notional:,.2f}")

                logger.info(f"📊 Buy sizing: equity=${equity:,.2f}, risk=${dollar_risk:.2f}, stop_dist=${stop_distance:.2f}, qty={qty:.6f}, notional=${notional:.2f}, total_cost=${total_cost:,.2f}, cash=${actual_available_cash:,.2f}, available=${available_cash:,.2f}")

                # Double-check: Ensure notional doesn't exceed available cash after all calculations
                # Alpaca may use slightly different price, so be conservative
                if notional > available_cash:
                    logger.warning(f"⚠️ Notional ${notional:,.2f} exceeds available ${available_cash:,.2f}, reducing...")
                    notional = available_cash * 0.95  # Additional 5% safety
                    qty = notional / price
                    total_cost = notional * (1 + fee_rate)
                    logger.info(f"💰 Reduced to: notional=${notional:,.2f}, qty={qty:.6f}, total_cost=${total_cost:,.2f}")

                # Pass notional - Alpaca will calculate: cost = notional * (1 + fee_rate)
                # We've ensured notional <= available_cash / (1 + fee_rate)
                result = self.executor.market('buy', price, notional, symbol=self.trade_symbol)
                if self.portfolio.can_afford(price, result.qty, result.fee):
                    self.portfolio.apply_fill('buy', price, result.qty, result.fee)
                    self.db_manager.insert_trade(
                        timestamp_ms, 'buy', self.symbol, price, result.qty, result.fee, 'auto_trade'
                    )
                    logger.info(f"✅ BUY EXECUTED: {result.qty:.6f} {self.symbol} @ ${price:.2f} (equity: ${equity:,.2f})")
                    self.last_signal_ts = timestamp_ms
                    # Initialize stop and bookkeeping
                    self._active_stop = price - stop_distance
                    self._last_trade_ts = now
                    self._last_atr = atr
                    return True
                else:
                    logger.warning(f"🚫 Cannot afford buy: cost=${price * result.qty + result.fee:.2f}, cash=${self.portfolio.cash_usd:.2f}")
                    return False
            
            # Execute sell signal
            elif signal == -1:  # Sell
                if self.portfolio.position_qty <= 0:
                    logger.info(f"⏭️  Skipping SELL signal: no position (qty={self.portfolio.position_qty:.6f}). Waiting for BUY signal first.")
                    return False
                
                # Profit protection: only close on SELL signals if profit >= threshold
                if self.protect_profits:
                    entry_price = self.portfolio.position_avg_price or 0.0
                    if entry_price > 0 and price < entry_price * (1.0 + max(self.min_profit_target, 0.0)):
                        profit_pct = (price - entry_price) / entry_price * 100.0
                        logger.info(
                            f"⏭️  Skipping SELL due to profit protection: price=${price:.2f}, entry=${entry_price:.2f}, "
                            f"profit={profit_pct:.3f}% < target={self.min_profit_target*100:.3f}%"
                        )
                        return False
                
                # Get actual position from Alpaca and close ENTIRE position
                # Use Alpaca's close position API or exact available quantity
                try:
                    api_key = os.getenv('ALPACA_API_KEY')
                    api_secret = os.getenv('ALPACA_SECRET_KEY')
                    if api_key and api_secret:
                        base_url = "https://paper-api.alpaca.markets"
                        headers = {
                            "APCA-API-KEY-ID": api_key,
                            "APCA-API-SECRET-KEY": api_secret
                        }
                        response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                        if response.status_code == 200:
                            positions = response.json()
                            crypto_symbol_base = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.replace('USD', '')
                            position_symbol = None
                            for pos in positions:
                                symbol = pos.get('symbol', '')
                                if crypto_symbol_base.upper() in symbol.upper():
                                    qty_str = pos.get('qty', '0')
                                    available_qty = abs(float(qty_str)) if qty_str else 0.0
                                    position_symbol = symbol  # Store for close position API
                                    if available_qty > 0:
                                        logger.info(f"📊 Closing entire position: Alpaca qty={available_qty:.9f}, portfolio qty={self.portfolio.position_qty:.9f}")
                                        
                                        # Try Alpaca's close position API first (DELETE /v2/positions/{symbol})
                                        try:
                                            close_response = requests.delete(
                                                f"{base_url}/v2/positions/{position_symbol}",
                                                headers=headers,
                                                timeout=10
                                            )
                                            if close_response.status_code in [200, 204]:
                                                # Position closed successfully via API
                                                logger.info(f"✅ Position closed via Alpaca API: {position_symbol}")
                                                # Calculate fee for the closed position
                                                notional = available_qty * price
                                                fee = (self.executor.fee_bps / 10000.0) * notional
                                                
                                                # Update portfolio - position is fully closed
                                                self.portfolio.position_qty = 0
                                                self.portfolio.cash_usd += notional - fee
                                                
                                                # Log trade
                                                self.db_manager.insert_trade(
                                                    timestamp_ms, 'sell', self.symbol, price, available_qty, fee, 'auto_trade'
                                                )
                                                logger.info(f"✅ SELL EXECUTED (full close via API): {available_qty:.9f} {self.symbol} @ ${price:.2f} (equity: ${equity:,.2f})")
                                                self.last_signal_ts = timestamp_ms
                                                self._active_stop = None
                                                self._last_trade_ts = now
                                                return True
                                            else:
                                                logger.info(f"Close position API returned {close_response.status_code}, using market sell with exact quantity")
                                        except Exception as close_err:
                                            logger.debug(f"Close position API not available, using market sell: {close_err}")
                                        
                                        # Fallback: Market sell with EXACT available quantity (no rounding)
                                        sell_qty = available_qty  # Use exact quantity from Alpaca
                                        notional = sell_qty * price
                                        result = self.executor.market('sell', price, notional, symbol=self.trade_symbol)
                                        # Override with exact quantity to ensure we sell everything
                                        result.qty = sell_qty
                                        self.portfolio.apply_fill('sell', price, result.qty, result.fee)
                                        self.db_manager.insert_trade(
                                            timestamp_ms, 'sell', self.symbol, price, result.qty, result.fee, 'auto_trade'
                                        )
                                        logger.info(f"✅ SELL EXECUTED (full close): {result.qty:.9f} {self.symbol} @ ${price:.2f} (equity: ${equity:,.2f})")
                                        self.last_signal_ts = timestamp_ms
                                        self._active_stop = None
                                        self._last_trade_ts = now
                                        return True
                                    break
                except Exception as e:
                    logger.warning(f"Could not fetch actual position from Alpaca for sell: {e}, using portfolio position_qty")
                
                # Fallback: Use exact portfolio position_qty (no rounding down)
                sell_qty = self.portfolio.position_qty
                
                if sell_qty <= 0:
                    logger.warning(f"⚠️ Invalid sell qty: {sell_qty:.9f}, position_qty={self.portfolio.position_qty:.9f}")
                    return False
                    
                notional = sell_qty * price
                result = self.executor.market('sell', price, notional, symbol=self.trade_symbol)
                # Override with exact quantity to close entire position
                result.qty = sell_qty
                self.portfolio.apply_fill('sell', price, result.qty, result.fee)
                self.db_manager.insert_trade(
                    timestamp_ms, 'sell', self.symbol, price, result.qty, result.fee, 'auto_trade'
                )
                logger.info(f"✅ SELL EXECUTED (full close): {result.qty:.9f} {self.symbol} @ ${price:.2f} (equity: ${equity:,.2f})")
                self.last_signal_ts = timestamp_ms
                self._active_stop = None
                self._last_trade_ts = now
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def log_portfolio_snapshot(self, price: float):
        """Log portfolio snapshot to database."""
        try:
            timestamp_ms = int(time.time() * 1000)
            equity = self.portfolio.equity(price)
            self.db_manager.insert_portfolio_metric(
                timestamp_ms,
                self.symbol,
                equity,
                self.portfolio.cash_usd,
                self.portfolio.position_qty,
                self.portfolio.position_avg_price if self.portfolio.position_qty > 0 else 0
            )
        except Exception as e:
            logger.warning(f"Failed to log portfolio snapshot: {e}")
    
    def check_and_reload_model(self) -> bool:
        """
        Check if a better model is available and reload it.
        
        Returns:
            True if model was reloaded, False otherwise
        """
        if not self.enable_ml or not self.ml_model:
            return False
        
        try:
            model_path = self.config.get('model_path') or os.getenv('ML_MODEL_PATH', 'ml_models/trained_model.pkl')
            
            # Check if model file was updated (newer version available)
            if os.path.exists(model_path):
                # Try to reload model if it's newer
                try:
                    new_model = CryptoPredictionModel(suppress_init_log=True)
                    new_model.load_model(model_path)
                    
                    # Compare algorithms (for logging)
                    current_algo = getattr(self.ml_model, 'algorithm', 'unknown')
                    new_algo = getattr(new_model, 'algorithm', 'unknown')
                    
                    if current_algo != new_algo:
                        logger.info(f"🔄 Model algorithm changed: {current_algo} → {new_algo}, reloading...")
                        self.ml_model = new_model
                        self.decision_support.ml_model = self.ml_model
                        return True
                except Exception as e:
                    logger.debug(f"Could not reload model: {e}")
            
            return False
        except Exception as e:
            logger.debug(f"Error checking model reload: {e}")
            return False
    
    def check_and_retrain_model(self) -> bool:
        """
        Check if model retraining is needed and retrain if necessary.
        
        Returns:
            True if retraining occurred, False otherwise
        """
        if not self.enable_auto_retraining or not self.retraining_service:
            return False
        
        try:
            now = datetime.now(timezone.utc)
            
            # Check if enough time has passed since last check
            if self._last_retraining_check is None or (now - self._last_retraining_check) >= self._retraining_check_interval:
                self._last_retraining_check = now
                
                # Check if retraining is needed
                result = self.retraining_service.retrain_if_needed(force=False)
                
                if result and 'error' not in result:
                    logger.info(f"🔄 New model trained: {result.get('version')}")
                    
                    # Check if new model was activated
                    if result.get('activated', False):
                        # Reload model from the version that was just activated
                        version = result.get('version')
                        if version and version in self.retraining_service.model_versions:
                            model_info = self.retraining_service.model_versions[version]
                            model_path = model_info.get('model_path')
                            if model_path and os.path.exists(model_path):
                                try:
                                    new_model = CryptoPredictionModel(suppress_init_log=True)
                                    new_model.load_model(model_path)
                                    self.ml_model = new_model
                                    self.decision_support.ml_model = self.ml_model
                                    logger.info(f"✅ Model reloaded: {model_path} (algorithm: {getattr(new_model, 'algorithm', 'unknown')})")
                                    return True
                                except Exception as e:
                                    logger.error(f"Failed to reload new model: {e}")
                        else:
                            # Fallback: try to reload from default path
                            model_path = self.config.get('model_path') or os.getenv('ML_MODEL_PATH', 'ml_models/trained_model.pkl')
                            if os.path.exists(model_path):
                                try:
                                    new_model = CryptoPredictionModel(suppress_init_log=True)
                                    new_model.load_model(model_path)
                                    self.ml_model = new_model
                                    self.decision_support.ml_model = self.ml_model
                                    logger.info(f"✅ Model reloaded from default path: {model_path} (algorithm: {getattr(new_model, 'algorithm', 'unknown')})")
                                    return True
                                except Exception as e:
                                    logger.error(f"Failed to reload model from default path: {e}")
                    
                    return False
                elif result and 'error' in result:
                    logger.debug(f"Retraining check: {result['error']}")
            
            return False
        except Exception as e:
            logger.error(f"Error in auto-retraining check: {e}")
            return False
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)  # Docker/container shutdown
        except (ValueError, OSError):
            # Signal handlers may not work in all environments (e.g., some Windows setups)
            logger.warning("Could not register signal handlers (may not work on this platform)")
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (Ctrl+C, SIGTERM).
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        try:
            signal_name = signal.Signals(signum).name
            logger.info(f"Received {signal_name} signal - initiating graceful shutdown...")
        except:
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def run(self):
        """
        Main trading loop - runs continuously until stopped.
        """
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        logger.info("=" * 60)
        logger.info("Starting continuous trading service")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)
        
        while not self.shutdown_requested:
            # Check for shutdown request at start of loop
            if self.shutdown_requested:
                logger.info("Shutdown requested, breaking loop...")
                break
            try:
                self.iteration_count += 1
                iteration_start = time.time()
                
                # Check for model updates and retrain if needed (every N iterations)
                if self.enable_ml and self.iteration_count % 60 == 0:  # Check every 60 iterations
                    self.check_and_reload_model()
                    if self.enable_auto_retraining:
                        self.check_and_retrain_model()
                
                # Fetch latest data
                df = self.fetch_latest_data()
                
                if df is None or df.empty:
                    logger.warning("No data available, backing off...")
                    sleep_time = min(60, self.backoff_seconds) + random.uniform(0, 0.5 * min(60, self.backoff_seconds))
                    time.sleep(sleep_time)
                    self.backoff_seconds = min(60, self.backoff_seconds * 1.5)  # Exponential backoff
                    continue
                
                # Calculate signals
                signal_info = self.calculate_signals(df)
                
                if 'error' not in signal_info:
                    current_price = signal_info['price']
                    self._last_atr = float(signal_info.get('atr', 0) or self._last_atr or 0)
                    signal = signal_info['signal']
                    rsi_value = signal_info.get('indicators', {}).get('rsi', None)
                    
                    # Log portfolio snapshot
                    self.log_portfolio_snapshot(current_price)
                    
                    # Trailing stop management for open position
                    if self.portfolio.position_qty > 0 and self._active_stop and self._last_atr and self._last_atr > 0:
                        new_trail = current_price - self.atr_trail_k * self._last_atr
                        self._active_stop = max(self._active_stop, new_trail)
                        if current_price <= self._active_stop:
                            # Stop-out: Get actual position from Alpaca and close ENTIRE position
                            try:
                                api_key = os.getenv('ALPACA_API_KEY')
                                api_secret = os.getenv('ALPACA_SECRET_KEY')
                                if api_key and api_secret:
                                    base_url = "https://paper-api.alpaca.markets"
                                    headers = {
                                        "APCA-API-KEY-ID": api_key,
                                        "APCA-API-SECRET-KEY": api_secret
                                    }
                                    response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                                    if response.status_code == 200:
                                        positions = response.json()
                                        crypto_symbol_base = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.replace('USD', '')
                                        position_symbol = None
                                        for pos in positions:
                                            symbol = pos.get('symbol', '')
                                            if crypto_symbol_base.upper() in symbol.upper():
                                                qty_str = pos.get('qty', '0')
                                                available_qty = abs(float(qty_str)) if qty_str else 0.0
                                                position_symbol = symbol
                                                if available_qty > 0:
                                                    # Use EXACT quantity to close entire position
                                                    sell_qty = available_qty
                                                    
                                                    # Try Alpaca close position API first
                                                    try:
                                                        close_response = requests.delete(
                                                            f"{base_url}/v2/positions/{position_symbol}",
                                                            headers=headers,
                                                            timeout=10
                                                        )
                                                        if close_response.status_code in [200, 204]:
                                                            logger.info(f"STOP OUT (full close via API): {sell_qty:.9f} {self.symbol} @ ${current_price:.2f}")
                                                            notional = sell_qty * current_price
                                                            fee = (self.executor.fee_bps / 10000.0) * notional
                                                            self.portfolio.position_qty = 0
                                                            self.portfolio.cash_usd += notional - fee
                                                            ts_ms = int(time.time() * 1000)
                                                            self.db_manager.insert_trade(ts_ms, 'sell', self.symbol, current_price, sell_qty, fee, 'stop_out')
                                                            self._active_stop = None
                                                            self._last_trade_ts = time.time()
                                                            continue  # Skip rest of loop iteration
                                                    except Exception:
                                                        pass  # Fall through to market sell
                                                    
                                                    # Fallback: Market sell with exact quantity
                                                    notional = sell_qty * current_price
                                                    res = self.executor.market('sell', current_price, notional, symbol=self.trade_symbol)
                                                    res.qty = sell_qty  # Use exact quantity
                                                    self.portfolio.apply_fill('sell', current_price, res.qty, res.fee)
                                                    ts_ms = int(time.time() * 1000)
                                                    self.db_manager.insert_trade(ts_ms, 'sell', self.symbol, current_price, res.qty, res.fee, 'stop_out')
                                                    logger.info(f"STOP OUT (full close): {res.qty:.9f} {self.symbol} @ ${current_price:.2f}")
                                                    self._active_stop = None
                                                    self._last_trade_ts = time.time()
                                                    continue  # Skip rest of loop iteration
                            except Exception as e:
                                logger.warning(f"Could not fetch position for stop-out: {e}, using portfolio qty")
                            
                            # Fallback: Use exact portfolio qty (no rounding)
                            sell_qty = self.portfolio.position_qty
                            if sell_qty > 0:
                                try:
                                    notional = sell_qty * current_price
                                    res = self.executor.market('sell', current_price, notional, symbol=self.trade_symbol)
                                    res.qty = sell_qty  # Use exact quantity
                                    self.portfolio.apply_fill('sell', current_price, res.qty, res.fee)
                                    ts_ms = int(time.time() * 1000)
                                    self.db_manager.insert_trade(ts_ms, 'sell', self.symbol, current_price, res.qty, res.fee, 'stop_out')
                                    logger.info(f"STOP OUT (full close): {res.qty:.9f} {self.symbol} @ ${current_price:.2f}")
                                    self._active_stop = None
                                    self._last_trade_ts = time.time()
                                except Exception as e:
                                    logger.error(f"Error executing stop-out: {e}")

                    # Execute trade if signal present
                    trade_executed = False
                    if signal != 0:
                        logger.info(f"🚀 Executing trade: Signal={signal}, Price=${current_price:.2f}")
                        try:
                            trade_result = self.execute_trade(signal_info)
                            trade_executed = trade_result
                            if not trade_result:
                                # Additional logging already done in execute_trade, but log summary
                                logger.debug(f"Trade execution returned False (see above logs for reason)")
                        except Exception as e:
                            logger.error(f"❌ Exception during trade execution: {e}", exc_info=True)
                    
                    # Update PPO agent with experience and rewards (if using PPO strategy)
                    if self.trading_strategy == 'ppo_rl' and self.ppo_agent is not None:
                        try:
                            # Get updated portfolio state after trade
                            updated_portfolio_value = self.portfolio.equity(current_price)
                            updated_position_size = self.portfolio.position_qty
                            
                            # Calculate reward
                            if self.last_portfolio_value is not None and hasattr(self, 'last_price'):
                                prev_price = getattr(self, 'last_price', current_price)
                                price_change_pct = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
                            else:
                                price_change_pct = 0.0
                            
                            if self.last_portfolio_value is not None:
                                
                                reward = self.reward_calculator.calculate_step_reward(
                                    previous_portfolio_value=self.last_portfolio_value,
                                    current_portfolio_value=updated_portfolio_value,
                                    previous_position_size=self.last_position_size,
                                    current_position_size=updated_position_size,
                                    previous_action=self.last_action if self.last_action is not None else 1,
                                    current_action=self.last_action if self.last_action is not None else 1,
                                    price_change_pct=price_change_pct,
                                    entry_price=self.portfolio.position_avg_price if self.portfolio.position_avg_price else None,
                                    execution_price=current_price if trade_executed else None,
                                    current_price=current_price,
                                    position_age_bars=self.ppo_step_count  # Simple tracking
                                )
                                
                                # Log reward calculation (debug level)
                                portfolio_change_pct = ((updated_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value * 100) if self.last_portfolio_value else 0.0
                                logger.debug(f"🎯 Reward: {reward:.6f} (portfolio: ${self.last_portfolio_value:,.2f} → ${updated_portfolio_value:,.2f}, change: {portfolio_change_pct:.3f}%)")
                                
                                # Store experience
                                if self.last_state is not None:
                                    next_state = self.feature_pipeline.compute_features(
                                        df=df,
                                        portfolio_value=updated_portfolio_value,
                                        cash=self.portfolio.cash_usd,
                                        position_size=updated_position_size,
                                        last_action=self.last_action,
                                        ml_prediction=None
                                    )
                                    
                                    # Get action log prob from last action (we stored it during act)
                                    action_log_prob = getattr(self, '_last_action_log_prob', 0.0)
                                    
                                    self.ppo_agent.store_experience(
                                        state=self.last_state,
                                        action=self.last_action if self.last_action is not None else 1,
                                        reward=reward,
                                        next_state=next_state,
                                        done=False,  # Continuous trading, no episode termination
                                        action_log_prob=action_log_prob
                                    )
                                    
                                    self.ppo_step_count += 1
                                    
                                    # Periodic PPO update
                                    update_interval = self.ppo_config.get('update_interval', 100) if self.ppo_config else 100
                                    if self.ppo_step_count > 0 and self.ppo_step_count % update_interval == 0:
                                        logger.info(f"🔄 Updating PPO policy at step {self.ppo_step_count}...")
                                        buffer_size = len(self.ppo_agent.experience_buffer)
                                        logger.info(f"   Using {buffer_size} experiences from buffer")
                                        training_metrics = self.ppo_agent.update()
                                        if 'error' not in training_metrics:
                                            logger.info(f"✅ PPO Policy Update Complete:")
                                            logger.info(f"   Policy Loss: {training_metrics.get('policy_loss', 0):.6f}")
                                            logger.info(f"   Value Loss: {training_metrics.get('value_loss', 0):.6f}")
                                            logger.info(f"   Entropy: {training_metrics.get('entropy', 0):.6f}")
                                            logger.info(f"   Mean Advantage: {training_metrics.get('mean_advantage', 0):.4f}")
                                            if 'learning_rate' in training_metrics:
                                                logger.info(f"   Learning Rate: {training_metrics.get('learning_rate', 0):.2e}")
                                            
                                            # Log to training logger if available
                                            if self.trading_logger:
                                                self.trading_logger.log_training(
                                                    step=self.ppo_step_count,
                                                    metrics=training_metrics
                                                )
                                        else:
                                            logger.warning(f"⚠️ PPO update had errors: {training_metrics.get('error')}")
                                        
                                        # Save checkpoint after successful update
                                        if 'error' not in training_metrics:
                                            ppo_model_dir = self.config.get('ppo_model_dir', 'models')
                                            os.makedirs(ppo_model_dir, exist_ok=True)
                                            checkpoint_path = Path(ppo_model_dir) / f'ppo_agent_step_{self.ppo_step_count}.pt'
                                            try:
                                                self.ppo_agent.save(str(checkpoint_path))
                                                logger.info(f"💾 Saved PPO checkpoint: {checkpoint_path}")
                                                
                                                # Cleanup old checkpoints
                                                if self.checkpoint_manager:
                                                    removed = self.checkpoint_manager.cleanup_old_checkpoints()
                                                    if removed > 0:
                                                        logger.info(f"🧹 Cleaned up {removed} old checkpoint(s)")
                                            except Exception as e:
                                                logger.warning(f"Could not save PPO checkpoint: {e}")
                                    
                                    # Save experiences periodically
                                    if self.experience_storage and self.ppo_step_count % 1000 == 0:
                                        try:
                                            experiences = self.ppo_agent.experience_buffer.get_all()
                                            if experiences:
                                                filepath = self.experience_storage.save_experiences(
                                                    experiences=experiences[-1000:],  # Save last 1000
                                                    tag=f"step_{self.ppo_step_count}"
                                                )
                                                if filepath:
                                                    logger.info(f"💾 Saved {len(experiences[-1000:])} experiences to storage")
                                        except Exception as e:
                                            logger.warning(f"Could not save experiences: {e}")
                            
                            # Update tracking variables
                            self.last_portfolio_value = updated_portfolio_value
                            self.last_position_size = updated_position_size
                            
                        except Exception as e:
                            logger.error(f"Error updating PPO agent: {e}", exc_info=True)
                    
                    # Log status every 10 iterations with more details
                    if self.iteration_count % 10 == 0:
                        equity = self.portfolio.equity(current_price)
                        total_return = ((equity - self.starting_cash) / self.starting_cash) * 100
                        # Get RSI for logging
                        rsi_log = 'N/A'
                        if rsi_value is not None and not pd.isna(rsi_value):
                            rsi_log = f"{float(rsi_value):.2f}"
                        # Get MA values from signal_info
                        indicators = signal_info.get('indicators', {})
                        fast_ma_log = indicators.get(f'sma_{self.strategy.cfg.fast}', 'N/A')
                        slow_ma_log = indicators.get(f'sma_{self.strategy.cfg.slow}', 'N/A')
                        if isinstance(fast_ma_log, (int, float)):
                            fast_ma_log = f"{fast_ma_log:.2f}"
                        if isinstance(slow_ma_log, (int, float)):
                            slow_ma_log = f"{slow_ma_log:.2f}"
                        logger.info(
                            f"Iteration {self.iteration_count}: "
                            f"Price=${current_price:.2f}, "
                            f"Signal={signal}, "
                            f"RSI={rsi_log}, "
                            f"MA({self.strategy.cfg.fast})={fast_ma_log}, "
                            f"MA({self.strategy.cfg.slow})={slow_ma_log}, "
                            f"Equity=${equity:,.2f}, "
                            f"Return={total_return:.2f}%"
                        )
                    
                    self.last_price = current_price
                
                # Reset backoff on success
                self.backoff_seconds = self.interval
                
                # Sleep until next interval
                elapsed = time.time() - iteration_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.shutdown_requested = True
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                # Back off on error
                sleep_time = min(60, self.backoff_seconds)
                time.sleep(sleep_time)
                self.backoff_seconds = min(60, self.backoff_seconds * 1.5)
        
        # Final portfolio summary
        logger.info("\n" + "=" * 60)
        logger.info("Continuous trading service stopped")
        
        # Save final PPO checkpoint if using PPO
        if self.trading_strategy == 'ppo_rl' and self.ppo_agent:
            logger.info("Performing final save before shutdown...")
            try:
                ppo_model_dir = self.config.get('ppo_model_dir', 'models')
                os.makedirs(ppo_model_dir, exist_ok=True)
                final_path = Path(ppo_model_dir) / 'ppo_agent_final.pt'
                self.ppo_agent.save(str(final_path))
                logger.info(f"💾 Saved final PPO checkpoint: {final_path}")
                
                # Save remaining experiences
                if self.experience_storage:
                    try:
                        experiences = self.ppo_agent.experience_buffer.get_all()
                        if experiences:
                            self.experience_storage.save_experiences(experiences, tag='final')
                            logger.info(f"💾 Saved {len(experiences)} final experiences")
                    except Exception as e:
                        logger.warning(f"Could not save final experiences: {e}")
                
                # Final checkpoint cleanup
                if self.checkpoint_manager:
                    removed = self.checkpoint_manager.cleanup_old_checkpoints()
                    if removed > 0:
                        logger.info(f"🧹 Final cleanup: removed {removed} old checkpoint(s)")
            except Exception as e:
                logger.error(f"Could not save final PPO checkpoint: {e}", exc_info=True)
        
        if self.last_price:
            final_equity = self.portfolio.equity(self.last_price)
            total_return = ((final_equity - self.starting_cash) / self.starting_cash) * 100
            logger.info(f"Final equity: ${final_equity:,.2f}")
            logger.info(f"Total return: {total_return:.2f}%")
            logger.info(f"Cash: ${self.portfolio.cash_usd:,.2f}")
            logger.info(f"Position: {self.portfolio.position_qty:.6f}")
            if self.trading_strategy == 'ppo_rl' and self.ppo_agent:
                logger.info(f"PPO steps completed: {self.ppo_step_count}")
        logger.info("=" * 60)
