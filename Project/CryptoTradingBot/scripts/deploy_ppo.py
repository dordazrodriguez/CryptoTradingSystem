"""
Deployment script for PPO hybrid trading system.
Combines Trend Following + ML + PPO for live trading.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import argparse
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_engine.ppo_env import TradingEnv, TradingEnvConfig
from trading_engine.ppo_agent import PPOAgent, HybridPPOStrategy
from ml_models.predictor import CryptoPredictionModel
from trading_engine.indicators import TechnicalIndicators
from trading_engine.portfolio import PortfolioManager
from trading_engine.simple_executor import SimpleExecutor
from data.collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PPOTradingSystem:
    """
    Complete PPO hybrid trading system for live deployment.
    
    Combines:
    1. Trend Following (EMA, ADX, ATR filters)
    2. ML Predictions (LightGBM probability)
    3. PPO Agent (dynamic decision-making)
    """
    
    def __init__(self,
                 ppo_model_path: str,
                 ml_model_path: Optional[str] = None,
                 symbol: str = 'BTC/USD',
                 timeframe: str = '1h',
                 initial_cash: float = 100000.0,
                 paper_trading: bool = True):
        """
        Initialize PPO trading system.
        
        Args:
            ppo_model_path: Path to trained PPO model
            ml_model_path: Path to trained ML model (optional)
            symbol: Trading symbol
            timeframe: Data timeframe
            initial_cash: Initial cash balance
            paper_trading: Use paper trading mode
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_cash = initial_cash
        self.paper_trading = paper_trading
        
        # Initialize components
        self.collector = DataCollector()
        self.indicators = TechnicalIndicators()
        self.portfolio = PortfolioManager(initial_cash=initial_cash)
        self.executor = SimpleExecutor(paper_trading=paper_trading)
        
        # Load ML model
        self.ml_model = None
        if ml_model_path and os.path.exists(ml_model_path):
            try:
                self.ml_model = CryptoPredictionModel(algorithm='lightgbm', model_type='classifier')
                self.ml_model.load_model(ml_model_path)
                logger.info(f"Loaded ML model from {ml_model_path}")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
        
        # Create environment for PPO agent
        # We'll create a dummy environment for loading the model
        config = TradingEnvConfig(initial_cash=initial_cash)
        dummy_df = self._get_dummy_data()
        self.env = TradingEnv(df=dummy_df, ml_model=self.ml_model, config=config)
        
        # Load PPO agent
        self.ppo_agent = PPOAgent(env=self.env, model_path=ppo_model_path)
        logger.info(f"Loaded PPO agent from {ppo_model_path}")
        
        # Create hybrid strategy
        self.strategy = HybridPPOStrategy(
            ppo_agent=self.ppo_agent,
            ml_model=self.ml_model,
            trend_filter_enabled=True,
            ml_filter_enabled=True
        )
        
        logger.info("PPO trading system initialized")
    
    def _get_dummy_data(self) -> pd.DataFrame:
        """Get dummy data for environment initialization."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq=self.timeframe)
        return pd.DataFrame({
            'timestamp': dates,
            'open': [50000] * 100,
            'high': [51000] * 100,
            'low': [49000] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })
    
    def _get_current_data(self, lookback: int = 200) -> pd.DataFrame:
        """
        Get current market data.
        
        Args:
            lookback: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            df = self.collector.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=None,
                limit=lookback
            )
            
            if df is None or len(df) == 0:
                raise ValueError("No data fetched")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _prepare_state(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare state vector from market data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            State vector
        """
        # Update environment with new data
        self.env.df = df.copy()
        self.env._prepare_features()
        
        # Get current state
        state = self.env._get_state()
        
        return state
    
    def _get_ml_prediction(self, df: pd.DataFrame) -> float:
        """
        Get ML prediction probability.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            ML prediction probability (0-1)
        """
        if self.ml_model and self.ml_model.is_trained:
            try:
                pred = self.ml_model.predict(df)
                prob = pred.get('confidence', 0.5) if pred.get('prediction', 0) == 1 else 1 - pred.get('confidence', 0.5)
                return float(prob)
            except Exception as e:
                logger.warning(f"ML prediction error: {e}")
        
        return 0.5
    
    def _get_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of indicator values
        """
        close = df['close'].iloc[-1]
        
        # Calculate indicators
        indicators = {}
        
        # EMA
        indicators['ema_12'] = self.indicators.calculate_ema(df['close'], 12).iloc[-1]
        indicators['ema_26'] = self.indicators.calculate_ema(df['close'], 26).iloc[-1]
        
        # ADX
        adx_data = self.indicators.calculate_adx(df['high'], df['low'], df['close'], window=14)
        indicators['adx'] = adx_data.get('adx', pd.Series(0, index=df.index)).iloc[-1]
        indicators['plus_di'] = adx_data.get('plus_di', pd.Series(0, index=df.index)).iloc[-1]
        indicators['minus_di'] = adx_data.get('minus_di', pd.Series(0, index=df.index)).iloc[-1]
        
        # ATR
        atr = self.indicators.calculate_atr(df['high'], df['low'], df['close'], window=14)
        indicators['atr'] = atr.iloc[-1]
        
        # RSI
        indicators['rsi'] = self.indicators.calculate_rsi(df['close'], window=14).iloc[-1]
        
        # Normalize price-based indicators
        indicators['ema_12'] = (indicators['ema_12'] / close) if close > 0 else 1.0
        indicators['ema_26'] = (indicators['ema_26'] / close) if close > 0 else 1.0
        indicators['atr'] = (indicators['atr'] / close) if close > 0 else 0.0
        
        return indicators
    
    def get_trading_decision(self) -> Dict[str, Any]:
        """
        Get trading decision from hybrid system.
        
        Returns:
            Decision dictionary with action, confidence, and reasoning
        """
        # Get current data
        df = self._get_current_data(lookback=200)
        
        if len(df) == 0:
            return {
                'action': 0,
                'action_name': 'Hold',
                'reason': 'No data available'
            }
        
        current_price = df['close'].iloc[-1]
        
        # Prepare state
        state = self._prepare_state(df)
        
        # Get indicators (normalized)
        indicators = self._get_indicators(df)
        
        # Get ML prediction
        ml_prob = self._get_ml_prediction(df.tail(100))
        
        # Get decision from hybrid strategy
        decision = self.strategy.get_decision(
            state=state,
            current_price=current_price,
            indicators=indicators,
            ml_probability=ml_prob
        )
        
        decision['timestamp'] = datetime.now().isoformat()
        decision['price'] = float(current_price)
        decision['symbol'] = self.symbol
        
        return decision
    
    def execute_trading_cycle(self) -> Dict[str, Any]:
        """
        Execute one trading cycle.
        
        Returns:
            Execution results
        """
        logger.info("Executing trading cycle")
        
        # Get decision
        decision = self.get_trading_decision()
        
        logger.info(f"Decision: {decision['action_name']} (Confidence: {decision['confidence']:.2f})")
        logger.info(f"Reasoning: {decision['reasoning']}")
        
        # Execute trade if not holding
        if decision['action'] != 0:
            try:
                # Determine trade direction
                if decision['action'] == 1:  # Long
                    trade_result = self.executor.execute_buy(
                        symbol=self.symbol,
                        quantity=None,  # Use default sizing
                        price=decision['price']
                    )
                elif decision['action'] == 2:  # Short
                    trade_result = self.executor.execute_sell(
                        symbol=self.symbol,
                        quantity=None,
                        price=decision['price']
                    )
                else:
                    trade_result = {'status': 'no_action'}
                
                decision['trade_result'] = trade_result
                
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                decision['trade_result'] = {'status': 'error', 'message': str(e)}
        
        return decision
    
    def run_continuous(self, interval_seconds: int = 3600):
        """
        Run trading system continuously.
        
        Args:
            interval_seconds: Time between trading cycles (seconds)
        """
        logger.info(f"Starting continuous trading (interval: {interval_seconds}s)")
        
        try:
            while True:
                # Execute trading cycle
                result = self.execute_trading_cycle()
                
                # Get portfolio status
                portfolio_value = self.portfolio.get_portfolio_value({self.symbol: result['price']})
                
                logger.info(f"Portfolio Value: ${portfolio_value['total_value']:.2f}")
                logger.info(f"Cash: ${portfolio_value['cash']:.2f}")
                logger.info(f"Positions: {portfolio_value.get('positions', {})}")
                
                # Wait for next cycle
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Trading error: {e}", exc_info=True)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy PPO hybrid trading system')
    parser.add_argument('--ppo-model', type=str, required=True, help='Path to trained PPO model')
    parser.add_argument('--ml-model', type=str, default=None, help='Path to trained ML model')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    parser.add_argument('--initial-cash', type=float, default=100000.0, help='Initial cash balance')
    parser.add_argument('--paper-trading', action='store_true', default=True, help='Use paper trading')
    parser.add_argument('--live-trading', action='store_true', help='Use live trading (override paper trading)')
    parser.add_argument('--interval', type=int, default=3600, help='Trading cycle interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuously')
    
    args = parser.parse_args()
    
    # Determine trading mode
    paper_trading = not args.live_trading if args.live_trading else args.paper_trading
    
    if not paper_trading:
        logger.warning("LIVE TRADING MODE - Real money will be used!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Trading cancelled")
            return
    
    # Initialize trading system
    system = PPOTradingSystem(
        ppo_model_path=args.ppo_model,
        ml_model_path=args.ml_model,
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_cash=args.initial_cash,
        paper_trading=paper_trading
    )
    
    if args.once:
        # Run once
        result = system.execute_trading_cycle()
        print("\n" + "="*50)
        print("Trading Decision")
        print("="*50)
        print(f"Action: {result['action_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Price: ${result['price']:.2f}")
        print("="*50)
    else:
        # Run continuously
        system.run_continuous(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

