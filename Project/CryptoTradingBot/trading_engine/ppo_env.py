"""
Gymnasium environment for PPO-based trading system.
Simulates cryptocurrency trading environment for reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

from trading_engine.indicators import TechnicalIndicators
from trading_engine.portfolio import PortfolioManager
from ml_models.predictor import CryptoPredictionModel
from ml_models.features import MLFeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    initial_cash: float = 100000.0
    commission_rate: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    position_size_limit: float = 1.0  # Maximum position size (as fraction of portfolio)
    max_drawdown_penalty: float = 0.1  # Penalty weight for max drawdown
    transaction_penalty: float = 0.01  # Penalty weight for transactions
    trend_bonus: float = 0.05  # Bonus weight for holding in strong trends
    lookback_window: int = 50  # Number of historical bars to include in state
    # Enhanced reward function parameters
    use_sharpe_reward: bool = True  # Use Sharpe ratio instead of simple returns
    use_sortino_reward: bool = False  # Use Sortino ratio (only downside volatility)
    fee_penalty_weight: float = 0.001  # Penalty for trading fees
    risk_penalty_weight: float = 0.05  # Penalty for high volatility/drawdowns
    whipsaw_penalty_weight: float = 0.02  # Penalty for rapid buy/sell reversals
    reward_lookback: int = 20  # Lookback window for Sharpe/Sortino calculation


class TradingEnv(gym.Env):
    """
    Trading environment for PPO agent.
    
    State Space:
    - Technical indicators (EMA, ADX, ATR, RSI, MACD, etc.)
    - ML prediction probability
    - Current position info (position size, unrealized PnL)
    - Price features (returns, volatility, momentum)
    
    Action Space:
    - Discrete: 0 = hold, 1 = go long, 2 = go short
    - Or continuous: position size between -1 and +1
    
    Reward Function:
    - Reward = Δ_equity - λ1 * max_drawdown - λ2 * transaction_costs + λ3 * trend_bonus
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 ml_model: Optional[CryptoPredictionModel] = None,
                 config: TradingEnvConfig = TradingEnvConfig(),
                 action_type: str = 'discrete'):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            ml_model: Trained ML model for prediction probability
            config: Environment configuration
            action_type: 'discrete' or 'continuous'
        """
        super().__init__()
        
        self.df = df.copy()
        self.ml_model = ml_model
        self.config = config
        self.action_type = action_type
        
        # Initialize components
        self.indicators = TechnicalIndicators()
        self.feature_engineer = MLFeatureEngineer()
        self.portfolio = PortfolioManager(initial_cash=config.initial_cash)
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Calculate all indicators
        self._prepare_features()
        
        # Determine state dimension
        self.state_dim = self._get_state_dimension()
        
        # Define action space - expanded to allow partial positions
        if action_type == 'discrete':
            # Expanded action space: 0=hold, 1=buy_25%, 2=buy_50%, 3=buy_100%, 4=sell_25%, 5=sell_50%, 6=sell_100%
            self.action_space = spaces.Discrete(7)
        elif action_type == 'continuous':
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.previous_equity = config.initial_cash
        self.max_equity = config.initial_cash
        self.max_drawdown = 0.0
        self.position_size = 0.0  # Current position (-1 to +1)
        self.entry_price = 0.0
        
        # Track returns for Sharpe/Sortino calculation
        self.returns_history = []
        self.equity_history = [config.initial_cash]
        
        # Track last actions for whipsaw penalty
        self.last_actions = []  # Track last few actions
        
        logger.info(f"Trading environment initialized: {len(self.df)} bars, state_dim={self.state_dim}")
    
    def _prepare_features(self):
        """Prepare all features and indicators."""
        # Calculate technical indicators
        close = self.df['close']
        
        # Trend indicators
        self.df['ema_12'] = self.indicators.calculate_ema(close, 12)
        self.df['ema_26'] = self.indicators.calculate_ema(close, 26)
        self.df['sma_20'] = self.indicators.calculate_sma(close, 20)
        self.df['sma_50'] = self.indicators.calculate_sma(close, 50)
        
        # ADX (trend strength)
        adx_data = self.indicators.calculate_adx(self.df['high'], self.df['low'], self.df['close'], window=14)
        self.df['adx'] = adx_data.get('adx', pd.Series(0, index=self.df.index))
        self.df['plus_di'] = adx_data.get('plus_di', pd.Series(0, index=self.df.index))
        self.df['minus_di'] = adx_data.get('minus_di', pd.Series(0, index=self.df.index))
        
        # ATR (volatility)
        atr_data = self.indicators.calculate_atr(self.df['high'], self.df['low'], self.df['close'], window=14)
        self.df['atr'] = atr_data
        
        # Momentum indicators
        self.df['rsi'] = self.indicators.calculate_rsi(close, window=14)
        
        # MACD
        macd_data = self.indicators.calculate_macd(close)
        self.df['macd'] = macd_data.get('macd', pd.Series(0, index=self.df.index))
        self.df['macd_signal'] = macd_data.get('signal', pd.Series(0, index=self.df.index))
        self.df['macd_hist'] = macd_data.get('histogram', pd.Series(0, index=self.df.index))
        
        # Price features
        featured_df = self.feature_engineer.create_price_features(self.df)
        for col in featured_df.columns:
            if col not in self.df.columns:
                self.df[col] = featured_df[col]
        
        # Fill NaN values
        self.df = self.df.bfill().fillna(0)
        
        # ML predictions if model available
        if self.ml_model and self.ml_model.is_trained:
            try:
                # Get ML prediction probability for each step
                ml_probs = []
                for i in range(len(self.df)):
                    if i >= 50:  # Need enough history
                        window_df = self.df.iloc[max(0, i-100):i+1]
                        try:
                            pred = self.ml_model.predict(window_df)
                            prob = pred.get('confidence', 0.5) if pred.get('prediction', 0) == 1 else 1 - pred.get('confidence', 0.5)
                            ml_probs.append(prob)
                        except:
                            ml_probs.append(0.5)
                    else:
                        ml_probs.append(0.5)
                self.df['ml_probability'] = ml_probs
            except Exception as e:
                logger.warning(f"Could not generate ML predictions: {e}")
                self.df['ml_probability'] = 0.5
        else:
            self.df['ml_probability'] = 0.5
    
    def _get_state_dimension(self) -> int:
        """Calculate state dimension."""
        # Core technical indicators
        tech_indicators = [
            'ema_12', 'ema_26', 'sma_20', 'sma_50',
            'adx', 'plus_di', 'minus_di', 'atr',
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Price features (momentum, volatility)
        price_features = [
            'price_change', 'momentum_5', 'momentum_10', 'volatility_10'
        ]
        
        # ML probability
        ml_features = ['ml_probability']
        
        # Position info
        position_features = ['position_size', 'unrealized_pnl_pct', 'entry_price_rel']
        
        # ML and trend filter signals (for conflict resolution)
        signal_features = ['ml_signal', 'ml_confidence', 'trend_signal', 'signal_conflict']
        
        # Lookback window features (recent price changes)
        lookback_features = [f'price_change_lag_{i}' for i in range(1, min(10, self.config.lookback_window))]
        
        all_features = tech_indicators + price_features + ml_features + position_features + signal_features + lookback_features
        
        return len(all_features)
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State vector as numpy array
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        row = self.df.iloc[self.current_step]
        state = []
        
        # Technical indicators (normalized)
        tech_features = [
            'ema_12', 'ema_26', 'sma_20', 'sma_50',
            'adx', 'plus_di', 'minus_di', 'atr',
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        current_price = row['close']
        for feat in tech_features:
            if feat in row:
                # Normalize by price for price-based indicators
                if feat in ['ema_12', 'ema_26', 'sma_20', 'sma_50', 'atr']:
                    val = (row[feat] / current_price) if current_price > 0 else 0
                else:
                    val = row[feat]
                state.append(float(val))
            else:
                state.append(0.0)
        
        # Price features
        price_features = ['price_change', 'momentum_5', 'momentum_10', 'volatility_10']
        for feat in price_features:
            if feat in row:
                state.append(float(row[feat]))
            else:
                state.append(0.0)
        
        # ML probability
        ml_prob = row.get('ml_probability', 0.5)
        state.append(float(ml_prob))
        
        # Position info
        state.append(float(self.position_size))  # Current position size
        
        # Unrealized PnL percentage
        if self.position_size != 0 and self.entry_price > 0:
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position_size
        else:
            unrealized_pnl_pct = 0.0
        state.append(float(unrealized_pnl_pct))
        
        # Entry price relative to current
        if self.entry_price > 0:
            entry_price_rel = (current_price - self.entry_price) / self.entry_price
        else:
            entry_price_rel = 0.0
        state.append(float(entry_price_rel))
        
        # ML signal and confidence
        ml_signal_val = 0.0  # -1 (sell) to +1 (buy)
        ml_confidence_val = 0.5
        
        if self.ml_model and self.ml_model.is_trained:
            try:
                window_df = self.df.iloc[max(0, self.current_step-100):self.current_step+1]
                ml_pred = self.ml_model.predict(window_df)
                if ml_pred.get('prediction', 0) == 1:  # Buy signal
                    ml_signal_val = 1.0
                else:  # Sell signal
                    ml_signal_val = -1.0
                ml_confidence_val = ml_pred.get('confidence', 0.5)
            except:
                pass
        
        state.append(float(ml_signal_val))
        state.append(float(ml_confidence_val))
        
        # Trend filter signal
        trend_signal_val = 0.0  # -1 (sell) to +1 (buy)
        if 'ema_12' in row and 'ema_26' in row:
            if row['ema_12'] > row['ema_26']:
                trend_signal_val = 1.0  # Uptrend
            elif row['ema_12'] < row['ema_26']:
                trend_signal_val = -1.0  # Downtrend
        
        state.append(float(trend_signal_val))
        
        # Signal conflict indicator (ML says buy but trend says sell, etc.)
        signal_conflict = abs(ml_signal_val - trend_signal_val) / 2.0  # 0 = no conflict, 1 = complete conflict
        state.append(float(signal_conflict))
        
        # Recent price changes (lookback)
        lookback = min(10, self.config.lookback_window)
        for i in range(1, lookback):
            if self.current_step >= i:
                lookback_row = self.df.iloc[self.current_step - i]
                price_change = (lookback_row['close'] - self.df.iloc[self.current_step - i - 1]['close']) / self.df.iloc[self.current_step - i - 1]['close'] if self.current_step > i else 0
            else:
                price_change = 0.0
            state.append(float(price_change))
        
        state_array = np.array(state, dtype=np.float32)
        
        # Handle NaN and Inf
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state_array
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0.0
        return (returns_array.mean() / returns_array.std()) * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (only downside volatility)."""
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        downside_std = downside_returns.std()
        return (returns_array.mean() / downside_std) * np.sqrt(252)  # Annualized
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate reward based on risk-adjusted returns (Sharpe/Sortino) and penalties.
        
        Args:
            action: Action taken (0=hold, 1=buy_25%, 2=buy_50%, 3=buy_100%, 4=sell_25%, 5=sell_50%, 6=sell_100%)
            current_price: Current asset price
            
        Returns:
            Reward value
        """
        # Calculate current equity
        portfolio_value = self.portfolio.get_portfolio_value({'BTC/USD': current_price})
        current_equity = portfolio_value['total_value']
        
        # Calculate return
        equity_change = current_equity - self.previous_equity
        equity_change_pct = (equity_change / self.previous_equity) if self.previous_equity > 0 else 0
        
        # Update history
        self.returns_history.append(equity_change_pct)
        self.equity_history.append(current_equity)
        
        # Keep only recent history for Sharpe/Sortino calculation
        if len(self.returns_history) > self.config.reward_lookback:
            self.returns_history = self.returns_history[-self.config.reward_lookback:]
        
        # Update max equity and drawdown
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        current_drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate risk-adjusted return (Sharpe or Sortino)
        if self.config.use_sortino_reward and len(self.returns_history) >= 5:
            risk_adjusted_return = self._calculate_sortino_ratio(self.returns_history) / 100.0  # Normalize
        elif self.config.use_sharpe_reward and len(self.returns_history) >= 5:
            risk_adjusted_return = self._calculate_sharpe_ratio(self.returns_history) / 10.0  # Normalize
        else:
            # Fallback to simple return if not enough history
            risk_adjusted_return = equity_change_pct
        
        # Fee penalty (account for trading costs)
        fee_penalty = 0.0
        if action != 0:  # Any trading action
            # Estimate fee based on position change
            portfolio_value_for_trade = self.previous_equity
            fee_penalty = self.config.fee_penalty_weight * (portfolio_value_for_trade * self.config.commission_rate)
        
        # Risk penalty (volatility and drawdown)
        volatility = np.std(self.returns_history[-10:]) if len(self.returns_history) >= 10 else 0
        risk_penalty = self.config.risk_penalty_weight * (current_drawdown + volatility)
        
        # Whipsaw penalty (rapid buy/sell reversals)
        whipsaw_penalty = 0.0
        if len(self.last_actions) >= 2:
            last_action = self.last_actions[-1]
            # Check for rapid reversal (buy then sell, or sell then buy)
            if (action in [1, 2, 3] and last_action in [4, 5, 6]) or \
               (action in [4, 5, 6] and last_action in [1, 2, 3]):
                whipsaw_penalty = self.config.whipsaw_penalty_weight
        
        # Transaction penalty (encourages holding)
        transaction_penalty = 0.0
        if action != 0 and (not hasattr(self, '_last_action') or action != self._last_action):
            transaction_penalty = self.config.transaction_penalty
        
        # Drawdown penalty
        drawdown_penalty = current_drawdown * self.config.max_drawdown_penalty
        
        # Trend bonus (bonus for holding during strong trends)
        trend_bonus = 0.0
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            adx = row.get('adx', 0)
            if adx > 25:  # Strong trend
                if self.position_size > 0 and row.get('ema_12', 0) > row.get('ema_26', 0):  # Long in uptrend
                    trend_bonus = self.config.trend_bonus * 0.01
                elif self.position_size < 0 and row.get('ema_12', 0) < row.get('ema_26', 0):  # Short in downtrend
                    trend_bonus = self.config.trend_bonus * 0.01
        
        # Total reward: risk-adjusted return minus all penalties, plus bonuses
        reward = risk_adjusted_return - fee_penalty - risk_penalty - whipsaw_penalty - transaction_penalty - drawdown_penalty + trend_bonus
        
        # Update previous equity
        self.previous_equity = current_equity
        
        return float(reward)
    
    def _execute_action(self, action: int, current_price: float):
        """
        Execute trading action with expanded action space.
        
        Args:
            action: Action to take:
                0 = hold
                1 = buy 25%
                2 = buy 50%
                3 = buy 100%
                4 = sell 25%
                5 = sell 50%
                6 = sell 100%
            current_price: Current asset price
        """
        # Determine target position based on action
        if action == 0:  # Hold
            target_position = self.position_size
        elif action == 1:  # Buy 25%
            target_position = min(self.position_size + 0.25, self.config.position_size_limit)
        elif action == 2:  # Buy 50%
            target_position = min(self.position_size + 0.50, self.config.position_size_limit)
        elif action == 3:  # Buy 100%
            target_position = self.config.position_size_limit
        elif action == 4:  # Sell 25%
            target_position = max(self.position_size - 0.25, -self.config.position_size_limit)
        elif action == 5:  # Sell 50%
            target_position = max(self.position_size - 0.50, -self.config.position_size_limit)
        elif action == 6:  # Sell 100%
            target_position = -self.config.position_size_limit
        else:
            target_position = 0
        
        # Calculate position change
        position_change = target_position - self.position_size
        
        if abs(position_change) > 0.01:  # Only trade if significant change
            # Calculate trade size
            portfolio_value = self.portfolio.get_portfolio_value({'BTC/USD': current_price})
            total_value = portfolio_value['total_value']
            
            trade_value = abs(position_change) * total_value
            trade_size = trade_value / current_price
            
            # Apply slippage and commission
            if position_change > 0:  # Buying
                execution_price = current_price * (1 + self.config.slippage)
            else:  # Selling
                execution_price = current_price * (1 - self.config.slippage)
            
            commission = trade_value * self.config.commission_rate
            
            # Update position
            if self.position_size == 0:
                # Opening new position
                self.entry_price = execution_price
            elif (self.position_size > 0 and position_change < 0) or (self.position_size < 0 and position_change > 0):
                # Closing or reversing position
                self.entry_price = execution_price
            
            self.position_size = target_position
            
            # Update portfolio (simplified - actual implementation would track positions)
            self.portfolio.cash_balance -= commission
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.previous_equity = self.config.initial_cash
        self.max_equity = self.config.initial_cash
        self.max_drawdown = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.portfolio = PortfolioManager(initial_cash=self.config.initial_cash)
        
        # Reset tracking variables
        self.returns_history = []
        self.equity_history = [self.config.initial_cash]
        self.last_actions = []
        
        state = self._get_state()
        info = {
            'step': self.current_step,
            'equity': self.previous_equity,
            'position': self.position_size
        }
        
        return state, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_step >= len(self.df) - 1:
            # End of episode
            state = self._get_state()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'step': self.current_step,
                'equity': self.previous_equity,
                'position': self.position_size,
                'max_drawdown': self.max_drawdown
            }
            return state, reward, terminated, truncated, info
        
        # Convert continuous action to discrete if needed
        if self.action_type == 'continuous':
            action_val = action[0]
            if action_val < -0.66:
                discrete_action = 6  # Sell 100%
            elif action_val < -0.33:
                discrete_action = 5  # Sell 50%
            elif action_val < 0:
                discrete_action = 4  # Sell 25%
            elif action_val == 0:
                discrete_action = 0  # Hold
            elif action_val <= 0.33:
                discrete_action = 1  # Buy 25%
            elif action_val <= 0.66:
                discrete_action = 2  # Buy 50%
            else:
                discrete_action = 3  # Buy 100%
        else:
            discrete_action = int(action)
        
        # Track actions for whipsaw penalty
        self.last_actions.append(discrete_action)
        if len(self.last_actions) > 5:
            self.last_actions = self.last_actions[-5:]
        
        # Get current price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action
        self._execute_action(discrete_action, current_price)
        
        # Calculate reward
        reward = self._calculate_reward(discrete_action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Get new state
        state = self._get_state()
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Store last action for transaction penalty
        self._last_action = discrete_action
        
        info = {
            'step': self.current_step,
            'equity': self.previous_equity,
            'position': self.position_size,
            'max_drawdown': self.max_drawdown,
            'price': current_price
        }
        
        return state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            portfolio_value = self.portfolio.get_portfolio_value({'BTC/USD': self.df.iloc[self.current_step]['close']})
            print(f"Step: {self.current_step}/{self.max_steps}, "
                  f"Equity: ${portfolio_value['total_value']:.2f}, "
                  f"Position: {self.position_size:.2f}, "
                  f"Drawdown: {self.max_drawdown:.2%}")

