"""Reward calculation for PPO agent training."""

import numpy as np
from typing import Dict, Any, Optional


class RewardCalculator:
    """Calculates rewards based on trading performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Optional configuration dict with reward settings.
                   Expected keys: transaction_cost_pct, volatility_penalty, 
                   volatility_coef, short_selling_enabled
        """
        if config is None:
            config = {}
        
        reward_config = config.get("reward", {})
        trading_config = config.get("trading", {})
        
        self.transaction_cost_pct = reward_config.get("transaction_cost_pct", 0.001)
        self.volatility_penalty = reward_config.get("volatility_penalty", False)
        self.volatility_coef = reward_config.get("volatility_coef", 0.1)
        
        self.short_selling_enabled = trading_config.get("short_selling_enabled", False)
        self.portfolio_history = []
        self.max_history = 100
    
    def calculate_reward(
        self,
        previous_portfolio_value: float,
        current_portfolio_value: float,
        previous_position_size: float,
        current_position_size: float,
        trade_executed: bool = False
    ) -> float:
        """
        Calculate reward based on portfolio value change and transaction costs.
        
        Args:
            previous_portfolio_value: Portfolio value before action
            current_portfolio_value: Portfolio value after action
            previous_position_size: Position size before action
            current_position_size: Position size after action
            trade_executed: Whether a trade was actually executed
            
        Returns:
            Reward value
        """
        if previous_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            portfolio_return = 0.0
        
        reward = portfolio_return
        
        if trade_executed:
            position_change = abs(current_position_size - previous_position_size)
            if position_change > 0:
                transaction_cost = position_change * self.transaction_cost_pct
                reward -= transaction_cost
        if self.volatility_penalty:
            self.portfolio_history.append(current_portfolio_value)
            if len(self.portfolio_history) > self.max_history:
                self.portfolio_history.pop(0)
            
            if len(self.portfolio_history) >= 10:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                volatility = np.std(returns) if len(returns) > 1 else 0.0
                reward -= self.volatility_coef * volatility
        
        return reward
    
    def calculate_step_reward(
        self,
        previous_portfolio_value: float,
        current_portfolio_value: float,
        previous_position_size: float,
        current_position_size: float,
        previous_action: int,
        current_action: int,
        price_change_pct: float = 0.0,
        entry_price: float = None,
        execution_price: float = None,  # Actual fill price when closing
        current_price: float = None,     # Latest market price
        position_age_bars: int = 0       # Bars since position opened (for reward shaping only)
    ) -> float:
        """
        Calculate reward for a single step, considering action taken.
        
        Args:
            previous_portfolio_value: Portfolio value at previous step
            current_portfolio_value: Portfolio value at current step
            previous_position_size: Position size at previous step
            current_position_size: Position size at current step
            previous_action: Action taken at previous step
            current_action: Action taken at current step
            price_change_pct: Percentage change in price
            entry_price: Entry price of position (for P&L calculation)
            execution_price: Actual fill price when closing position
            current_price: Latest market price
            position_age_bars: Number of bars since position opened
            
        Returns:
            Reward value
        """
        trade_executed = (
            (previous_action != current_action) or
            (abs(current_position_size - previous_position_size) > 1e-6)
        )
        
        reward = self.calculate_reward(
            previous_portfolio_value,
            current_portfolio_value,
            previous_position_size,
            current_position_size,
            trade_executed
        )
        
        if current_action == 1:
            if self.short_selling_enabled:
                if abs(price_change_pct) > 0.01:
                    reward -= 0.0001
            else:
                if price_change_pct > 0.01:
                    reward -= 0.0001
                elif price_change_pct < -0.01:
                    reward += 0.00005
        
        if trade_executed:
            if current_action == 0:
                if price_change_pct > 0:
                    reward += 0.0001
                elif price_change_pct < 0:
                    reward -= 0.0002
            
            elif current_action == 2:
                if price_change_pct < 0:
                    reward += 0.0001
                elif price_change_pct > 0:
                    reward -= 0.0002
        
        if trade_executed and current_action == 2:
            if previous_position_size > 0 and entry_price:
                exit_price = execution_price if execution_price is not None else current_price
                
                if exit_price:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    
                    if pnl_pct < 0:
                        loss_pct = abs(pnl_pct)
                        reward -= 0.01 * loss_pct * 10
                        
                        if position_age_bars < 3:
                            reward -= 0.005
                    
                    elif pnl_pct > 0:
                        if position_age_bars < 3:
                            reward += 0.001 * pnl_pct
                        else:
                            reward += 0.002 * pnl_pct
        
        if current_action == 1 and current_position_size > 0:
            if entry_price and current_price:
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct > 0.01:
                    reward += 0.0001 * pnl_pct
                elif pnl_pct < -0.01:
                    if position_age_bars > 10:
                        reward -= 0.00005
        
        if trade_executed and position_age_bars < 2:
            reward -= 0.001
        
        return reward
    
    def reset(self):
        """Reset reward calculator state."""
        self.portfolio_history = []
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get reward statistics."""
        if len(self.portfolio_history) < 2:
            return {"error": "Insufficient history"}
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        return {
            "mean_return": float(np.mean(returns)) if len(returns) > 0 else 0.0,
            "std_return": float(np.std(returns)) if len(returns) > 0 else 0.0,
            "total_return": float((self.portfolio_history[-1] - self.portfolio_history[0]) / self.portfolio_history[0]) if len(self.portfolio_history) > 0 else 0.0,
        }

