"""
Risk management system for cryptocurrency trading bot.
Implements stop-loss, position sizing, portfolio limits, and risk controls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk controls and position sizing for trading."""
    
    def __init__(self, max_position_percent: float = 0.2, max_portfolio_risk: float = 0.1,
                 stop_loss_percent: float = 0.05, max_leverage: float = 1.0):
        """
        Initialize risk manager.
        
        Args:
            max_position_percent: Maximum percentage of portfolio per position
            max_portfolio_risk: Maximum portfolio risk (VaR)
            stop_loss_percent: Default stop-loss percentage
            max_leverage: Maximum leverage allowed
        """
        self.max_position_percent = max_position_percent
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_percent = stop_loss_percent
        self.max_leverage = max_leverage
        
        self.db_manager = get_db_manager()
        
        # Risk tracking
        self.daily_loss_limit = 0.02  # 2% daily loss limit
        self.max_drawdown_limit = 0.15  # 15% maximum drawdown
        self.daily_trades_limit = 50  # Maximum trades per day
        
        # Current day tracking
        self.current_day = datetime.now().date()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        logger.info(f"Risk manager initialized with max position: {max_position_percent*100}%, "
                   f"stop-loss: {stop_loss_percent*100}%")
    
    def check_position_size(self, symbol: str, quantity: float, price: float, 
                           portfolio_value: float) -> Dict[str, Any]:
        """
        Check if position size is within risk limits.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            price: Price per unit
            portfolio_value: Current portfolio value
            
        Returns:
            Risk check result
        """
        position_value = quantity * price
        position_percent = position_value / portfolio_value
        
        if position_percent > self.max_position_percent:
            return {
                'approved': False,
                'reason': 'Position size exceeds maximum allowed',
                'current_percent': position_percent,
                'max_allowed': self.max_position_percent,
                'recommended_quantity': (portfolio_value * self.max_position_percent) / price
            }
        
        return {
            'approved': True,
            'position_percent': position_percent,
            'position_value': position_value
        }
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                              volatility: float = 0.02, confidence_level: float = 0.95,
                              ml_confidence: Optional[float] = None, atr: Optional[float] = None) -> float:
        """
        Calculate dynamic position size based on confidence and volatility.
        
        Args:
            symbol: Trading symbol
            price: Current price
            portfolio_value: Portfolio value
            volatility: Asset volatility (annual)
            confidence_level: Confidence level for VaR calculation
            ml_confidence: ML model confidence (0-1, higher = larger position)
            atr: Average True Range (for volatility-based sizing)
            
        Returns:
            Recommended position size (quantity)
        """
        # Base position size from max_position_percent
        base_position_percent = self.max_position_percent
        
        # Adjust based on ML confidence
        confidence_multiplier = 1.0
        if ml_confidence is not None:
            # Scale from 0.5 (low confidence) to 1.0 (high confidence)
            # Low confidence (0.6) -> 0.5x, High confidence (0.93) -> 1.0x
            confidence_multiplier = 0.5 + (ml_confidence - 0.5) * 1.25  # Map 0.5-1.0 to 0.5-1.0
            confidence_multiplier = max(0.3, min(1.0, confidence_multiplier))  # Clamp between 0.3 and 1.0
        
        # Adjust based on volatility/ATR
        volatility_multiplier = 1.0
        if atr is not None and atr > 0 and price > 0:
            # Normalize ATR as percentage of price
            atr_percent = atr / price
            # High volatility (ATR > 2%) -> reduce position size
            # Low volatility (ATR < 0.5%) -> increase position size
            if atr_percent > 0.02:  # High volatility
                volatility_multiplier = 0.5  # Reduce position by 50%
            elif atr_percent < 0.005:  # Low volatility
                volatility_multiplier = 1.2  # Increase position by 20%
            else:
                # Linear interpolation between 0.5 and 1.2
                volatility_multiplier = 0.5 + (0.02 - atr_percent) / 0.015 * 0.7
                volatility_multiplier = max(0.5, min(1.2, volatility_multiplier))
        elif volatility > 0.03:  # High volatility fallback
            volatility_multiplier = 0.5
        
        # Calculate final position percentage
        position_percent = base_position_percent * confidence_multiplier * volatility_multiplier
        position_percent = min(position_percent, self.max_position_percent)  # Cap at max
        
        # Calculate position value and quantity
        position_value = portfolio_value * position_percent
        quantity = position_value / price
        
        return quantity
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                          volatility: float = 0.02, atr: Optional[float] = None,
                          atr_multiplier: float = 2.5) -> float:
        """
        Calculate dynamic ATR-based stop-loss price.
        
        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            volatility: Asset volatility (fallback if ATR not provided)
            atr: Average True Range (preferred for dynamic stops)
            atr_multiplier: Multiplier for ATR (default 2.5)
            
        Returns:
            Stop-loss price
        """
        # Use ATR-based stop loss (preferred method)
        if atr is not None and atr > 0:
            stop_distance = atr * atr_multiplier
        else:
            # Fallback to volatility-based stop
            stop_distance = entry_price * volatility * atr_multiplier
        
        if side == 'buy':
            stop_loss = entry_price - stop_distance
        else:  # sell
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               side: str, atr: Optional[float] = None,
                               atr_multiplier: float = 2.5,
                               trailing_percent: Optional[float] = None) -> float:
        """
        Calculate trailing stop-loss that moves with price.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            side: 'buy' or 'sell'
            atr: Average True Range
            atr_multiplier: Multiplier for ATR
            trailing_percent: Percentage-based trailing stop (alternative to ATR)
            
        Returns:
            Trailing stop-loss price
        """
        if side == 'buy':
            # For long positions, trailing stop moves up with price
            if atr is not None and atr > 0:
                stop_distance = atr * atr_multiplier
                trailing_stop = current_price - stop_distance
            elif trailing_percent is not None:
                trailing_stop = current_price * (1 - trailing_percent)
            else:
                # Default: 2% trailing stop
                trailing_stop = current_price * 0.98
            
            # Trailing stop should never move down
            if hasattr(self, '_last_trailing_stop') and self._last_trailing_stop is not None:
                trailing_stop = max(trailing_stop, self._last_trailing_stop)
            
            self._last_trailing_stop = trailing_stop
            return trailing_stop
        else:
            # For short positions, trailing stop moves down with price
            if atr is not None and atr > 0:
                stop_distance = atr * atr_multiplier
                trailing_stop = current_price + stop_distance
            elif trailing_percent is not None:
                trailing_stop = current_price * (1 + trailing_percent)
            else:
                # Default: 2% trailing stop
                trailing_stop = current_price * 1.02
            
            # Trailing stop should never move up
            if hasattr(self, '_last_trailing_stop') and self._last_trailing_stop is not None:
                trailing_stop = min(trailing_stop, self._last_trailing_stop)
            
            self._last_trailing_stop = trailing_stop
            return trailing_stop
    
    def check_stop_loss(self, symbol: str, current_price: float, 
                       entry_price: float, side: str, 
                       atr: Optional[float] = None,
                       trailing_stop: Optional[float] = None) -> Dict[str, Any]:
        """
        Check if stop-loss should be triggered (supports both static and ATR-based stops).
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Entry price
            side: 'buy' or 'sell'
            atr: Average True Range (for dynamic stops)
            trailing_stop: Trailing stop price (if using trailing stops)
            
        Returns:
            Stop-loss check result
        """
        # Use trailing stop if provided, otherwise calculate ATR-based or static stop
        if trailing_stop is not None:
            stop_loss_price = trailing_stop
        elif atr is not None and atr > 0:
            # Use ATR-based stop
            stop_loss_price = self.calculate_stop_loss(entry_price, side, atr=atr)
        else:
            # Use static percentage-based stop
            if side == 'buy':
                stop_loss_price = entry_price * (1 - self.stop_loss_percent)
            else:
                stop_loss_price = entry_price * (1 + self.stop_loss_percent)
        
        # Check if triggered
        if side == 'buy':
            triggered = current_price <= stop_loss_price
            loss_percent = (current_price - entry_price) / entry_price
        else:
            triggered = current_price >= stop_loss_price
            loss_percent = (entry_price - current_price) / entry_price
        
        return {
            'triggered': triggered,
            'stop_loss_price': stop_loss_price,
            'current_price': current_price,
            'loss_percent': loss_percent,
            'side': side,
            'stop_type': 'trailing' if trailing_stop is not None else ('atr' if atr is not None else 'static')
        }
    
    def check_portfolio_risk(self, positions: Dict[str, Dict], 
                           current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Check overall portfolio risk.
        
        Args:
            positions: Current positions
            current_prices: Current market prices
            
        Returns:
            Portfolio risk assessment
        """
        total_value = 0
        total_exposure = 0
        position_risks = {}
        
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                total_value += position_value
                total_exposure += position_value
                
                # Calculate individual position risk
                position_risks[symbol] = {
                    'value': position_value,
                    'percent': position_value / total_value if total_value > 0 else 0,
                    'unrealized_pnl': position_value - (position['quantity'] * position['avg_cost'])
                }
        
        # Calculate portfolio VaR (simplified)
        portfolio_var = self.calculate_portfolio_var(positions, current_prices)
        
        # Check concentration risk
        max_position_percent = max([risk['percent'] for risk in position_risks.values()]) if position_risks else 0
        
        return {
            'total_value': total_value,
            'total_exposure': total_exposure,
            'portfolio_var': portfolio_var,
            'max_position_percent': max_position_percent,
            'concentration_risk': max_position_percent > self.max_position_percent,
            'position_risks': position_risks,
            'risk_within_limits': portfolio_var <= self.max_portfolio_risk
        }
    
    def calculate_portfolio_var(self, positions: Dict[str, Dict], 
                              current_prices: Dict[str, float], 
                              confidence_level: float = 0.95) -> float:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            positions: Current positions
            current_prices: Current market prices
            confidence_level: Confidence level for VaR
            
        Returns:
            Portfolio VaR
        """
        if not positions:
            return 0.0
        
        # Simplified VaR calculation
        # In practice, you'd use historical data and correlation matrices
        
        total_value = 0
        weighted_volatility = 0
        
        for symbol, position in positions.items():
            if symbol in current_prices:
                position_value = position['quantity'] * current_prices[symbol]
                total_value += position_value
                
                # Assume 2% daily volatility for crypto
                asset_volatility = 0.02
                weight = position_value / total_value if total_value > 0 else 0
                weighted_volatility += weight * asset_volatility
        
        # Calculate VaR (simplified)
        z_score = 1.645  # 95% confidence level
        var = total_value * weighted_volatility * z_score
        
        return var
    
    def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check daily trading limits.
        
        Returns:
            Daily limits status
        """
        today = datetime.now().date()
        
        # Reset daily counters if new day
        if today != self.current_day:
            self.current_day = today
            self.daily_trades = 0
            self.daily_pnl = 0.0
        
        # Check daily trade limit
        trades_exceeded = self.daily_trades >= self.daily_trades_limit
        
        # Check daily loss limit
        daily_loss_exceeded = self.daily_pnl <= -self.daily_loss_limit
        
        return {
            'daily_trades': self.daily_trades,
            'daily_trades_limit': self.daily_trades_limit,
            'trades_exceeded': trades_exceeded,
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': -self.daily_loss_limit,
            'loss_exceeded': daily_loss_exceeded,
            'can_trade': not (trades_exceeded or daily_loss_exceeded)
        }
    
    def update_daily_trade(self, pnl: float):
        """
        Update daily trade statistics.
        
        Args:
            pnl: Profit/loss from the trade
        """
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        logger.info(f"Daily trade #{self.daily_trades}, PnL: ${pnl:.2f}, Total daily PnL: ${self.daily_pnl:.2f}")
    
    def check_max_drawdown(self, current_value: float, peak_value: float) -> Dict[str, Any]:
        """
        Check maximum drawdown limit.
        
        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            Drawdown check result
        """
        if peak_value <= 0:
            return {'drawdown_percent': 0, 'within_limits': True}
        
        drawdown_percent = (peak_value - current_value) / peak_value
        exceeded = drawdown_percent > self.max_drawdown_limit
        
        return {
            'drawdown_percent': drawdown_percent,
            'max_drawdown_limit': self.max_drawdown_limit,
            'exceeded': exceeded,
            'within_limits': not exceeded
        }
    
    def get_risk_report(self, positions: Dict[str, Dict], 
                       current_prices: Dict[str, float],
                       portfolio_value: float) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            positions: Current positions
            current_prices: Current market prices
            portfolio_value: Total portfolio value
            
        Returns:
            Comprehensive risk report
        """
        # Portfolio risk
        portfolio_risk = self.check_portfolio_risk(positions, current_prices)
        
        # Daily limits
        daily_limits = self.check_daily_limits()
        
        # Position analysis
        position_analysis = {}
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                
                position_analysis[symbol] = {
                    'quantity': position['quantity'],
                    'avg_cost': position['avg_cost'],
                    'current_price': current_price,
                    'position_value': position_value,
                    'position_percent': position_value / portfolio_value,
                    'unrealized_pnl': position_value - (position['quantity'] * position['avg_cost']),
                    'unrealized_pnl_percent': (position_value - (position['quantity'] * position['avg_cost'])) / (position['quantity'] * position['avg_cost'])
                }
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'portfolio_value': portfolio_value,
            'portfolio_risk': portfolio_risk,
            'daily_limits': daily_limits,
            'position_analysis': position_analysis,
            'risk_settings': {
                'max_position_percent': self.max_position_percent,
                'max_portfolio_risk': self.max_portfolio_risk,
                'stop_loss_percent': self.stop_loss_percent,
                'max_leverage': self.max_leverage,
                'daily_loss_limit': self.daily_loss_limit,
                'max_drawdown_limit': self.max_drawdown_limit,
                'daily_trades_limit': self.daily_trades_limit
            },
            'overall_risk_status': {
                'can_trade': daily_limits['can_trade'],
                'risk_within_limits': portfolio_risk['risk_within_limits'],
                'concentration_risk': portfolio_risk['concentration_risk']
            }
        }
    
    def log_risk_event(self, event_type: str, symbol: str, details: Dict[str, Any]):
        """
        Log risk management events.
        
        Args:
            event_type: Type of risk event
            symbol: Trading symbol
            details: Event details
        """
        self.db_manager.log_system_event(
            level='WARNING',
            message=f"Risk event: {event_type}",
            module='risk_manager',
            data={
                'event_type': event_type,
                'symbol': symbol,
                'details': details,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    def adjust_risk_parameters(self, **kwargs):
        """
        Adjust risk management parameters.
        
        Args:
            **kwargs: Risk parameters to update
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"Updated risk parameter {param} to {value}")
