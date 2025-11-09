"""
Simple portfolio management implementation from the completed project.
Provides lightweight portfolio tracking for the trading bot.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimplePortfolio:
    """
    Simple portfolio implementation for cryptocurrency trading.
    Tracks cash, positions, and calculates equity.
    """
    cash_usd: float
    position_qty: float = 0.0
    position_avg_price: Optional[float] = None

    def equity(self, last_price: float) -> float:
        """
        Calculate total equity.
        
        Args:
            last_price: Current price of the position
            
        Returns:
            Total equity (cash + position value)
        """
        pos_val = (self.position_qty * last_price) if self.position_qty else 0.0
        return self.cash_usd + pos_val

    def apply_fill(self, side: str, price: float, qty: float, fee: float) -> None:
        """
        Apply a trade fill to the portfolio.
        
        Args:
            side: "buy" or "sell"
            price: Execution price
            qty: Quantity traded
            fee: Transaction fee
        """
        if side == "buy":
            cost = price * qty + fee
            # Update average price using weighted average
            new_qty = self.position_qty + qty
            if new_qty <= 0:
                self.position_avg_price = None
            else:
                if self.position_qty <= 0:
                    self.position_avg_price = price
                else:
                    assert self.position_avg_price is not None
                    self.position_avg_price = (
                        self.position_avg_price * self.position_qty + price * qty
                    ) / new_qty
            self.position_qty = new_qty
            self.cash_usd -= cost
            logger.info(f"Buy fill: qty={qty:.6f} @ ${price:.2f}, cash=${self.cash_usd:.2f}")
            
        elif side == "sell":
            proceeds = price * qty - fee
            self.position_qty -= qty
            if self.position_qty <= 0:
                self.position_avg_price = None
            self.cash_usd += proceeds
            logger.info(f"Sell fill: qty={qty:.6f} @ ${price:.2f}, cash=${self.cash_usd:.2f}")
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
    
    def can_afford(self, price: float, qty: float, fee: float) -> bool:
        """
        Check if portfolio can afford a trade.
        
        Args:
            price: Trade price
            qty: Trade quantity
            fee: Transaction fee
            
        Returns:
            True if affordable, False otherwise
        """
        cost = (price * qty) + fee
        return cost <= self.cash_usd
    
    def get_position_value(self, current_price: float) -> float:
        """
        Get current position value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Position value
        """
        if self.position_qty <= 0:
            return 0.0
        return self.position_qty * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if self.position_avg_price is None or self.position_qty <= 0:
            return 0.0
        return (current_price - self.position_avg_price) * self.position_qty
    
    def get_metrics(self, current_price: float) -> dict:
        """
        Get portfolio metrics.
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with portfolio metrics
        """
        position_value = self.get_position_value(current_price)
        unrealized_pnl = self.get_unrealized_pnl(current_price)
        
        return {
            'cash_usd': self.cash_usd,
            'position_qty': self.position_qty,
            'position_avg_price': self.position_avg_price,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': self.equity(current_price)
        }
    
    def __repr__(self) -> str:
        """String representation of portfolio."""
        return (f"Portfolio(cash=${self.cash_usd:.2f}, "
                f"qty={self.position_qty:.6f}, "
                f"avg_price=${self.position_avg_price or 0:.2f})")
