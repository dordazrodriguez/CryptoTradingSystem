"""
Virtual portfolio manager for cryptocurrency trading bot.
Handles position tracking, PnL calculation, and trade execution simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages virtual portfolio for cryptocurrency trading."""
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize portfolio manager.
        
        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions = {}  # symbol -> {'quantity': float, 'avg_cost': float}
        self.db_manager = get_db_manager()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
        logger.info(f"Portfolio manager initialized with ${initial_cash:,.2f}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Portfolio value breakdown
        """
        invested_value = 0.0
        unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                market_value = position['quantity'] * current_price
                cost_basis = position['quantity'] * position['avg_cost']
                
                invested_value += market_value
                unrealized_pnl += (market_value - cost_basis)
        
        total_value = self.cash_balance + invested_value
        total_pnl = unrealized_pnl + self.get_realized_pnl()
        
        return {
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'invested_value': invested_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.get_realized_pnl(),
            'total_pnl': total_pnl,
            'total_return': (total_value - self.initial_cash) / self.initial_cash,
            'positions': self.positions.copy()
        }
    
    def get_realized_pnl(self) -> float:
        """Get total realized PnL from closed trades."""
        query = """
        SELECT SUM(net_amount) as total_pnl 
        FROM trades 
        WHERE side = 'sell'
        """
        result = self.db_manager.execute_query(query)
        return result[0]['total_pnl'] if result and result[0]['total_pnl'] else 0.0
    
    def can_buy(self, symbol: str, quantity: float, price: float, commission: float = 0.001) -> bool:
        """
        Check if we can afford to buy the specified quantity.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            price: Price per unit
            commission: Commission rate (default: 0.1%)
            
        Returns:
            True if we can afford the trade
        """
        total_cost = (quantity * price) * (1 + commission)
        return total_cost <= self.cash_balance
    
    def can_sell(self, symbol: str, quantity: float) -> bool:
        """
        Check if we have enough quantity to sell.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            
        Returns:
            True if we have enough quantity
        """
        if symbol not in self.positions:
            return False
        return self.positions[symbol]['quantity'] >= quantity
    
    def buy(self, symbol: str, quantity: float, price: float, 
            commission: float = 0.001, strategy: str = None) -> Dict[str, Any]:
        """
        Execute a buy order.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            price: Price per unit
            commission: Commission rate
            strategy: Strategy name
            
        Returns:
            Trade execution result
        """
        if not self.can_buy(symbol, quantity, price, commission):
            return {
                'success': False,
                'error': 'Insufficient cash balance',
                'required': (quantity * price) * (1 + commission),
                'available': self.cash_balance
            }
        
        # Calculate costs
        gross_amount = quantity * price
        commission_fee = gross_amount * commission
        net_amount = gross_amount + commission_fee
        
        # Update cash balance
        self.cash_balance -= net_amount
        
        # Update position
        if symbol in self.positions:
            # Average cost calculation
            old_quantity = self.positions[symbol]['quantity']
            old_avg_cost = self.positions[symbol]['avg_cost']
            
            new_quantity = old_quantity + quantity
            new_avg_cost = ((old_quantity * old_avg_cost) + gross_amount) / new_quantity
            
            self.positions[symbol] = {
                'quantity': new_quantity,
                'avg_cost': new_avg_cost
            }
        else:
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_cost': price
            }
        
        # Record trade
        trade_id = f"buy_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order_id = f"order_{trade_id}"
        
        # Insert order
        self.db_manager.execute_insert("""
            INSERT INTO orders 
            (order_id, symbol, side, order_type, quantity, price, status, filled_quantity, 
             filled_price, commission, strategy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, symbol, 'buy', 'market', quantity, price, 'filled', 
              quantity, price, commission_fee, strategy, 
              datetime.now(timezone.utc).isoformat(), 
              datetime.now(timezone.utc).isoformat()))
        
        # Insert trade
        self.db_manager.execute_insert("""
            INSERT INTO trades 
            (trade_id, order_id, symbol, side, quantity, price, commission, net_amount, timestamp, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, order_id, symbol, 'buy', quantity, price, commission_fee, 
              net_amount, datetime.now(timezone.utc).isoformat(), strategy))
        
        # Update position in database
        self.db_manager.update_position(
            symbol=symbol,
            quantity=self.positions[symbol]['quantity'],
            avg_cost=self.positions[symbol]['avg_cost']
        )
        
        # Update statistics
        self.total_trades += 1
        self.total_fees += commission_fee
        
        logger.info(f"Buy order executed: {quantity} {symbol} at ${price:.2f}")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'buy',
            'quantity': quantity,
            'price': price,
            'commission': commission_fee,
            'net_amount': net_amount,
            'remaining_cash': self.cash_balance
        }
    
    def sell(self, symbol: str, quantity: float, price: float, 
             commission: float = 0.001, strategy: str = None) -> Dict[str, Any]:
        """
        Execute a sell order.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            price: Price per unit
            commission: Commission rate
            strategy: Strategy name
            
        Returns:
            Trade execution result
        """
        if not self.can_sell(symbol, quantity):
            return {
                'success': False,
                'error': 'Insufficient quantity',
                'required': quantity,
                'available': self.positions[symbol]['quantity'] if symbol in self.positions else 0
            }
        
        # Calculate proceeds
        gross_amount = quantity * price
        commission_fee = gross_amount * commission
        net_amount = gross_amount - commission_fee
        
        # Calculate PnL
        cost_basis = quantity * self.positions[symbol]['avg_cost']
        realized_pnl = net_amount - cost_basis
        
        # Update cash balance
        self.cash_balance += net_amount
        
        # Update position
        self.positions[symbol]['quantity'] -= quantity
        if self.positions[symbol]['quantity'] <= 0:
            del self.positions[symbol]
        
        # Record trade
        trade_id = f"sell_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order_id = f"order_{trade_id}"
        
        # Insert order
        self.db_manager.execute_insert("""
            INSERT INTO orders 
            (order_id, symbol, side, order_type, quantity, price, status, filled_quantity, 
             filled_price, commission, strategy, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, symbol, 'sell', 'market', quantity, price, 'filled', 
              quantity, price, commission_fee, strategy,
              datetime.now(timezone.utc).isoformat(), 
              datetime.now(timezone.utc).isoformat()))
        
        # Insert trade
        self.db_manager.execute_insert("""
            INSERT INTO trades 
            (trade_id, order_id, symbol, side, quantity, price, commission, net_amount, timestamp, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, order_id, symbol, 'sell', quantity, price, commission_fee, 
              net_amount, datetime.now(timezone.utc).isoformat(), strategy))
        
        # Update position in database
        if symbol in self.positions:
            self.db_manager.update_position(
                symbol=symbol,
                quantity=self.positions[symbol]['quantity'],
                avg_cost=self.positions[symbol]['avg_cost']
            )
        else:
            # Position closed, remove from database
            self.db_manager.execute_update(
                "DELETE FROM positions WHERE symbol = ?", (symbol,)
            )
        
        # Update statistics
        self.total_trades += 1
        self.total_fees += commission_fee
        
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"Sell order executed: {quantity} {symbol} at ${price:.2f}, PnL: ${realized_pnl:.2f}")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': symbol,
            'side': 'sell',
            'quantity': quantity,
            'price': price,
            'commission': commission_fee,
            'net_amount': net_amount,
            'realized_pnl': realized_pnl,
            'remaining_cash': self.cash_balance
        }
    
    def get_position_size(self, symbol: str, price: float, target_percent: float = 0.1) -> float:
        """
        Calculate position size based on target portfolio percentage.
        
        Args:
            symbol: Trading symbol
            price: Current price
            target_percent: Target percentage of portfolio
            
        Returns:
            Recommended quantity to buy
        """
        portfolio_value = self.get_portfolio_value({symbol: price})['total_value']
        target_value = portfolio_value * target_percent
        
        # Account for commission
        commission_rate = 0.001
        gross_value = target_value / (1 + commission_rate)
        
        return gross_value / price
    
    def rebalance_portfolio(self, target_allocations: Dict[str, float], 
                          current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Rebalance portfolio to target allocations.
        
        Args:
            target_allocations: Dictionary of symbol -> target percentage
            current_prices: Current prices for all symbols
            
        Returns:
            List of trades executed
        """
        trades = []
        portfolio_value = self.get_portfolio_value(current_prices)['total_value']
        
        for symbol, target_percent in target_allocations.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            target_value = portfolio_value * target_percent
            
            if symbol in self.positions:
                current_value = self.positions[symbol]['quantity'] * current_price
            else:
                current_value = 0
            
            value_diff = target_value - current_value
            
            if abs(value_diff) > portfolio_value * 0.01:  # 1% threshold
                if value_diff > 0:
                    # Need to buy more
                    quantity = value_diff / current_price
                    if self.can_buy(symbol, quantity, current_price):
                        result = self.buy(symbol, quantity, current_price, strategy='rebalance')
                        if result['success']:
                            trades.append(result)
                else:
                    # Need to sell
                    quantity = abs(value_diff) / current_price
                    if self.can_sell(symbol, quantity):
                        result = self.sell(symbol, quantity, current_price, strategy='rebalance')
                        if result['success']:
                            trades.append(result)
        
        return trades
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            current_prices: Current prices for all positions
            
        Returns:
            Performance metrics
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Basic metrics
        total_return = portfolio_value['total_return']
        total_pnl = portfolio_value['total_pnl']
        
        # Trade statistics
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe ratio (simplified)
        # In a real implementation, you'd calculate this from historical returns
        sharpe_ratio = total_return / 0.1 if total_return > 0 else 0  # Assuming 10% volatility
        
        # Calculate maximum drawdown (simplified)
        # In a real implementation, you'd track this over time
        max_drawdown = 0.05  # Placeholder
        
        return {
            'total_value': portfolio_value['total_value'],
            'total_return': total_return,
            'total_pnl': total_pnl,
            'cash_balance': portfolio_value['cash_balance'],
            'invested_value': portfolio_value['invested_value'],
            'unrealized_pnl': portfolio_value['unrealized_pnl'],
            'realized_pnl': portfolio_value['realized_pnl'],
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_fees': self.total_fees,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positions_count': len(self.positions)
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trade history.
        
        Args:
            limit: Number of recent trades to retrieve
            
        Returns:
            List of trade records
        """
        query = """
        SELECT * FROM trades 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        return self.db_manager.execute_query(query, (limit,))
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.cash_balance = self.initial_cash
        self.positions = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
        # Clear database
        self.db_manager.execute_update("DELETE FROM positions")
        self.db_manager.execute_update("DELETE FROM orders")
        self.db_manager.execute_update("DELETE FROM trades")
        
        logger.info("Portfolio reset to initial state")
