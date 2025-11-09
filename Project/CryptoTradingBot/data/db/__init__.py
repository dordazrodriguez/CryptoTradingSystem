"""
Database initialization and management module for the Crypto Trading Bot.
Handles SQLite database creation, connection, and basic operations.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for the trading bot."""
    
    def __init__(self, db_path: str = "data/db/trading_bot.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database with schema."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()
            
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last row ID.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an UPDATE query and return number of affected rows.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def insert_price_data(self, symbol: str, timestamp: str, open_price: float, 
                         high: float, low: float, close: float, volume: float, 
                         timeframe: str = '1m') -> int:
        """Insert price data into the database."""
        query = """
        INSERT OR REPLACE INTO price_data 
        (symbol, timestamp, open, high, low, close, volume, timeframe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        return self.execute_insert(query, (symbol, timestamp, open_price, high, 
                                         low, close, volume, timeframe))
    
    def insert_indicators(self, symbol: str, timestamp: str, indicators: Dict[str, float],
                         timeframe: str = '1m') -> int:
        """Insert technical indicators into the database."""
        query = """
        INSERT OR REPLACE INTO indicators 
        (symbol, timestamp, timeframe, rsi, macd, macd_signal, macd_histogram,
         sma_20, sma_50, sma_200, ema_12, ema_26, bb_upper, bb_middle, bb_lower, bb_width)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            symbol, timestamp, timeframe,
            indicators.get('rsi'),
            indicators.get('macd'),
            indicators.get('macd_signal'),
            indicators.get('macd_histogram'),
            indicators.get('sma_20'),
            indicators.get('sma_50'),
            indicators.get('sma_200'),
            indicators.get('ema_12'),
            indicators.get('ema_26'),
            indicators.get('bb_upper'),
            indicators.get('bb_middle'),
            indicators.get('bb_lower'),
            indicators.get('bb_width')
        )
        return self.execute_insert(query, params)
    
    def insert_ml_prediction(self, symbol: str, timestamp: str, model_name: str,
                            prediction: float, confidence: float, 
                            features: Optional[Dict] = None) -> int:
        """Insert ML prediction into the database."""
        query = """
        INSERT INTO ml_predictions 
        (symbol, timestamp, model_name, prediction, confidence, features)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        features_json = json.dumps(features) if features else None
        return self.execute_insert(query, (symbol, timestamp, model_name, 
                                         prediction, confidence, features_json))
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest price data for a symbol."""
        query = """
        SELECT * FROM price_data 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        results = self.execute_query(query, (symbol,))
        return results[0] if results else None
    
    def get_price_history(self, symbol: str, limit: int = 100, 
                         timeframe: str = '1m') -> List[Dict[str, Any]]:
        """Get price history for a symbol."""
        query = """
        SELECT * FROM price_data 
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        return self.execute_query(query, (symbol, timeframe, limit))
    
    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """Get current portfolio positions."""
        query = "SELECT * FROM positions"
        return self.execute_query(query)
    
    def update_position(self, symbol: str, quantity: float, avg_cost: float,
                       current_price: float = None) -> int:
        """Update or insert a position."""
        if current_price is None:
            current_price = avg_cost
        
        market_value = quantity * current_price
        unrealized_pnl = market_value - (quantity * avg_cost)
        
        query = """
        INSERT OR REPLACE INTO positions 
        (symbol, quantity, avg_cost, current_price, market_value, unrealized_pnl)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        return self.execute_insert(query, (symbol, quantity, avg_cost, 
                                         current_price, market_value, unrealized_pnl))
    
    def log_system_event(self, level: str, message: str, module: str = None,
                        data: Dict = None) -> int:
        """Log a system event."""
        query = """
        INSERT INTO system_logs (level, message, module, data)
        VALUES (?, ?, ?, ?)
        """
        data_json = json.dumps(data) if data else None
        return self.execute_insert(query, (level, message, module, data_json))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Count records in each table
        tables = ['price_data', 'indicators', 'ml_predictions', 'positions', 
                 'orders', 'trades', 'portfolio_metrics', 'system_logs']
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query)
            stats[f"{table}_count"] = result[0]['count'] if result else 0
        
        # Get database file size
        if os.path.exists(self.db_path):
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        
        return stats

    def insert_trade(self, timestamp_ms: int, side: str, symbol: str,
                     price: float, quantity: float, fee: float,
                     strategy: Optional[str] = None) -> int:
        """
        Insert a filled trade and corresponding order record.

        Args:
            timestamp_ms: Event time in milliseconds since epoch
            side: 'buy' or 'sell'
            symbol: Trading symbol (e.g., 'BTC/USD')
            price: Execution price
            quantity: Filled quantity
            fee: Commission/fee amount (absolute)
            strategy: Optional strategy label

        Returns:
            Row id of the inserted trade
        """
        # Derive identifiers and values
        ts_iso = datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat()
        trade_id = f"{side}_{symbol.replace('/', '')}_{timestamp_ms}"
        order_id = f"order_{trade_id}"

        # Net amount represents cash impact as positive number
        gross = price * quantity
        if side.lower() == 'buy':
            net_amount = gross + fee
        else:
            net_amount = gross - fee

        # Insert order record
        insert_order_sql = """
        INSERT INTO orders 
        (order_id, symbol, side, order_type, quantity, price, status, filled_quantity, 
         filled_price, commission, strategy, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute_insert(
            insert_order_sql,
            (
                order_id, symbol, side.lower(), 'market', quantity, price, 'filled',
                quantity, price, fee, strategy, ts_iso, ts_iso
            ),
        )

        # Insert trade record
        insert_trade_sql = """
        INSERT INTO trades 
        (trade_id, order_id, symbol, side, quantity, price, commission, net_amount, timestamp, strategy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        return self.execute_insert(
            insert_trade_sql,
            (
                trade_id, order_id, symbol, side.lower(), quantity, price, fee, net_amount, ts_iso, strategy
            ),
        )

    def insert_portfolio_metric(self, timestamp_ms: int, symbol: str, total_value: float,
                                cash_balance: float, position_qty: float,
                                position_avg_price: float) -> int:
        """
        Insert a portfolio snapshot into portfolio_metrics.

        The schema requires several performance fields; when not available,
        default conservative values (zeros/NULLs) are stored.
        """
        ts_iso = datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat()
        invested_value = position_qty * position_avg_price

        insert_sql = """
        INSERT INTO portfolio_metrics (
            timestamp, total_value, cash_balance, invested_value, total_pnl, daily_pnl,
            total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Defaults for performance metrics when not computed elsewhere
        return self.execute_insert(
            insert_sql,
            (
                ts_iso,
                float(total_value),
                float(cash_balance),
                float(invested_value),
                0.0,  # total_pnl
                0.0,  # daily_pnl
                0.0,  # total_return
                None,  # sharpe_ratio
                None,  # max_drawdown
                None,  # win_rate
                None,  # profit_factor
            ),
        )


# Global database instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager
