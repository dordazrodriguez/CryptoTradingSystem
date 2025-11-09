"""
Live data collector for cryptocurrency prices using CCXT library.
Connects to Alpaca for crypto trading data and paper trading.
"""

import ccxt
import time
import logging
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional, Any
import pandas as pd
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class AlpacaDataCollector:
    """Collects live cryptocurrency data from Alpaca using CCXT."""
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """
        Initialize Alpaca data collector.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Whether to use paper trading (default: True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        
        # Initialize CCXT Alpaca exchange
        self.exchange = ccxt.alpaca({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': paper_trading,  # Use sandbox for paper trading
            'enableRateLimit': True,
        })
        
        self.db_manager = get_db_manager()
        
        # Determine tracked symbols
        # 1) Read desired symbols from env ALPACA_SYMBOLS (comma-separated)
        env_symbols = os.getenv('ALPACA_SYMBOLS', '')
        desired_symbols = [s.strip() for s in env_symbols.split(',') if s.strip()] if env_symbols else []

        # 2) Load available markets from exchange
        available = set()
        try:
            markets = self.exchange.load_markets()
            available = set(markets.keys())
        except Exception as e:
            logger.warning(f"Could not load markets from Alpaca: {e}")

        # 3) Default desired list if none provided (single symbol to reduce load)
        if not desired_symbols:
            desired_symbols = ['BTC/USD']

        # 4) Filter desired symbols by availability (fall back to BTC/ETH if empty)
        filtered = [s for s in desired_symbols if (not available or s in available)]
        if not filtered:
            fallback = ['BTC/USD', 'ETH/USD']
            filtered = [s for s in fallback if (not available or s in available)] or fallback

        self.symbols = filtered

        logger.info(
            f"Alpaca data collector initialized (paper_trading={paper_trading}), symbols={self.symbols}"
        )
    
    def test_connection(self) -> bool:
        """Test connection to Alpaca API."""
        try:
            # Test with a simple API call
            markets = self.exchange.load_markets()
            logger.info(f"Successfully connected to Alpaca. Available markets: {len(markets)}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            account = self.exchange.fetch_balance()
            return account
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            
        Returns:
            Current price or None if failed
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '1m', 
                           limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe ('1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None
    
    def store_price_data(self, df: pd.DataFrame) -> int:
        """
        Store price data in database.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Number of records stored
        """
        stored_count = 0
        
        for _, row in df.iterrows():
            try:
                self.db_manager.insert_price_data(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'].isoformat(),
                    open_price=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timeframe=row['timeframe']
                )
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store price data: {e}")
        
        return stored_count
    
    def collect_and_store_data(self, symbols: List[str] = None, 
                              timeframe: str = '1m', limit: int = 100) -> Dict[str, int]:
        """
        Collect and store data for multiple symbols.
        
        Args:
            symbols: List of symbols to collect (default: self.symbols)
            timeframe: Timeframe for data collection
            limit: Number of candles per symbol
            
        Returns:
            Dictionary with symbol -> records_stored mapping
        """
        if symbols is None:
            symbols = self.symbols
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")
            
            # Get historical data
            df = self.get_historical_data(symbol, timeframe, limit)
            
            if df is not None:
                # Store in database
                stored_count = self.store_price_data(df)
                results[symbol] = stored_count
                logger.info(f"Stored {stored_count} records for {symbol}")
            else:
                results[symbol] = 0
                logger.warning(f"No data collected for {symbol}")
            
            # Rate limiting
            time.sleep(0.1)
        
        return results
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all tracked symbols."""
        prices = {}
        
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def start_live_collection(self, interval: int = 60):
        """
        Start live data collection loop.
        
        Args:
            interval: Collection interval in seconds
        """
        logger.info(f"Starting live data collection (interval: {interval}s)")
        
        while True:
            try:
                # Collect current prices
                prices = self.get_latest_prices()
                
                # Store latest prices as 1-minute candles
                current_time = datetime.now(timezone.utc)
                
                for symbol, price in prices.items():
                    self.db_manager.insert_price_data(
                        symbol=symbol,
                        timestamp=current_time.isoformat(),
                        open_price=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0,  # Volume not available for current price
                        timeframe='1m'
                    )
                
                logger.info(f"Collected prices: {prices}")
                
                # Wait for next collection
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Live data collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in live collection: {e}")
                time.sleep(interval)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        try:
            markets = self.exchange.load_markets()
            crypto_symbols = [symbol for symbol in markets.keys() 
                            if '/USD' in symbol and markets[symbol]['type'] == 'spot']
            return crypto_symbols
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []


class DataCollectorManager:
    """Manages data collection operations."""
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        """Initialize data collector manager."""
        self.collector = AlpacaDataCollector(api_key, secret_key, paper_trading)
        self.db_manager = get_db_manager()
    
    def initialize_data(self, days_back: int = 30):
        """
        Initialize database with historical data.
        
        Args:
            days_back: Number of days of historical data to fetch
        """
        logger.info(f"Initializing data for last {days_back} days")
        
        # Calculate limit based on timeframe (assuming 1-minute data)
        limit = days_back * 24 * 60  # 1 minute candles
        
        results = self.collector.collect_and_store_data(limit=min(limit, 1000))
        
        total_stored = sum(results.values())
        logger.info(f"Initialization complete. Total records stored: {total_stored}")
        
        return results
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        summary = self.db_manager.get_database_stats()
        
        # Get latest prices
        latest_prices = {}
        for symbol in self.collector.symbols:
            latest = self.db_manager.get_latest_price(symbol)
            if latest:
                latest_prices[symbol] = latest['close']
        
        summary['latest_prices'] = latest_prices
        return summary


class DataCollector:
    """
    Simple wrapper for data collection.
    Can use AlpacaDataCollector or provide a simpler interface.
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper_trading: bool = True):
        """
        Initialize data collector.
        
        Args:
            api_key: Alpaca API key (optional, can use env vars)
            secret_key: Alpaca secret key (optional, can use env vars)
            paper_trading: Use paper trading mode
        """
        import os
        
        # Get API keys from args or environment
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper_trading = paper_trading
        
        # Initialize Alpaca collector if API keys available
        if self.api_key and self.secret_key:
            try:
                self.collector = AlpacaDataCollector(self.api_key, self.secret_key, paper_trading)
                self.has_collector = True
            except Exception as e:
                logger.warning(f"Could not initialize Alpaca collector: {e}")
                self.has_collector = False
        else:
            self.has_collector = False
            self.collector = None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[datetime] = None, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            since: Start datetime
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if self.has_collector and self.collector:
            try:
                # Use Alpaca collector
                ohlcv = self.collector.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=int(since.timestamp() * 1000) if since else None,
                    limit=limit
                )
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV from Alpaca: {e}")
        
        # Fallback: return None (caller can generate synthetic data)
        return None


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing
    import os
    
    # Load API keys from environment variables
    api_key = os.getenv('ALPACA_API_KEY', 'your_api_key_here')
    secret_key = os.getenv('ALPACA_SECRET_KEY', 'your_secret_key_here')
    
    if api_key == 'your_api_key_here':
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    else:
        # Test the collector
        collector = AlpacaDataCollector(api_key, secret_key, paper_trading=True)
        
        if collector.test_connection():
            print("Connection successful!")
            
            # Get available symbols
            symbols = collector.get_available_symbols()
            print(f"Available symbols: {symbols[:10]}...")  # Show first 10
            
            # Get latest prices
            prices = collector.get_latest_prices()
            print(f"Latest prices: {prices}")
        else:
            print("Connection failed!")
