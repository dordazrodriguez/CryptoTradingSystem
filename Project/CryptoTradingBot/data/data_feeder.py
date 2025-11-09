"""
Data feeder module that provides working implementations from the completed project.
Supports CCXT for exchange connectivity and Alpaca for crypto trading.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import ccxt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedConfig:
    """Configuration for data feed."""
    exchange: str
    symbol: str
    timeframe: str = "1m"
    limit: int = 200


class DataFeed:
    """Data feed using CCXT library for cryptocurrency exchange connectivity."""
    
    def __init__(self, cfg: FeedConfig) -> None:
        """
        Initialize data feed.
        
        Args:
            cfg: Feed configuration
        """
        self.cfg = cfg
        ex_cls = getattr(ccxt, cfg.exchange)
        # Use rateLimit and enable built-in throttling where possible
        self.exchange = ex_cls({"enableRateLimit": True})
        logger.info(f"DataFeed initialized for {cfg.exchange}")

    def fetch_ohlcv(self, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            since: Timestamp in milliseconds (optional)
            limit: Number of candles to fetch (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        lim = limit or self.cfg.limit
        try:
            raw = self.exchange.fetch_ohlcv(
                self.cfg.symbol, 
                timeframe=self.cfg.timeframe, 
                since=since, 
                limit=lim
            )
            # Columns: [ timestamp, open, high, low, close, volume ]
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            logger.info(f"Fetched {len(df)} candles for {self.cfg.symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca data feed."""
    symbol: str
    api_key: str
    api_secret: str
    timeframe: str = "1m"
    limit: int = 200


class AlpacaFeed:
    """Data feed using Alpaca API for crypto data."""
    
    def __init__(self, cfg: AlpacaConfig) -> None:
        """
        Initialize Alpaca data feed.
        
        Args:
            cfg: Alpaca configuration
        """
        self.cfg = cfg
        # Use CCXT's Alpaca integration
        self.exchange = ccxt.alpaca({
            'apiKey': cfg.api_key,
            'secret': cfg.api_secret,
            'sandbox': True,  # Paper trading
            'enableRateLimit': True,
        })
        logger.info(f"AlpacaFeed initialized for {cfg.symbol}")

    def fetch_ohlcv(self, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Alpaca.
        
        Args:
            since: Timestamp in milliseconds (optional)
            limit: Number of candles to fetch (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        lim = limit or self.cfg.limit
        try:
            raw = self.exchange.fetch_ohlcv(
                self.cfg.symbol,
                timeframe=self.cfg.timeframe,
                since=since,
                limit=lim
            )
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            logger.info(f"Fetched {len(df)} candles from Alpaca for {self.cfg.symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV from Alpaca: {e}")
            return pd.DataFrame()


# Helper functions for simple indicator calculations
def sma(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame
        period: Moving average period
        price_col: Column name for price data
        
    Returns:
        Series with SMA values
    """
    return df[price_col].rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
    """
    Calculate Relative Strength Index (Wilder's method).
    
    Args:
        df: DataFrame
        period: RSI period (default: 14)
        price_col: Column name for price data
        
    Returns:
        Series with RSI values
    """
    # Wilder's RSI
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename(f"rsi_{period}")


def normalize_for_provider(provider: str, symbol: str, *, use_for: str = "data") -> str:
    """
    Normalize symbol for provider and use-case.
    - provider: "ccxt" or "alpaca"
    - use_for: "data" or "trading"
    Examples:
      - Alpaca data:  BTC/USD
      - Alpaca trading: BTCUSD
      - CCXT uses common exchange formats (e.g., BTC/USDT).
    
    Args:
        provider: Data provider
        symbol: Trading symbol
        use_for: Use case ("data" or "trading")
        
    Returns:
        Normalized symbol
    """
    p = provider.lower()
    if p == "alpaca":
        # Accept common forms and normalize
        s = symbol.replace("-", "/").upper()
        if use_for == "data":
            # Alpaca data expects slash, e.g., BTC/USD
            if "/" not in s and len(s) >= 6:
                # BTCUSD -> BTC/USD
                s = s[:3] + "/" + s[3:]
            return s
        else:
            # trading expects no slash, e.g., BTCUSD
            return s.replace("/", "")
    # Default: return as-is for CCXT and others
    return symbol


# Example usage
if __name__ == "__main__":
    # Test basic data feed
    config = FeedConfig(exchange="binance", symbol="BTC/USDT", timeframe="1m", limit=100)
    feed = DataFeed(config)
    df = feed.fetch_ohlcv()
    
    if not df.empty:
        print(f"Successfully fetched {len(df)} candles")
        print(df.head())
        
        # Add indicators
        df["sma_20"] = sma(df, 20)
        df["rsi_14"] = rsi(df, 14)
        
        print("\nWith indicators:")
        print(df.tail())
    else:
        print("Failed to fetch data")
