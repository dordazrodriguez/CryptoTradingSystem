"""
Technical indicators implementation for cryptocurrency trading.
Implements RSI, MACD, Bollinger Bands, SMA/EMA as descriptive methods.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Implements various technical indicators for trading analysis."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.db_manager = get_db_manager()
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series
            window: Moving average window
            
        Returns:
            SMA series
        """
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            window: EMA window
            
        Returns:
            EMA series
        """
        return prices.ewm(span=window).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            window: RSI window (default: 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line EMA window
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, lower bands and width
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K window
            d_window: %D window
            
        Returns:
            Dictionary with %K and %D
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Calculation window
            
        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: ATR window
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: ADX window
            
        Returns:
            Dictionary with ADX, +DI, -DI
        """
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Calculate smoothed values
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: CCI window
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        result_df = df.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            result_df[f'sma_{window}'] = self.calculate_sma(df['close'], window)
            result_df[f'ema_{window}'] = self.calculate_ema(df['close'], window)
        
        # RSI
        result_df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = bb_data['width']
        
        # Stochastic
        stoch_data = self.calculate_stochastic(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = stoch_data['k_percent']
        result_df['stoch_d'] = stoch_data['d_percent']
        
        # Williams %R
        result_df['williams_r'] = self.calculate_williams_r(df['high'], df['low'], df['close'])
        
        # ATR
        result_df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # ADX
        adx_data = self.calculate_adx(df['high'], df['low'], df['close'])
        result_df['adx'] = adx_data['adx']
        result_df['plus_di'] = adx_data['plus_di']
        result_df['minus_di'] = adx_data['minus_di']
        
        # CCI
        result_df['cci'] = self.calculate_cci(df['high'], df['low'], df['close'])
        
        logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
        
        return result_df
    
    def get_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with trading signals
        """
        signals_df = df.copy()
        
        # RSI signals
        signals_df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        signals_df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD signals
        signals_df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                                     (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        signals_df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                                     (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Bollinger Bands signals
        signals_df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5).astype(int)
        signals_df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(int)
        signals_df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Moving Average signals
        signals_df['sma_crossover_bull'] = ((df['close'] > df['sma_20']) & 
                                         (df['close'].shift(1) <= df['sma_20'].shift(1))).astype(int)
        signals_df['sma_crossover_bear'] = ((df['close'] < df['sma_20']) & 
                                          (df['close'].shift(1) >= df['sma_20'].shift(1))).astype(int)
        
        # Stochastic signals
        signals_df['stoch_oversold'] = ((df['stoch_k'] < 20) & (df['stoch_d'] < 20)).astype(int)
        signals_df['stoch_overbought'] = ((df['stoch_k'] > 80) & (df['stoch_d'] > 80)).astype(int)
        
        # Williams %R signals
        signals_df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
        signals_df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
        
        # Combined signals
        signals_df['bullish_signals'] = (signals_df['rsi_oversold'] + 
                                       signals_df['macd_bullish'] + 
                                       signals_df['sma_crossover_bull'] + 
                                       signals_df['stoch_oversold'] + 
                                       signals_df['williams_oversold'])
        
        signals_df['bearish_signals'] = (signals_df['rsi_overbought'] + 
                                       signals_df['macd_bearish'] + 
                                       signals_df['sma_crossover_bear'] + 
                                       signals_df['stoch_overbought'] + 
                                       signals_df['williams_overbought'])
        
        # Overall signal strength
        signals_df['signal_strength'] = signals_df['bullish_signals'] - signals_df['bearish_signals']
        
        logger.info("Generated trading signals based on technical indicators")
        
        return signals_df
    
    def store_indicators(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Store calculated indicators in database.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with indicators
            
        Returns:
            Number of records stored
        """
        stored_count = 0
        
        for _, row in df.iterrows():
            try:
                indicators = {
                    'rsi': row.get('rsi'),
                    'macd': row.get('macd'),
                    'macd_signal': row.get('macd_signal'),
                    'macd_histogram': row.get('macd_histogram'),
                    'sma_20': row.get('sma_20'),
                    'sma_50': row.get('sma_50'),
                    'sma_200': row.get('sma_200'),
                    'ema_12': row.get('ema_12'),
                    'ema_26': row.get('ema_26'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_middle': row.get('bb_middle'),
                    'bb_lower': row.get('bb_lower'),
                    'bb_width': row.get('bb_width')
                }
                
                self.db_manager.insert_indicators(
                    symbol=symbol,
                    timestamp=row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                    indicators=indicators
                )
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store indicators: {e}")
        
        return stored_count
    
    def get_indicator_summary(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get summary of latest indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of recent records
            
        Returns:
            Dictionary with indicator summary
        """
        query = """
        SELECT * FROM indicators 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        results = self.db_manager.execute_query(query, (symbol, limit))
        
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'symbol': symbol,
            'latest_timestamp': df['timestamp'].iloc[0],
            'total_records': len(df),
            'indicators': {}
        }
        
        # Get latest values for each indicator
        indicator_columns = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
        
        for col in indicator_columns:
            if col in df.columns:
                latest_value = df[col].iloc[0]
                if pd.notna(latest_value):
                    summary['indicators'][col] = {
                        'latest': float(latest_value),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
        
        return summary
