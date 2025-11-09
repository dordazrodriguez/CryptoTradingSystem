"""
Data processing and feature engineering module.
Handles data cleaning, validation, and feature creation for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and cleans trading data."""
    
    def __init__(self):
        """Initialize data processor."""
        self.db_manager = get_db_manager()
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by handling missing values and outliers.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]
        
        # Remove rows where high < low (invalid data)
        df = df[df['high'] >= df['low']]
        
        # Remove extreme outliers (prices that are 10x different from previous)
        for col in price_columns:
            df[f'{col}_pct_change'] = df[col].pct_change()
            df = df[abs(df[f'{col}_pct_change']) < 10]  # Remove >1000% changes
            df = df.drop(columns=[f'{col}_pct_change'])
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {'valid': False, 'issues': ['Empty dataset']}
        
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            issues.append(f"Missing values: {missing_counts.to_dict()}")
        
        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
        
        # Check for invalid price relationships
        invalid_ohlc = df[(df['high'] < df['low']) | 
                         (df['high'] < df['open']) | 
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close'])].shape[0]
        
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        # Check for zero or negative prices
        zero_prices = df[(df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)].shape[0]
        if zero_prices > 0:
            issues.append(f"Zero or negative prices: {zero_prices}")
        
        # Check for extreme price changes
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()  # >50% change
        if extreme_changes > 0:
            issues.append(f"Extreme price changes: {extreme_changes}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'missing_values': missing_counts.to_dict(),
            'duplicates': duplicates,
            'invalid_ohlc': invalid_ohlc,
            'zero_prices': zero_prices,
            'extreme_changes': extreme_changes
        }
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features for ML models.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with additional features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price position features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        
        # Momentum features
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)  # Rate of change
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target variable for ML (next period return)
        df['target_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
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
    
    def prepare_ml_dataset(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        Prepare dataset for ML training.
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back
            
        Returns:
            Prepared DataFrame for ML
        """
        # Get historical data
        df = pd.DataFrame(self.db_manager.get_price_history(symbol, limit=lookback_days * 24 * 60))
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Clean data
        df = self.clean_price_data(df)
        
        # Create features
        df = self.create_features(df)
        
        # Remove rows with NaN values (from rolling calculations)
        df = df.dropna()
        
        logger.info(f"Prepared ML dataset: {len(df)} records for {symbol}")
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze feature importance and correlations.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature analysis
        """
        if df.empty:
            return {}
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        # Calculate correlations with target
        if 'target_return' in numeric_df.columns:
            correlations = numeric_df.corr()['target_return'].abs().sort_values(ascending=False)
        else:
            correlations = None
        
        # Calculate feature statistics
        feature_stats = numeric_df.describe()
        
        # Check for highly correlated features
        corr_matrix = numeric_df.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return {
            'total_features': len(numeric_cols),
            'correlations_with_target': correlations.to_dict() if correlations is not None else {},
            'feature_statistics': feature_stats.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'missing_values': numeric_df.isnull().sum().to_dict()
        }


class FeatureEngineer:
    """Advanced feature engineering for ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.processor = DataProcessor()
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for ML models.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with advanced features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Market regime features
        df['trend_strength'] = self.calculate_trend_strength(df['close'])
        df['market_regime'] = self.classify_market_regime(df['close'])
        
        # Volatility clustering
        df['volatility_cluster'] = self.detect_volatility_clusters(df['close'])
        
        # Support and resistance levels
        df['support_level'] = self.calculate_support_level(df)
        df['resistance_level'] = self.calculate_resistance_level(df)
        
        # Price action patterns
        df['doji_pattern'] = self.detect_doji_pattern(df)
        df['hammer_pattern'] = self.detect_hammer_pattern(df)
        df['engulfing_pattern'] = self.detect_engulfing_pattern(df)
        
        # Cross-asset features (if multiple symbols available)
        df['relative_strength'] = self.calculate_relative_strength(df)
        
        return df
    
    def calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        from scipy import stats
        
        trend_strength = []
        for i in range(len(prices)):
            if i < window:
                trend_strength.append(0)
            else:
                y = prices.iloc[i-window:i].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                trend_strength.append(slope)
        
        return pd.Series(trend_strength, index=prices.index)
    
    def classify_market_regime(self, prices: pd.Series) -> pd.Series:
        """Classify market regime based on volatility and trend."""
        volatility = prices.pct_change().rolling(20).std()
        trend = prices.rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        
        # Simple classification
        regime = pd.Series(index=prices.index, dtype=str)
        regime[(volatility < volatility.quantile(0.33)) & (trend > 0)] = 'bull_low_vol'
        regime[(volatility < volatility.quantile(0.33)) & (trend < 0)] = 'bear_low_vol'
        regime[(volatility > volatility.quantile(0.67)) & (trend > 0)] = 'bull_high_vol'
        regime[(volatility > volatility.quantile(0.67)) & (trend < 0)] = 'bear_high_vol'
        regime[regime.isna()] = 'sideways'
        
        return regime
    
    def detect_volatility_clusters(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Detect volatility clustering periods."""
        returns = prices.pct_change()
        volatility = returns.rolling(window).std()
        
        # Identify high volatility periods
        high_vol_threshold = volatility.quantile(0.8)
        clusters = (volatility > high_vol_threshold).astype(int)
        
        return clusters
    
    def calculate_support_level(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate dynamic support level."""
        return df['low'].rolling(window).min()
    
    def calculate_resistance_level(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate dynamic resistance level."""
        return df['high'].rolling(window).max()
    
    def detect_doji_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect doji candlestick pattern."""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        
        # Doji: small body relative to total range
        doji = (body_size / total_range < 0.1).astype(int)
        return doji
    
    def detect_hammer_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern."""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Hammer: small upper shadow, long lower shadow
        hammer = ((upper_shadow < body_size * 0.1) & 
                 (lower_shadow > body_size * 2)).astype(int)
        return hammer
    
    def detect_engulfing_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect engulfing candlestick pattern."""
        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body = abs(df['close'] - df['open'])
        
        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulfing = ((df['close'] > df['open']) & 
                           (df['close'].shift(1) < df['open'].shift(1)) &
                           (df['open'] < df['close'].shift(1)) &
                           (df['close'] > df['open'].shift(1))).astype(int)
        
        return bullish_engulfing
    
    def calculate_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate relative strength (placeholder for multi-asset analysis)."""
        # This would require data from multiple symbols
        # For now, return a simple momentum measure
        return df['close'].pct_change(5)


# Example usage
if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTC/USD',
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50000 + np.random.randn(100) * 100 + 50,
        'low': 50000 + np.random.randn(100) * 100 - 50,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': np.random.randint(1000, 10000, 100),
        'timeframe': '1m'
    })
    
    # Process the data
    cleaned_data = processor.clean_price_data(sample_data)
    quality_report = processor.validate_data_quality(cleaned_data)
    features_data = processor.create_features(cleaned_data)
    
    print("Data processing test completed!")
    print(f"Quality report: {quality_report}")
    print(f"Features created: {len(features_data.columns)}")
