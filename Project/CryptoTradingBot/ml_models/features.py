"""
Feature engineering pipeline for machine learning models.
Creates price momentum, volatility, volume patterns, and other ML features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """Advanced feature engineering for machine learning models."""
    
    def __init__(self):
        """Initialize ML feature engineer."""
        self.scalers = {}
        self.feature_importance = {}
        self.selected_features = []
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features for ML.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        result_df = df.copy()
        
        # Basic price features
        result_df['price_change'] = df['close'].pct_change()
        result_df['price_change_abs'] = result_df['price_change'].abs()
        result_df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price momentum features
        for window in [1, 2, 3, 5, 10, 20]:
            result_df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            result_df[f'roc_{window}'] = df['close'].pct_change(window)  # Rate of change
            result_df[f'price_acceleration_{window}'] = result_df[f'momentum_{window}'].diff()
        
        # Price volatility features
        for window in [5, 10, 20, 50]:
            result_df[f'volatility_{window}'] = result_df['price_change'].rolling(window).std()
            result_df[f'volatility_ratio_{window}'] = result_df[f'volatility_{window}'] / result_df[f'volatility_{window}'].rolling(50).mean()
            result_df[f'price_range_{window}'] = (df['high'].rolling(window).max() - df['low'].rolling(window).min()) / df['close']
        
        # Price position features
        for window in [10, 20, 50]:
            result_df[f'price_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / (df['high'].rolling(window).max() - df['low'].rolling(window).min())
            result_df[f'price_percentile_{window}'] = df['close'].rolling(window).rank(pct=True)
        
        # Gap features
        result_df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        result_df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        result_df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Intraday features
        result_df['intraday_return'] = (df['close'] - df['open']) / df['open']
        result_df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
        result_df['body_size'] = abs(df['close'] - df['open']) / df['open']
        result_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        result_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} price features")
        return result_df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features for ML.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        result_df = df.copy()
        
        # Basic volume features
        result_df['volume_change'] = df['volume'].pct_change()
        result_df['volume_change_abs'] = result_df['volume_change'].abs()
        result_df['log_volume'] = np.log(df['volume'] + 1)  # Add 1 to avoid log(0)
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            result_df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            result_df[f'volume_ema_{window}'] = df['volume'].ewm(span=window).mean()
            result_df[f'volume_ratio_{window}'] = df['volume'] / result_df[f'volume_sma_{window}']
        
        # Volume volatility
        for window in [10, 20]:
            result_df[f'volume_volatility_{window}'] = result_df['volume_change'].rolling(window).std()
        
        # Volume-price relationship
        result_df['volume_price_trend'] = result_df['price_change'] * result_df['volume_change']
        result_df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        result_df['vwap_deviation'] = (df['close'] - result_df['volume_weighted_price']) / result_df['volume_weighted_price']
        
        # Volume patterns
        result_df['volume_spike'] = (result_df['volume_ratio_20'] > 2.0).astype(int)
        result_df['volume_dry_up'] = (result_df['volume_ratio_20'] < 0.5).astype(int)
        
        # On-balance volume (simplified)
        result_df['obv'] = (result_df['price_change'] > 0).astype(int) * df['volume'] - (result_df['price_change'] < 0).astype(int) * df['volume']
        result_df['obv'] = result_df['obv'].cumsum()
        result_df['obv_change'] = result_df['obv'].pct_change()
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} volume features")
        return result_df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features for ML.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        result_df = df.copy()
        
        # Moving averages
        for window in [5, 10, 12, 20, 26, 50, 200]:
            result_df[f'sma_{window}'] = df['close'].rolling(window).mean()
            result_df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            result_df[f'price_sma_ratio_{window}'] = df['close'] / result_df[f'sma_{window}']
            result_df[f'price_ema_ratio_{window}'] = df['close'] / result_df[f'ema_{window}']
        
        # Moving average crossovers
        result_df['sma_cross_5_20'] = (result_df['sma_5'] > result_df['sma_20']).astype(int)
        result_df['sma_cross_10_50'] = (result_df['sma_10'] > result_df['sma_50']).astype(int)
        result_df['ema_cross_12_26'] = (result_df['ema_12'] > result_df['ema_26']).astype(int)
        
        # RSI
        result_df['rsi'] = self._calculate_rsi(df['close'])
        result_df['rsi_oversold'] = (result_df['rsi'] < 30).astype(int)
        result_df['rsi_overbought'] = (result_df['rsi'] > 70).astype(int)
        result_df['rsi_divergence'] = self._detect_rsi_divergence(df['close'], result_df['rsi'])
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        result_df['macd_cross'] = (result_df['macd'] > result_df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = bb_data['width']
        result_df['bb_position'] = (df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        result_df['bb_squeeze'] = (result_df['bb_width'] < result_df['bb_width'].rolling(20).mean() * 0.5).astype(int)
        
        # Stochastic
        stoch_data = self._calculate_stochastic(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = stoch_data['k_percent']
        result_df['stoch_d'] = stoch_data['d_percent']
        result_df['stoch_oversold'] = ((result_df['stoch_k'] < 20) & (result_df['stoch_d'] < 20)).astype(int)
        result_df['stoch_overbought'] = ((result_df['stoch_k'] > 80) & (result_df['stoch_d'] > 80)).astype(int)
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} technical features")
        return result_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for ML.
        
        Args:
            df: DataFrame with timestamp column (may be 'ts' or 'timestamp')
            
        Returns:
            DataFrame with time features
        """
        result_df = df.copy()
        
        # Normalize timestamp column name (might be 'ts' or 'timestamp')
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'ts'
        if timestamp_col == 'ts' and 'timestamp' not in result_df.columns:
            result_df['timestamp'] = result_df[timestamp_col]
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in result_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        else:
            # If no timestamp column exists, create one from index
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df['timestamp'] = result_df.index
            else:
                logger.warning("No timestamp column found, skipping time features")
                return result_df
        
        # Extract time components
        result_df['hour'] = result_df['timestamp'].dt.hour
        result_df['minute'] = result_df['timestamp'].dt.minute
        result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
        result_df['day_of_month'] = result_df['timestamp'].dt.day
        result_df['month'] = result_df['timestamp'].dt.month
        result_df['quarter'] = result_df['timestamp'].dt.quarter
        
        # Cyclical encoding
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df['minute'] / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df['minute'] / 60)
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # Market session features
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
        result_df['is_market_open'] = ((result_df['hour'] >= 9) & (result_df['hour'] <= 16)).astype(int)
        result_df['is_lunch_time'] = ((result_df['hour'] >= 12) & (result_df['hour'] <= 13)).astype(int)
        result_df['is_end_of_day'] = (result_df['hour'] >= 15).astype(int)
        
        # Time since market open
        result_df['minutes_since_open'] = (result_df['hour'] - 9) * 60 + result_df['timestamp'].dt.minute
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} time features")
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, target_column: str = 'close', 
                           lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Create lagged features for ML.
        
        Args:
            df: DataFrame with data
            target_column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result_df = df.copy()
        
        for lag in lags:
            result_df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            result_df[f'{target_column}_pct_change_lag_{lag}'] = df[target_column].pct_change(lag)
        
        logger.info(f"Created {len(lags) * 2} lag features")
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different indicators.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with interaction features
        """
        result_df = df.copy()
        
        # Price-volume interactions
        if 'price_change' in df.columns and 'volume_change' in df.columns:
            result_df['price_volume_interaction'] = df['price_change'] * df['volume_change']
            result_df['price_volume_abs_interaction'] = df['price_change'].abs() * df['volume_change'].abs()
        
        # RSI-MACD interactions
        if 'rsi' in df.columns and 'macd' in df.columns:
            result_df['rsi_macd_interaction'] = df['rsi'] * df['macd']
            result_df['rsi_macd_divergence'] = ((df['rsi'] > 50) & (df['macd'] < 0)).astype(int)
        
        # Bollinger Band interactions
        if 'bb_position' in df.columns and 'bb_width' in df.columns:
            result_df['bb_position_width_interaction'] = df['bb_position'] * df['bb_width']
        
        # Volatility-momentum interactions
        if 'volatility_20' in df.columns and 'momentum_5' in df.columns:
            result_df['volatility_momentum_interaction'] = df['volatility_20'] * df['momentum_5'].abs()
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} interaction features")
        return result_df
    
    def create_alternative_data_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create alternative data features (on-chain, sentiment, market-wide).
        Note: These are placeholder implementations. In production, you'd fetch real data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with alternative data features
        """
        result_df = df.copy()
        
        # Placeholder for on-chain data features (BTC-specific)
        # In production, fetch from APIs like Glassnode, Blockchain.com, etc.
        # For now, we'll create proxy features based on price/volume patterns
        
        # Proxy for network activity (high volume = high network activity)
        result_df['network_activity_proxy'] = df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean()  # 24h vs 7d
        result_df['network_activity_ma_cross'] = (result_df['network_activity_proxy'] > 1.2).astype(int)
        
        # Proxy for whale movements (large price moves with high volume)
        price_change = df['close'].pct_change()
        result_df['whale_movement_proxy'] = (price_change.abs() * df['volume']) / df['volume'].rolling(24).mean()
        result_df['whale_buy_pressure'] = ((price_change > 0) & (result_df['whale_movement_proxy'] > 2.0)).astype(int)
        result_df['whale_sell_pressure'] = ((price_change < 0) & (result_df['whale_movement_proxy'] > 2.0)).astype(int)
        
        # Proxy for exchange inflow/outflow (price drops on high volume = outflow)
        result_df['exchange_flow_proxy'] = (price_change * df['volume']) / df['volume'].rolling(24).mean()
        result_df['exchange_outflow'] = (result_df['exchange_flow_proxy'] < -1.5).astype(int)
        result_df['exchange_inflow'] = (result_df['exchange_flow_proxy'] > 1.5).astype(int)
        
        # Placeholder for sentiment features
        # In production, fetch from Twitter/Reddit APIs or sentiment analysis services
        # Proxy: volatility spikes often correlate with sentiment extremes
        volatility = df['close'].pct_change().rolling(24).std()
        result_df['sentiment_volatility_proxy'] = volatility / volatility.rolling(168).mean()
        result_df['high_sentiment_volatility'] = (result_df['sentiment_volatility_proxy'] > 1.5).astype(int)
        
        # Placeholder for Crypto Fear & Greed Index
        # In production, fetch from alternative.me API
        # Proxy: use market conditions to estimate fear/greed
        price_change_7d = df['close'].pct_change(168)  # 7 days if 1min data
        volatility_7d = df['close'].pct_change().rolling(168).std()
        result_df['fear_greed_proxy'] = np.tanh(price_change_7d / (volatility_7d + 0.001))  # Normalized to -1 to 1
        result_df['extreme_fear'] = (result_df['fear_greed_proxy'] < -0.7).astype(int)
        result_df['extreme_greed'] = (result_df['fear_greed_proxy'] > 0.7).astype(int)
        
        # Placeholder for BTC dominance
        # In production, fetch from CoinGecko or similar
        # Proxy: use relative strength vs volume patterns
        result_df['btc_dominance_proxy'] = df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean()
        
        # Placeholder for funding rates (perpetual futures)
        # In production, fetch from exchange APIs (Binance, Bybit, etc.)
        # Proxy: use price momentum as indicator of funding rate sentiment
        result_df['funding_rate_proxy'] = np.tanh(df['close'].pct_change(24))  # Positive = long funding, negative = short funding
        
        # Market-wide features
        result_df['market_regime'] = self._classify_market_regime(df)
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} alternative data features")
        return result_df
    
    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify market regime: 0=trending_up, 1=trending_down, 2=sideways, 3=volatile."""
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean() if len(df) >= 50 else sma_20
        price = df['close']
        volatility = df['close'].pct_change().rolling(20).std()
        
        # Classify regime
        regime = pd.Series(2, index=df.index)  # Default: sideways
        
        # Trending up
        regime[(price > sma_20) & (sma_20 > sma_50) & (volatility < volatility.rolling(50).mean() * 1.2)] = 0
        
        # Trending down
        regime[(price < sma_20) & (sma_20 < sma_50) & (volatility < volatility.rolling(50).mean() * 1.2)] = 1
        
        # Volatile
        regime[volatility > volatility.rolling(50).mean() * 1.5] = 3
        
        return regime.fillna(2)
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all ML features including alternative data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Apply all feature engineering steps
        result_df = self.create_price_features(df)
        result_df = self.create_volume_features(result_df)
        result_df = self.create_technical_features(result_df)
        result_df = self.create_time_features(result_df)
        result_df = self.create_lag_features(result_df)
        result_df = self.create_interaction_features(result_df)
        result_df = self.create_alternative_data_features(result_df)  # Add alternative data
        
        # Handle NaN values more intelligently
        original_length = len(result_df)
        
        # Fill NaN values with forward fill and backward fill for some features
        # This preserves more data while handling lookback windows
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Forward fill first (carry last known value forward)
            result_df[col] = result_df[col].ffill()
            # Then backward fill (fill remaining NaNs at start)
            result_df[col] = result_df[col].bfill()
            # Fill any remaining NaNs with 0 (shouldn't happen, but safety)
            result_df[col] = result_df[col].fillna(0)
        
        # Remove rows that still have NaN in critical columns (timestamp, close)
        critical_cols = ['timestamp', 'close'] if 'timestamp' in result_df.columns else ['close']
        result_df = result_df.dropna(subset=critical_cols)
        removed_rows = original_length - len(result_df)
        
        logger.info(f"Feature engineering complete: {len(result_df.columns)} features, "
                   f"{removed_rows} rows removed (only critical columns), {len(result_df)} rows available")
        
        return result_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """
        Select top k features using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        # Handle infinite values
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_columns)))
        selector.fit(X_numeric, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        self.selected_features = numeric_columns[selected_mask].tolist()
        
        # Store feature importance scores
        self.feature_importance = dict(zip(numeric_columns, selector.scores_))
        
        logger.info(f"Selected {len(self.selected_features)} features out of {len(numeric_columns)}")
        
        return self.selected_features
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale features for ML models.
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Scaled feature matrix
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info(f"Features scaled using {method} scaling")
        
        return X_scaled_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
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
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def _detect_rsi_divergence(self, prices: pd.Series, rsi: pd.Series, window: int = 20) -> pd.Series:
        """Detect RSI divergence (simplified)."""
        price_trend = prices.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        rsi_trend = rsi.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Bullish divergence: price down, RSI up
        bullish_div = ((price_trend < 0) & (rsi_trend > 0)).astype(int)
        
        # Bearish divergence: price up, RSI down
        bearish_div = ((price_trend > 0) & (rsi_trend < 0)).astype(int)
        
        return bullish_div - bearish_div


# Example usage and testing
if __name__ == "__main__":
    # Test the ML feature engineer
    feature_engineer = MLFeatureEngineer()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(1000) * 100,
        'high': 50000 + np.random.randn(1000) * 100 + 50,
        'low': 50000 + np.random.randn(1000) * 100 - 50,
        'close': 50000 + np.random.randn(1000) * 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create all features
    featured_data = feature_engineer.create_all_features(sample_data)
    
    print("ML Feature Engineering test completed!")
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"With features: {len(featured_data.columns)}")
    print(f"Features added: {len(featured_data.columns) - len(sample_data.columns)}")
    
    # Test feature selection
    if len(featured_data) > 0:
        # Create target variable
        y = featured_data['close'].shift(-1) - featured_data['close']  # Next period return
        X = featured_data.drop(columns=['timestamp', 'close'])
        
        # Remove NaN values
        valid_idx = ~(y.isna() | X.isna().any(axis=1))
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) > 0:
            selected_features = feature_engineer.select_features(X_clean, y_clean, k=20)
            print(f"Selected features: {len(selected_features)}")
            print(f"Top 5 features: {selected_features[:5]}")
