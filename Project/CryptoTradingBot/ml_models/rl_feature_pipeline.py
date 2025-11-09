"""Feature pipeline for RL that combines technical indicators, ML predictions, and portfolio status."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from trading_engine.indicators import TechnicalIndicators


class RLFeaturePipeline:
    """Combines technical indicators, ML predictions, and portfolio status into state vectors for PPO."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            config: Optional configuration dict with feature settings.
                   Expected structure: {'features': {'lookback_window': 100, 'normalization_method': 'z-score'}}
        """
        if config is None:
            config = {}
        
        features_config = config.get("features", {})
        self.lookback_window = features_config.get("lookback_window", 100)
        self.normalization_method = features_config.get("normalization_method", "z-score")
        
        # Initialize components - use existing TechnicalIndicators from CryptoTradingBot
        self.technical_indicators = TechnicalIndicators()
        
        # ML predictor will be injected by caller (optional)
        self.ml_predictor = None
        
        # State space dimensions (will be set after first feature computation)
        self.state_dim = None
        self.feature_names = []
    
    def set_ml_predictor(self, ml_predictor):
        """Set the ML predictor instance."""
        self.ml_predictor = ml_predictor
    
    def compute_features(
        self,
        df: pd.DataFrame,
        portfolio_value: float = 0.0,
        cash: float = 0.0,
        position_size: float = 0.0,
        last_action: Optional[int] = None,
        ml_prediction: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute complete feature vector for PPO agent.
        
        Args:
            df: DataFrame with OHLCV data (should have enough history for indicators)
            portfolio_value: Current portfolio value
            cash: Current cash balance
            position_size: Current position size (positive for long, negative for short)
            last_action: Last action taken (for action history feature)
            ml_prediction: Optional ML prediction. If None and ml_predictor is set, will compute.
            
        Returns:
            Numpy array with feature vector (float64 dtype)
        """
        # Calculate technical indicators using existing method
        df_with_indicators = self.technical_indicators.calculate_all_indicators(df)
        
        # Get ML prediction if not provided
        if ml_prediction is None and self.ml_predictor is not None:
            try:
                if hasattr(self.ml_predictor, 'model') and self.ml_predictor.model is not None:
                    # Try to predict using the ML predictor
                    if hasattr(self.ml_predictor, 'predict'):
                        pred_result = self.ml_predictor.predict(df_with_indicators)
                        if isinstance(pred_result, dict):
                            ml_prediction = pred_result.get('prediction', 0.0)
                        elif isinstance(pred_result, np.ndarray):
                            ml_prediction = float(pred_result[0] if len(pred_result) > 0 else 0.0)
                        else:
                            ml_prediction = float(pred_result)
                    else:
                        ml_prediction = 0.0
                else:
                    ml_prediction = 0.0
            except Exception as e:
                # If model not trained or fails, use neutral prediction
                if not hasattr(self, '_prediction_failure_logged'):
                    self._prediction_failure_logged = True
                    print(f"⚠ ML prediction failed (will use neutral 0.0): {e}")
                ml_prediction = 0.0
        elif ml_prediction is None:
            ml_prediction = 0.0
        
        # Get latest row for most recent features
        latest_row = df_with_indicators.iloc[-1]
        
        # Extract technical indicator features
        # Get indicator columns (excluding OHLCV base columns)
        base_cols = ['timestamp', 'ts', 'open', 'high', 'low', 'close', 'volume']
        indicator_cols = [col for col in df_with_indicators.columns if col not in base_cols]
        
        tech_features = latest_row[indicator_cols].values.astype(float)
        
        # Handle NaN values
        tech_features = np.nan_to_num(tech_features, nan=0.0, posinf=0.0, neginf=0.0)
        tech_features = tech_features.astype(np.float64)
        
        # Price features
        price_features = np.array([
            float(latest_row['close']),
            float(latest_row['open']),
            float(latest_row['high']),
            float(latest_row['low']),
            float(latest_row['volume']),
            float(latest_row.get('price_change_pct', 0.0) or 0.0) if 'price_change_pct' in df_with_indicators.columns else 0.0,
            float(latest_row.get('volume_change_pct', 0.0) or 0.0) if 'volume_change_pct' in df_with_indicators.columns else 0.0,
        ], dtype=np.float64)
        
        # Normalize price features (except percentages)
        price_features_normalized = np.array([
            self._normalize_value(price_features[0], df_with_indicators['close']),
            self._normalize_value(price_features[1], df_with_indicators['open']),
            self._normalize_value(price_features[2], df_with_indicators['high']),
            self._normalize_value(price_features[3], df_with_indicators['low']),
            self._normalize_value(price_features[4], df_with_indicators['volume']),
            price_features[5],  # price_change_pct
            price_features[6],  # volume_change_pct
        ], dtype=np.float64)
        
        # Portfolio features (normalized)
        portfolio_features = np.array([
            float(portfolio_value / (portfolio_value + 1e-10)),
            float(cash / (portfolio_value + 1e-10) if portfolio_value > 0 else 0.0),
            float(position_size / (abs(position_size) + 1e-10) if position_size != 0 else 0.0),
            float(abs(position_size) / (portfolio_value + 1e-10) if portfolio_value > 0 else 0.0),
        ], dtype=np.float64)
        
        # Action history (one-hot)
        if last_action is None:
            action_features = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            # One-hot encode: [Buy, Hold, Sell]
            action_features = np.array([
                1.0 if last_action == 0 else 0.0,
                1.0 if last_action == 1 else 0.0,
                1.0 if last_action == 2 else 0.0,
            ], dtype=np.float64)
        
        # ML prediction feature
        ml_feature = np.array([float(ml_prediction)], dtype=np.float64)
        
        # Normalize technical indicators if needed
        if self.normalization_method == "z-score" and len(tech_features) > 0:
            mean = tech_features.mean()
            std = tech_features.std()
            if std > 1e-10:
                tech_features = (tech_features - mean) / std
            tech_features = tech_features.astype(np.float64)
        
        # Combine all feature groups
        state_vector = np.concatenate([
            tech_features,
            price_features_normalized,
            portfolio_features,
            action_features,
            ml_feature,
        ]).astype(np.float64)
        
        # Handle any NaN or inf values
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Store state dimension
        if self.state_dim is None:
            self.state_dim = len(state_vector)
            self._update_feature_names(indicator_cols)
        
        # Ensure consistent state dimension
        if len(state_vector) != self.state_dim:
            if len(state_vector) < self.state_dim:
                state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)), mode='constant', constant_values=0.0)
            else:
                state_vector = state_vector[:self.state_dim]
        
        return state_vector.astype(np.float64)
    
    def _normalize_value(self, value: float, series: pd.Series) -> float:
        """Normalize a single value using series statistics."""
        if self.normalization_method == "z-score":
            mean = series.mean()
            std = series.std()
            if std > 1e-10:
                return (value - mean) / std
            return 0.0
        else:  # minmax
            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val
            if range_val > 1e-10:
                return 2 * ((value - min_val) / range_val) - 1  # Scale to [-1, 1]
            return 0.0
    
    def _update_feature_names(self, indicator_cols: List[str]):
        """Update feature names for debugging/logging."""
        self.feature_names = (
            indicator_cols +
            ['close', 'open', 'high', 'low', 'volume', 'price_change_pct', 'volume_change_pct'] +
            ['portfolio_value', 'cash', 'position_direction', 'position_size_ratio'] +
            ['action_buy', 'action_hold', 'action_sell'] +
            ['ml_prediction']
        )
    
    def get_state_dimension(self) -> int:
        """Get the state space dimension."""
        if self.state_dim is None:
            raise RuntimeError("State dimension not yet computed. Call compute_features() first.")
        return self.state_dim

