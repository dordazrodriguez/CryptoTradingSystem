"""
Model retraining service for continuous model improvement.
Monitors model performance and retrains when necessary.
"""

import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

from ml_models.predictor import CryptoPredictionModel
from ml_models.evaluation import ModelEvaluator
from data.db import get_db_manager
from data.data_feeder import DataFeed, FeedConfig, AlpacaFeed, AlpacaConfig, normalize_for_provider

logger = logging.getLogger(__name__)

# Optional Prefect integration
try:
    from ml_models.prefect_flows import model_retraining_flow
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    logger.warning("Prefect not available. Using standard retraining workflow.")


class ModelRetrainingService:
    """
    Service for retraining ML models based on performance monitoring.
    Supports scheduled retraining and performance-based retraining triggers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model retraining service.
        
        Args:
            config: Configuration dictionary with:
                - model_dir: Directory for storing models
                - min_accuracy_threshold: Minimum accuracy to trigger retraining (default: 0.55)
                - performance_window_days: Days of performance data to analyze
                - retrain_interval_days: Days between scheduled retrains (default: 7)
                - min_training_samples: Minimum samples required for training (default: 1000)
                - symbol: Trading symbol
                - exchange: Exchange name
                - provider: Data provider
                - alpaca_api_key: Alpaca API key (if using Alpaca)
                - alpaca_secret_key: Alpaca secret key (if using Alpaca)
        """
        self.config = config
        self.model_dir = Path(config.get('model_dir', 'ml_models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_accuracy_threshold = config.get('min_accuracy_threshold', 0.55)
        self.performance_window_days = config.get('performance_window_days', 7)
        self.retrain_interval_days = config.get('retrain_interval_days', 7)
        self.min_training_samples = config.get('min_training_samples', 1000)
        self.use_prefect = config.get('use_prefect', False) and PREFECT_AVAILABLE
        
        self.db_manager = get_db_manager()
        self.evaluator = ModelEvaluator()
        
        # Initialize data feed for training data collection
        self.symbol = config.get('symbol', 'BTC/USD')
        self.provider = config.get('provider', 'alpaca')
        self.exchange = config.get('exchange', 'binance')
        
        feed_symbol = normalize_for_provider(self.provider, self.symbol, use_for="data")
        if self.provider == "alpaca":
            api_key = config.get('alpaca_api_key')
            secret_key = config.get('alpaca_secret_key')
            if api_key and secret_key:
                self.data_feed = AlpacaFeed(AlpacaConfig(
                    symbol=feed_symbol,
                    timeframe="1m",
                    limit=5000,  # More data for training
                    api_key=api_key,
                    api_secret=secret_key
                ))
            else:
                self.data_feed = None
        else:
            feed_config = FeedConfig(
                exchange=self.exchange,
                symbol=feed_symbol,
                timeframe="1m",
                limit=5000
            )
            self.data_feed = DataFeed(feed_config)
        
        # Model versioning
        self.version_file = self.model_dir / 'model_versions.json'
        self.load_model_versions()
        
        logger.info("Model retraining service initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Min accuracy threshold: {self.min_accuracy_threshold}")
    
    def load_model_versions(self):
        """Load model version history."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    self.model_versions = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model versions: {e}")
                self.model_versions = {}
        else:
            self.model_versions = {}
    
    def save_model_versions(self):
        """Save model version history."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.model_versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")
    
    def get_current_model_version(self) -> Optional[str]:
        """Get current active model version."""
        if not self.model_versions:
            return None
        
        # Find active version
        for version, info in self.model_versions.items():
            if info.get('status') == 'active':
                return version
        
        # Return most recent if no active
        if self.model_versions:
            return max(self.model_versions.keys(), key=lambda x: self.model_versions[x].get('created_at', ''))
        
        return None
    
    def check_model_performance(self) -> Dict[str, Any]:
        """
        Check current model performance from recent predictions.
        
        Returns:
            Performance analysis dictionary
        """
        try:
            # Get recent predictions and outcomes from database
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.performance_window_days)
            
            # Query recent predictions
            # Note: This assumes predictions are stored with actual outcomes
            # In a real system, you'd track predictions vs actuals
            
            # For now, we'll estimate based on recent price movements
            # In production, you'd have a prediction tracking system
            
            logger.info("Checking model performance...")
            
            # Get recent market data to simulate performance check
            if self.data_feed:
                df = self.data_feed.fetch_ohlcv()
                if len(df) < 100:
                    return {'error': 'Insufficient data for performance check'}
                
                # Simulate performance analysis
                # In production, compare stored predictions with actual outcomes
                performance = {
                    'samples_analyzed': len(df),
                    'window_days': self.performance_window_days,
                    'needs_retraining': False,  # Will be determined by actual predictions
                    'estimated_accuracy': None  # Would come from prediction tracking
                }
                
                return performance
            else:
                return {'error': 'No data feed available'}
                
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return {'error': str(e)}
    
    def should_retrain(self, force: bool = False) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            force: Force retraining regardless of conditions
            
        Returns:
            True if model should be retrained
        """
        if force:
            return True
        
        # Check if scheduled retraining is due
        current_version = self.get_current_model_version()
        if current_version and current_version in self.model_versions:
            last_trained = self.model_versions[current_version].get('created_at')
            if last_trained:
                try:
                    last_trained_date = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
                    days_since = (datetime.now(timezone.utc) - last_trained_date).days
                    if days_since >= self.retrain_interval_days:
                        logger.info(f"Scheduled retraining due ({days_since} days since last training)")
                        return True
                except Exception as e:
                    logger.warning(f"Error parsing last training date: {e}")
        
        # Check performance-based trigger
        performance = self.check_model_performance()
        if 'error' not in performance:
            if performance.get('needs_retraining'):
                logger.info("Performance-based retraining triggered")
                return True
            
            estimated_accuracy = performance.get('estimated_accuracy')
            if estimated_accuracy and estimated_accuracy < self.min_accuracy_threshold:
                logger.info(f"Accuracy below threshold ({estimated_accuracy:.3f} < {self.min_accuracy_threshold})")
                return True
        
        return False
    
    def collect_training_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Collect training data from exchange.
        
        Args:
            days: Number of days of historical data to collect
            
        Returns:
            DataFrame with OHLCV data or None if collection fails
        """
        try:
            logger.info(f"Collecting training data for last {days} days...")
            
            if not self.data_feed:
                logger.error("No data feed available")
                return None
            
            # Fetch recent data
            df = self.data_feed.fetch_ohlcv()
            
            if df.empty:
                logger.error("No data collected")
                return None
            
            # Filter to requested time period
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            if 'ts' in df.columns:
                df = df[df['ts'] >= cutoff]
            
            if len(df) < self.min_training_samples:
                logger.warning(f"Insufficient data: {len(df)} < {self.min_training_samples}")
                # Try to fetch more historical data
                # In production, you'd implement pagination or historical data fetching
                pass
            
            logger.info(f"Collected {len(df)} samples for training")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return None
    
    def train_new_model(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train a new model version.
        
        Args:
            training_data: Training data (if None, will be collected)
            
        Returns:
            Training results dictionary
        """
        # Use Prefect workflow if enabled
        if self.use_prefect:
            return self._train_with_prefect(training_data)
        
        # Standard training workflow
        return self._train_standard(training_data)
    
    def _train_with_prefect(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train model using Prefect workflow."""
        try:
            logger.info("Starting model retraining with Prefect workflow...")
            
            # Get current model info for comparison
            current_version = self.get_current_model_version()
            current_model_path = None
            current_metrics = None
            
            if current_version and current_version in self.model_versions:
                current_info = self.model_versions[current_version]
                current_model_path = current_info.get('model_path')
                current_metrics = {
                    'validation_accuracy': current_info.get('metrics', {}).get('validation_accuracy', 0)
                }
            
            # Run Prefect workflow
            result = model_retraining_flow(
                symbol=self.symbol,
                exchange=self.exchange,
                provider=self.provider,
                days=30,
                model_dir=str(self.model_dir),
                alpaca_api_key=self.config.get('alpaca_api_key'),
                alpaca_secret_key=self.config.get('alpaca_secret_key'),
                current_model_path=current_model_path,
                current_metrics=current_metrics
            )
            
            if result.get('success'):
                # Update version history from Prefect result
                version = result['version']
                self.model_versions[version] = {
                    'version': version,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'model_path': result['model_path'],
                    'status': 'active' if result.get('activated') else 'testing',
                    'metrics': result['metrics'],
                    'feature_count': len(result.get('feature_importance', {}))
                }
                self.save_model_versions()
            
            return result
            
        except Exception as e:
            logger.error(f"Prefect workflow failed: {e}", exc_info=True)
            # Fallback to standard training
            logger.info("Falling back to standard training workflow...")
            return self._train_standard(training_data)
    
    def _train_standard(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Standard training workflow (non-Prefect)."""
        try:
            logger.info("Starting model retraining...")
            
            # Collect training data if not provided
            if training_data is None:
                training_data = self.collect_training_data(days=30)
            
            if training_data is None or len(training_data) < self.min_training_samples:
                return {'error': f'Insufficient training data: {len(training_data) if training_data is not None else 0} samples'}
            
            # Initialize model
            model = CryptoPredictionModel(
                model_type='classifier',
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Prepare data
            X, y = model.prepare_data(training_data)
            
            if len(X) < self.min_training_samples:
                return {'error': f'Insufficient features after preparation: {len(X)} samples'}
            
            # Train model
            results = model.train(X, y, validation_split=0.2)
            
            # Get validation metrics
            val_metrics = results.get('val_metrics', {})
            accuracy = val_metrics.get('accuracy', 0)
            
            # Create version identifier
            version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            # Save model
            model_path = self.model_dir / f'trained_model_{version}.pkl'
            model.save_model(str(model_path))
            
            # Update version history
            self.model_versions[version] = {
                'version': version,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'model_path': str(model_path),
                'status': 'testing',  # New models start as testing
                'metrics': {
                    'validation_accuracy': accuracy,
                    'validation_precision': val_metrics.get('precision', 0),
                    'validation_recall': val_metrics.get('recall', 0),
                    'validation_f1': val_metrics.get('f1_score', 0),
                    'training_samples': results.get('training_samples', 0),
                    'validation_samples': results.get('validation_samples', 0)
                },
                'feature_count': len(model.feature_names)
            }
            
            self.save_model_versions()
            
            logger.info(f"Model {version} trained with accuracy: {accuracy:.4f}")
            
            return {
                'success': True,
                'version': version,
                'model_path': str(model_path),
                'metrics': results['val_metrics'],
                'feature_importance': results.get('feature_importance', {})
            }
            
        except Exception as e:
            logger.error(f"Error training new model: {e}", exc_info=True)
            return {'error': str(e)}
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version1: First version identifier
            version2: Second version identifier
            
        Returns:
            Comparison results
        """
        if version1 not in self.model_versions or version2 not in self.model_versions:
            return {'error': 'One or both versions not found'}
        
        v1_info = self.model_versions[version1]
        v2_info = self.model_versions[version2]
        
        v1_metrics = v1_info.get('metrics', {})
        v2_metrics = v2_info.get('metrics', {})
        
        comparison = {
            'version1': {
                'version': version1,
                'accuracy': v1_metrics.get('validation_accuracy', 0),
                'f1_score': v1_metrics.get('validation_f1', 0),
                'created_at': v1_info.get('created_at')
            },
            'version2': {
                'version': version2,
                'accuracy': v2_metrics.get('validation_accuracy', 0),
                'f1_score': v2_metrics.get('validation_f1', 0),
                'created_at': v2_info.get('created_at')
            }
        }
        
        # Determine which is better
        v1_score = v1_metrics.get('validation_accuracy', 0)
        v2_score = v2_metrics.get('validation_accuracy', 0)
        
        if v2_score > v1_score + 0.01:  # 1% improvement threshold
            comparison['better_version'] = version2
            comparison['improvement'] = v2_score - v1_score
        elif v1_score > v2_score + 0.01:
            comparison['better_version'] = version1
            comparison['improvement'] = v1_score - v2_score
        else:
            comparison['better_version'] = 'tie'
            comparison['improvement'] = 0
        
        return comparison
    
    def activate_model(self, version: str) -> bool:
        """
        Activate a model version (makes it the active model).
        
        Args:
            version: Version identifier
            
        Returns:
            True if activation successful
        """
        if version not in self.model_versions:
            logger.error(f"Version {version} not found")
            return False
        
        # Deactivate current active model
        for v in self.model_versions:
            if self.model_versions[v].get('status') == 'active':
                self.model_versions[v]['status'] = 'archived'
        
        # Activate new model
        self.model_versions[version]['status'] = 'active'
        self.save_model_versions()
        
        # Create symlink to active model
        try:
            active_model_path = self.model_dir / 'trained_model.pkl'
            model_path = Path(self.model_versions[version]['model_path'])
            if active_model_path.exists():
                active_model_path.unlink()
            active_model_path.symlink_to(model_path)
            logger.info(f"Model {version} activated")
            return True
        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")
            return False
    
    def retrain_if_needed(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if retraining is needed and retrain if necessary.
        
        Args:
            force: Force retraining regardless of conditions
            
        Returns:
            Training results if retraining occurred, None otherwise
        """
        if self.should_retrain(force=force):
            result = self.train_new_model()
            if 'error' not in result:
                # Compare with current model
                current_version = self.get_current_model_version()
                if current_version:
                    comparison = self.compare_models(current_version, result['version'])
                    if comparison.get('better_version') == result['version']:
                        logger.info(f"New model {result['version']} is better, activating...")
                        self.activate_model(result['version'])
                    else:
                        logger.info(f"New model {result['version']} is not better, keeping {current_version}")
            return result
        else:
            logger.info("Model retraining not needed at this time")
            return None
    
    def get_model_history(self) -> List[Dict[str, Any]]:
        """Get history of all model versions."""
        return list(self.model_versions.values())
