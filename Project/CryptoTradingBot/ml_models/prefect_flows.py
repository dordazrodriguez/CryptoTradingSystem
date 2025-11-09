"""
Prefect workflows for ML model training and retraining.
Provides orchestration, monitoring, and error handling for model pipelines.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import pandas as pd
from pathlib import Path

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.logging import get_run_logger

from ml_models.predictor import CryptoPredictionModel
from ml_models.evaluation import ModelEvaluator
from ml_models.retraining_service import ModelRetrainingService
from data.data_feeder import DataFeed, FeedConfig, AlpacaFeed, AlpacaConfig, normalize_for_provider

logger = logging.getLogger(__name__)


@task(name="collect_training_data", retries=2, retry_delay_seconds=30)
def collect_training_data_task(
    symbol: str,
    provider: str,
    exchange: str,
    days: int,
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Task to collect training data from exchange.
    
    Args:
        symbol: Trading symbol
        provider: Data provider ("ccxt" or "alpaca")
        exchange: Exchange name
        days: Number of days of historical data
        alpaca_api_key: Alpaca API key (if using Alpaca)
        alpaca_secret_key: Alpaca secret key (if using Alpaca)
        
    Returns:
        DataFrame with OHLCV data
    """
    run_logger = get_run_logger()
    run_logger.info(f"Collecting training data for {symbol} from last {days} days...")
    
    feed_symbol = normalize_for_provider(provider, symbol, use_for="data")
    
    if provider == "alpaca":
        if not alpaca_api_key or not alpaca_secret_key:
            raise ValueError("Alpaca API keys required when using Alpaca provider")
        data_feed = AlpacaFeed(AlpacaConfig(
            symbol=feed_symbol,
            timeframe="1m",
            limit=days * 1440,  # Rough estimate: 1440 minutes per day
            api_key=alpaca_api_key,
            api_secret=alpaca_secret_key
        ))
    else:
        feed_config = FeedConfig(
            exchange=exchange,
            symbol=feed_symbol,
            timeframe="1m",
            limit=days * 1440
        )
        data_feed = DataFeed(feed_config)
    
    df = data_feed.fetch_ohlcv()
    
    if df.empty:
        raise ValueError(f"No data collected for {symbol}")
    
    run_logger.info(f"Collected {len(df)} samples for training")
    return df


@task(name="prepare_features")
def prepare_features_task(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Task to prepare features from raw data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with prepared features and target
    """
    run_logger = get_run_logger()
    run_logger.info("Preparing features for training...")
    
    model = CryptoPredictionModel()
    X, y = model.prepare_data(df)
    
    run_logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
    
    return {
        'X': X,
        'y': y,
        'feature_names': model.feature_names
    }


@task(name="train_model")
def train_model_task(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'classifier',
    n_estimators: int = 100,
    max_depth: int = 10
) -> Dict[str, Any]:
    """
    Task to train a new model.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: Model type ('classifier' or 'regressor')
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        
    Returns:
        Training results dictionary
    """
    run_logger = get_run_logger()
    run_logger.info(f"Training {model_type} model with {n_estimators} estimators...")
    
    model = CryptoPredictionModel(
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    results = model.train(X, y, validation_split=0.2)
    
    # Get validation metrics
    val_metrics = results.get('val_metrics', {})
    accuracy = val_metrics.get('accuracy', val_metrics.get('r2_score', 0))
    
    run_logger.info(f"Model training completed with validation accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'metrics': results.get('val_metrics', {}),
        'train_metrics': results.get('train_metrics', {}),
        'feature_importance': results.get('feature_importance', {}),
        'feature_names': model.feature_names,
        'training_samples': results.get('training_samples', 0),
        'validation_samples': results.get('validation_samples', 0)
    }


@task(name="save_model")
def save_model_task(
    model: CryptoPredictionModel,
    model_dir: str,
    version: str
) -> str:
    """
    Task to save trained model.
    
    Args:
        model: Trained model
        model_dir: Directory to save model
        version: Model version identifier
        
    Returns:
        Path to saved model
    """
    run_logger = get_run_logger()
    run_logger.info(f"Saving model version {version}...")
    
    model_path = Path(model_dir) / f'trained_model_{version}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(model_path))
    
    run_logger.info(f"Model saved to {model_path}")
    return str(model_path)


@task(name="evaluate_model_performance")
def evaluate_model_performance_task(
    model: CryptoPredictionModel,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Task to evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Evaluation metrics
    """
    run_logger = get_run_logger()
    run_logger.info("Evaluating model performance...")
    
    evaluator = ModelEvaluator()
    
    # Make predictions
    X_test_scaled = model.scaler.transform(X_test)
    predictions = model.model.predict(X_test_scaled)
    
    if model.model_type == 'classifier':
        metrics = evaluator.evaluate_classification_model(y_test, pd.Series(predictions))
    else:
        metrics = evaluator.evaluate_regression_model(y_test, pd.Series(predictions))
    
    run_logger.info(f"Model evaluation completed. Accuracy/R2: {metrics.get('accuracy', metrics.get('r2_score', 0)):.4f}")
    
    return metrics


@task(name="compare_models")
def compare_models_task(
    current_model_path: Optional[str],
    new_model_path: str,
    current_metrics: Optional[Dict[str, Any]],
    new_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Task to compare new model with current model.
    
    Args:
        current_model_path: Path to current active model
        new_model_path: Path to newly trained model
        current_metrics: Metrics of current model
        new_metrics: Metrics of new model
        
    Returns:
        Comparison results
    """
    run_logger = get_run_logger()
    run_logger.info("Comparing models...")
    
    if not current_model_path or not current_metrics:
        run_logger.info("No existing model found, new model will be activated")
        return {
            'better_model': 'new',
            'improvement': 0.0,
            'should_activate': True
        }
    
    # Compare metrics
    current_score = current_metrics.get('accuracy', current_metrics.get('r2_score', 0))
    new_score = new_metrics.get('accuracy', new_metrics.get('r2_score', 0))
    
    improvement = new_score - current_score
    improvement_threshold = 0.01  # 1% improvement required
    
    comparison = {
        'current_score': current_score,
        'new_score': new_score,
        'improvement': improvement,
        'improvement_pct': (improvement / current_score * 100) if current_score > 0 else 0,
        'should_activate': improvement > improvement_threshold
    }
    
    if comparison['should_activate']:
        run_logger.info(f"New model is better! Improvement: {improvement:.4f} ({comparison['improvement_pct']:.2f}%)")
        comparison['better_model'] = 'new'
    else:
        run_logger.info(f"Current model is better. New model improvement: {improvement:.4f}")
        comparison['better_model'] = 'current'
    
    return comparison


@task(name="activate_model")
def activate_model_task(
    model_path: str,
    model_dir: str,
    version: str,
    metrics: Dict[str, Any]
) -> bool:
    """
    Task to activate a model version.
    
    Args:
        model_path: Path to model file
        model_dir: Model directory
        version: Model version
        metrics: Model metrics
        
    Returns:
        True if activation successful
    """
    run_logger = get_run_logger()
    run_logger.info(f"Activating model version {version}...")
    
    from ml_models.retraining_service import ModelRetrainingService
    
    # Load retraining service to use version management
    config = {
        'model_dir': model_dir,
        'symbol': 'BTC/USDT',  # Default, should be configurable
        'exchange': 'binance',
        'provider': 'ccxt'
    }
    retraining_service = ModelRetrainingService(config)
    
    # Update version history
    retraining_service.model_versions[version] = {
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_path': model_path,
        'status': 'active',
        'metrics': metrics,
        'feature_count': len(metrics.get('feature_names', []))
    }
    retraining_service.save_model_versions()
    
    # Create symlink to active model
    try:
        active_model_path = Path(model_dir) / 'trained_model.pkl'
        if active_model_path.exists() or active_model_path.is_symlink():
            active_model_path.unlink()
        active_model_path.symlink_to(Path(model_path))
        run_logger.info(f"Model {version} activated successfully")
        return True
    except Exception as e:
        run_logger.error(f"Failed to activate model: {e}")
        return False


@flow(
    name="model-retraining-pipeline",
    description="Complete pipeline for ML model retraining with evaluation and activation",
    task_runner=SequentialTaskRunner(),
    log_prints=True
)
def model_retraining_flow(
    symbol: str = "BTC/USD",
    exchange: str = "alpaca",
    provider: str = "alpaca",
    days: int = 30,
    model_dir: str = "ml_models",
    model_type: str = "classifier",
    n_estimators: int = 100,
    max_depth: int = 10,
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
    current_model_path: Optional[str] = None,
    current_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete workflow for model retraining.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        provider: Data provider
        days: Days of historical data
        model_dir: Model directory
        model_type: Model type
        n_estimators: Number of trees
        max_depth: Max tree depth
        alpaca_api_key: Alpaca API key (if using Alpaca)
        alpaca_secret_key: Alpaca secret key (if using Alpaca)
        current_model_path: Path to current active model
        current_metrics: Metrics of current model
        
    Returns:
        Workflow results
    """
    logger.info("=" * 60)
    logger.info("Starting Model Retraining Pipeline")
    logger.info("=" * 60)
    
    # Generate version identifier
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Step 1: Collect training data
        training_data = collect_training_data_task(
            symbol=symbol,
            provider=provider,
            exchange=exchange,
            days=days,
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key
        )
        
        # Step 2: Prepare features
        features = prepare_features_task(training_data)
        
        # Step 3: Train model
        training_results = train_model_task(
            X=features['X'],
            y=features['y'],
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
        # Step 4: Save model
        model_path = save_model_task(
            model=training_results['model'],
            model_dir=model_dir,
            version=version
        )
        
        # Step 5: Evaluate model
        # For evaluation, we'd typically use a held-out test set
        # Here we use validation metrics from training
        new_metrics = {
            'validation_accuracy': training_results['metrics'].get('accuracy', training_results['metrics'].get('r2_score', 0)),
            'validation_precision': training_results['metrics'].get('precision', 0),
            'validation_recall': training_results['metrics'].get('recall', 0),
            'validation_f1': training_results['metrics'].get('f1_score', 0),
            'training_samples': training_results['training_samples'],
            'validation_samples': training_results['validation_samples']
        }
        
        # Step 6: Compare with current model
        comparison = compare_models_task(
            current_model_path=current_model_path,
            new_model_path=model_path,
            current_metrics=current_metrics,
            new_metrics=new_metrics
        )
        
        # Step 7: Activate if better
        activated = False
        if comparison['should_activate']:
            activated = activate_model_task(
                model_path=model_path,
                model_dir=model_dir,
                version=version,
                metrics=new_metrics
            )
        
        result = {
            'success': True,
            'version': version,
            'model_path': model_path,
            'metrics': new_metrics,
            'comparison': comparison,
            'activated': activated,
            'feature_importance': training_results.get('feature_importance', {})
        }
        
        logger.info("=" * 60)
        logger.info("Model Retraining Pipeline Completed Successfully")
        logger.info(f"Version: {version}")
        logger.info(f"Activated: {activated}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Model retraining pipeline failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'version': version
        }


@flow(
    name="data-collection-pipeline",
    description="Pipeline for collecting and processing market data",
    task_runner=SequentialTaskRunner()
)
def data_collection_flow(
    symbol: str = "BTC/USD",
    exchange: str = "alpaca",
    provider: str = "alpaca",
    hours: int = 24,
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Workflow for collecting market data.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        provider: Data provider
        hours: Hours of data to collect
        alpaca_api_key: Alpaca API key (if using Alpaca)
        alpaca_secret_key: Alpaca secret key (if using Alpaca)
        
    Returns:
        DataFrame with collected data
    """
    logger.info(f"Starting data collection for {symbol} (last {hours} hours)...")
    
    data = collect_training_data_task(
        symbol=symbol,
        provider=provider,
        exchange=exchange,
        days=max(1, hours // 24),
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key
    )
    
    logger.info(f"Data collection completed: {len(data)} samples")
    return data


# Example usage and testing
if __name__ == "__main__":
    # Test the workflow locally
    result = model_retraining_flow(
        symbol="BTC/USDT",
        exchange="binance",
        provider="ccxt",
        days=7,  # Use smaller dataset for testing
        model_dir="ml_models",
        current_model_path=None  # No existing model
    )
    print(f"\nWorkflow result: {result}")
