#!/usr/bin/env python3
"""
Script to train the ML model for the first time.
This will create a trained_model.pkl file that can be used by the continuous trader.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_models.predictor import CryptoPredictionModel
from data.data_feeder import DataFeed, FeedConfig, AlpacaFeed, AlpacaConfig, normalize_for_provider
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(symbol: str = "BTC/USD", exchange: str = "alpaca",
                provider: str = "alpaca", timeframe: str = "1m",
                limit: int = 2000, model_path: str = "ml_models/trained_model.pkl",
                algorithm: str = "random_forest"):
    """
    Train the ML model and save it.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        provider: Data provider ("ccxt" or "alpaca")
        timeframe: Data timeframe
        limit: Number of candles to fetch
        model_path: Path to save the model
    """
    logger.info("=" * 60)
    logger.info("Training ML Model")
    logger.info("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Create model directory if it doesn't exist
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data feed
    feed_symbol = normalize_for_provider(provider, symbol, use_for="data")
    
    if provider == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            logger.error("ALPACA_API_KEY/ALPACA_SECRET_KEY not set. Set them in .env or export them.")
            return False
        
        feed_config = AlpacaConfig(
            symbol=feed_symbol,
            timeframe=timeframe,
            limit=limit,
            api_key=api_key,
            api_secret=api_secret
        )
        feed = AlpacaFeed(feed_config)
    else:
        feed_config = FeedConfig(
            exchange=exchange,
            symbol=feed_symbol,
            timeframe=timeframe,
            limit=limit
        )
        feed = DataFeed(feed_config)
    
    # Fetch training data
    logger.info(f"Fetching {limit} candles of training data for {symbol}...")
    df = feed.fetch_ohlcv()
    
    if df.empty:
        logger.error("Failed to fetch training data")
        return False
    
    logger.info(f"Fetched {len(df)} candles")
    
    # Initialize model with algorithm selection
    logger.info(f"Initializing {algorithm.upper()} Classifier...")
    model = CryptoPredictionModel(
        algorithm=algorithm,
        model_type='classifier',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Prepare data
    logger.info("Preparing features...")
    try:
        X, y = model.prepare_data(df)
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return False
    
    # Reduce minimum samples requirement if data is limited (allow as few as 50)
    min_samples = 50
    if len(X) < min_samples:
        logger.error(f"Insufficient data: {len(X)} samples (need at least {min_samples})")
        logger.error(f"Try fetching more data by increasing --limit (current: {limit})")
        logger.error(f"Note: After feature engineering, many rows are dropped due to NaN values from lookback windows")
        return False
    elif len(X) < 100:
        logger.warning(f"Limited data: {len(X)} samples (recommended: 100+, but proceeding with {min_samples}+)")
    
    # Train model
    logger.info("Training model...")
    try:
        results = model.train(X, y, validation_split=0.2)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False
    
    # Display results
    val_metrics = results.get('val_metrics', {})
    train_metrics = results.get('train_metrics', {})
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)
    logger.info(f"Training Accuracy: {train_metrics.get('accuracy', 0):.2%}")
    logger.info(f"Validation Accuracy: {val_metrics.get('accuracy', 0):.2%}")
    logger.info(f"Validation Precision: {val_metrics.get('precision', 0):.3f}")
    logger.info(f"Validation Recall: {val_metrics.get('recall', 0):.3f}")
    logger.info(f"Validation F1-Score: {val_metrics.get('f1_score', 0):.3f}")
    logger.info(f"Training Samples: {results.get('training_samples', 0)}")
    logger.info(f"Validation Samples: {results.get('validation_samples', 0)}")
    logger.info("=" * 60)
    
    # Save model
    logger.info(f"Saving model to {model_path}...")
    try:
        model.save_model(model_path)
        logger.info(f"✅ Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False
    
    # Verify file exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        logger.info(f"✅ Model file verified: {file_size:.2f} MB")
        return True
    else:
        logger.error("❌ Model file not found after saving")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ML model for trading bot")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--exchange", default="alpaca", help="Exchange name")
    parser.add_argument("--provider", choices=["ccxt", "alpaca"], default="alpaca",
                       help="Data provider")
    parser.add_argument("--algorithm", 
                       choices=["random_forest", "xgboost", "lightgbm"],
                       default="random_forest",
                       help="ML algorithm to use")
    parser.add_argument("--timeframe", default="1m", help="Data timeframe")
    parser.add_argument("--limit", type=int, default=2000,
                       help="Number of candles to fetch")
    parser.add_argument("--model-path", default="ml_models/trained_model.pkl",
                       help="Path to save the model")
    
    args = parser.parse_args()
    
    success = train_model(
        symbol=args.symbol,
        exchange=args.exchange,
        provider=args.provider,
        timeframe=args.timeframe,
        limit=args.limit,
        model_path=args.model_path,
        algorithm=args.algorithm
    )
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Model training completed successfully!")
        logger.info(f"✅ Model saved to: {args.model_path}")
        logger.info("=" * 60)
        logger.info(f"\nModel trained with {args.algorithm.upper()} algorithm")
        logger.info("\nTo use the model in continuous trading:")
        logger.info("1. Set TRADING_STRATEGY=decision_support")
        logger.info("2. Run: python main.py --mode run --enable-ml")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("\n❌ Model training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

