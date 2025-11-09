#!/usr/bin/env python3
"""
Script to retrain the ML model.
Can be run manually or scheduled via cron.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.retraining_service import ModelRetrainingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for model retraining."""
    parser = argparse.ArgumentParser(description="Retrain ML model for trading bot")
    parser.add_argument("--force", action="store_true",
                       help="Force retraining regardless of conditions")
    parser.add_argument("--symbol", default="BTC/USD",
                       help="Trading symbol")
    parser.add_argument("--exchange", default="binance",
                       help="Exchange name")
    parser.add_argument("--provider", choices=["ccxt", "alpaca"], default="alpaca",
                       help="Data provider")
    parser.add_argument("--model-dir", default="ml_models",
                       help="Directory for storing models")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'model_dir': args.model_dir,
        'min_accuracy_threshold': 0.55,
        'performance_window_days': 7,
        'retrain_interval_days': 7,
        'min_training_samples': 1000,
        'symbol': args.symbol,
        'exchange': args.exchange,
        'provider': args.provider
    }
    
    # Add Alpaca credentials if using Alpaca
    if args.provider == "alpaca":
        config['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
        config['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
    
    # Initialize retraining service
    retraining_service = ModelRetrainingService(config)
    
    # Check if retraining is needed
    logger.info("Checking if model retraining is needed...")
    result = retraining_service.retrain_if_needed(force=args.force)
    
    if result:
        if 'error' in result:
            logger.error(f"Retraining failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info(f"Model retraining completed successfully")
            logger.info(f"Version: {result['version']}")
            logger.info(f"Metrics: {result['metrics']}")
            sys.exit(0)
    else:
        logger.info("Model retraining not needed at this time")
        sys.exit(0)


if __name__ == "__main__":
    main()
