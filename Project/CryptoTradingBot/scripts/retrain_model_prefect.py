#!/usr/bin/env python3
"""
Script to retrain the ML model using Prefect workflows.
Provides better orchestration, monitoring, and error handling.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.prefect_flows import model_retraining_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Prefect-based model retraining."""
    parser = argparse.ArgumentParser(description="Retrain ML model using Prefect workflow")
    parser.add_argument("--symbol", default="BTC/USD",
                       help="Trading symbol")
    parser.add_argument("--exchange", default="binance",
                       help="Exchange name")
    parser.add_argument("--provider", choices=["ccxt", "alpaca"], default="alpaca",
                       help="Data provider")
    parser.add_argument("--model-dir", default="ml_models",
                       help="Directory for storing models")
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data to collect")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees in Random Forest")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Maximum depth of trees")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Prefect Model Retraining Workflow")
    logger.info("=" * 60)
    
    # Run Prefect flow
    result = model_retraining_flow(
        symbol=args.symbol,
        exchange=args.exchange,
        provider=args.provider,
        days=args.days,
        model_dir=args.model_dir,
        model_type="classifier",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        alpaca_api_key=os.getenv('ALPACA_API_KEY'),
        alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
        current_model_path=None,  # Will be detected automatically
        current_metrics=None
    )
    
    if result.get('success'):
        logger.info("\n" + "=" * 60)
        logger.info("Model Retraining Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Version: {result['version']}")
        logger.info(f"Model Path: {result['model_path']}")
        logger.info(f"Metrics: {result['metrics']}")
        logger.info(f"Activated: {result.get('activated', False)}")
        if result.get('comparison'):
            comp = result['comparison']
            logger.info(f"Improvement: {comp.get('improvement', 0):.4f}")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error(f"\nModel retraining failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
