"""
Prefect deployment configuration for model retraining workflows.
Deploy workflows to Prefect Cloud or local server.
"""

from prefect import serve
from ml_models.prefect_flows import model_retraining_flow, data_collection_flow


if __name__ == "__main__":
    # Deploy workflows as long-running services
    # These can be scheduled or triggered via API
    
    # Model retraining workflow - runs on schedule or trigger
    model_retraining_flow.serve(
        name="model-retraining",
        parameters={
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "provider": "ccxt",
            "days": 30,
            "model_dir": "ml_models"
        },
        interval=60 * 60 * 24 * 7,  # Run weekly (7 days in seconds)
        cron=None,  # Or use cron: "0 2 * * *" for daily at 2 AM
    )
    
    # Data collection workflow - can be triggered separately
    data_collection_flow.serve(
        name="data-collection",
        parameters={
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "provider": "ccxt",
            "hours": 24
        }
    )
