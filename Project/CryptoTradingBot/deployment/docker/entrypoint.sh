#!/bin/sh
# Entrypoint script for trader service
# Builds command dynamically based on environment variables

set -e

# Base command
CMD="python main.py --mode run --provider alpaca --symbol BTC/USD --timeframe 1m --interval 15"

# Add ML flag if enabled
if [ "$ENABLE_ML" = "true" ]; then
    CMD="$CMD --enable-ml"
    
    # Add model path if specified
    if [ -n "$ML_MODEL_PATH" ]; then
        CMD="$CMD --model-path $ML_MODEL_PATH"
    fi
fi

# Add auto-retraining flag if enabled
if [ "$ENABLE_AUTO_RETRAINING" = "true" ]; then
    CMD="$CMD --enable-auto-retraining"
fi

# Execute command
exec $CMD

