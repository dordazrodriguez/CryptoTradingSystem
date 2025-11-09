"""
Simple trading strategy implementation from the completed project.
Provides Moving Average Crossover strategy as a reference implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossoverConfig:
    """Configuration for Moving Average Crossover strategy."""
    fast: int = 12
    slow: int = 26


class MovingAverageCrossover:
    """
    Simple Moving Average Crossover trading strategy.
    
    This is the descriptive method for the capstone project.
    Generates buy signals (+1) when fast MA crosses above slow MA.
    Generates sell signals (-1) when fast MA crosses below slow MA.
    """
    
    def __init__(self, cfg: CrossoverConfig) -> None:
        """
        Initialize Moving Average Crossover strategy.
        
        Args:
            cfg: Crossover configuration
        """
        if cfg.fast >= cfg.slow:
            raise ValueError("fast MA period must be < slow period")
        self.cfg = cfg
        logger.info(f"Initialized MA Crossover: fast={cfg.fast}, slow={cfg.slow}")

    def generate(self, df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            df: DataFrame with moving average columns
            fast_col: Name of fast MA column
            slow_col: Name of slow MA column
            
        Returns:
            Series with signals: +1 for buy, -1 for sell, 0 for no signal
        """
        # Initialize signal series
        signal = pd.Series(0, index=df.index, name="signal")
        
        # Buy signal: fast MA crosses above slow MA
        cond_buy = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
        
        # Sell signal: fast MA crosses below slow MA
        cond_sell = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
        
        # Apply signals
        signal = signal.mask(cond_buy, 1)
        signal = signal.mask(cond_sell, -1)
        
        return signal
    
    def get_signal_strength(self, df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
        """
        Calculate signal strength based on MA separation.
        
        Args:
            df: DataFrame with moving average columns
            fast_col: Name of fast MA column
            slow_col: Name of slow MA column
            
        Returns:
            Series with signal strength (0-1 scale)
        """
        ma_separation = (df[fast_col] - df[slow_col]) / df[slow_col]
        
        # Normalize to 0-1 range
        strength = abs(ma_separation)
        
        return strength


@dataclass
class MomentumConfig:
    """Configuration for Momentum strategy."""
    lookback: int = 14
    threshold: float = 0.01


class MomentumStrategy:
    """
    Momentum-based trading strategy.
    Uses price momentum to generate trading signals.
    """
    
    def __init__(self, cfg: MomentumConfig) -> None:
        """Initialize momentum strategy."""
        self.cfg = cfg
        logger.info(f"Initialized Momentum Strategy: lookback={cfg.lookback}")

    def generate(self, df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        """
        Generate momentum-based trading signals.
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            
        Returns:
            Series with signals: +1 for buy, -1 for sell, 0 for no signal
        """
        # Calculate momentum
        momentum = df[price_col].diff(self.cfg.lookback) / df[price_col]
        
        # Generate signals
        signal = pd.Series(0, index=df.index, name="signal")
        
        # Buy when momentum is positive and above threshold
        signal = signal.mask(momentum > self.cfg.threshold, 1)
        
        # Sell when momentum is negative and below threshold
        signal = signal.mask(momentum < -self.cfg.threshold, -1)
        
        return signal
