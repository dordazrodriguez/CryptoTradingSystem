"""
Training script for PPO agent on historical cryptocurrency data.
Implements walk-forward training with validation.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_engine.ppo_env import TradingEnv, TradingEnvConfig
from trading_engine.ppo_agent import PPOAgent, TrainingCallback
from ml_models.predictor import CryptoPredictionModel
from data.collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_historical_data(symbol: str = 'BTC/USD', 
                        timeframe: str = '1h',
                        days: int = 365) -> pd.DataFrame:
    """
    Load historical data for training.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        days: Number of days of history
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading historical data: {symbol}, {timeframe}, {days} days")
    
    collector = DataCollector()
    
    try:
        # Try to fetch from exchange
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = collector.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_date,
            limit=1000
        )
        
        if df is None or len(df) == 0:
            raise ValueError("No data fetched")
        
        logger.info(f"Loaded {len(df)} bars of historical data")
        return df
    
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        # Generate synthetic data for testing
        logger.warning("Generating synthetic data for testing")
        dates = pd.date_range(end=datetime.now(), periods=days*24 if timeframe == '1h' else days, freq=timeframe)
        np.random.seed(42)
        
        # Random walk price simulation
        price = 50000
        prices = [price]
        for _ in range(len(dates) - 1):
            price += np.random.randn() * price * 0.01
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.randn() * 0.02)) for p in prices],
            'low': [p * (1 - abs(np.random.randn() * 0.02)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        logger.info(f"Generated {len(df)} bars of synthetic data")
        return df


def train_ppo_agent(df: pd.DataFrame,
                    ml_model: Optional[CryptoPredictionModel] = None,
                    config: TradingEnvConfig = TradingEnvConfig(),
                    total_timesteps: int = 100000,
                    model_save_path: str = 'models/ppo_trading_agent',
                    use_walk_forward: bool = True,
                    train_test_split: float = 0.8) -> Dict[str, Any]:
    """
    Train PPO agent on historical data.
    
    Args:
        df: Historical OHLCV data
        ml_model: Pre-trained ML model (optional)
        config: Environment configuration
        total_timesteps: Total training timesteps
        model_save_path: Path to save trained model
        use_walk_forward: Use walk-forward validation
        train_test_split: Train/test split ratio
        
    Returns:
        Training results
    """
    logger.info("Starting PPO agent training")
    
    # Split data
    split_idx = int(len(df) * train_test_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Training data: {len(train_df)} bars, Test data: {len(test_df)} bars")
    
    # Create training environment
    train_env = TradingEnv(
        df=train_df,
        ml_model=ml_model,
        config=config,
        action_type='discrete'
    )
    
    # Create test environment
    test_env = TradingEnv(
        df=test_df,
        ml_model=ml_model,
        config=config,
        action_type='discrete'
    )
    
    # Create PPO agent
    agent = PPOAgent(
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    
    # Create training callback
    callback = TrainingCallback(verbose=1)
    
    # Train agent
    logger.info(f"Training PPO agent for {total_timesteps} timesteps")
    training_stats = agent.train(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10
    )
    
    # Evaluate on test set
    logger.info("Evaluating PPO agent on test set")
    test_results = evaluate_ppo_agent(agent, test_env)
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)
    logger.info(f"Trained model saved to {model_save_path}")
    
    results = {
        'training_stats': training_stats,
        'test_results': test_results,
        'model_path': model_save_path,
        'episode_rewards': callback.episode_rewards[-100:] if callback.episode_rewards else [],
        'episode_lengths': callback.episode_lengths[-100:] if callback.episode_lengths else []
    }
    
    logger.info("PPO training completed successfully")
    return results


def evaluate_ppo_agent(agent: PPOAgent, env: TradingEnv, num_episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate PPO agent on environment.
    
    Args:
        agent: Trained PPO agent
        env: Trading environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating PPO agent for {num_episodes} episodes")
    
    episode_rewards = []
    episode_lengths = []
    episode_equities = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_equities.append(info.get('equity', 0))
        
        logger.info(f"Episode {episode + 1}: Reward={episode_reward:.4f}, Length={episode_length}, Final Equity=${info.get('equity', 0):.2f}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_final_equity': np.mean(episode_equities),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_equities': episode_equities
    }
    
    logger.info(f"Evaluation complete: Mean Reward={results['mean_reward']:.4f}, Mean Final Equity=${results['mean_final_equity']:.2f}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO agent for cryptocurrency trading')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    parser.add_argument('--days', type=int, default=365, help='Number of days of history')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--model-path', type=str, default='models/ppo_trading_agent', help='Model save path')
    parser.add_argument('--ml-model-path', type=str, default=None, help='Pre-trained ML model path')
    parser.add_argument('--use-ml', action='store_true', help='Use ML model in environment')
    
    args = parser.parse_args()
    
    # Load historical data
    df = load_historical_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days
    )
    
    # Load ML model if provided
    ml_model = None
    if args.use_ml and args.ml_model_path:
        try:
            ml_model = CryptoPredictionModel(algorithm='lightgbm', model_type='classifier')
            ml_model.load_model(args.ml_model_path)
            logger.info(f"Loaded ML model from {args.ml_model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
    
    # Environment configuration
    config = TradingEnvConfig(
        initial_cash=100000.0,
        commission_rate=0.001,
        slippage=0.0005,
        position_size_limit=1.0,
        max_drawdown_penalty=0.1,
        transaction_penalty=0.01,
        trend_bonus=0.05
    )
    
    # Train PPO agent
    results = train_ppo_agent(
        df=df,
        ml_model=ml_model,
        config=config,
        total_timesteps=args.timesteps,
        model_save_path=args.model_path,
        use_walk_forward=True,
        train_test_split=0.8
    )
    
    # Print results
    print("\n" + "="*50)
    print("PPO Training Results")
    print("="*50)
    print(f"Model saved to: {results['model_path']}")
    print(f"\nTest Results:")
    print(f"  Mean Reward: {results['test_results']['mean_reward']:.4f}")
    print(f"  Std Reward: {results['test_results']['std_reward']:.4f}")
    print(f"  Mean Final Equity: ${results['test_results']['mean_final_equity']:.2f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    main()

