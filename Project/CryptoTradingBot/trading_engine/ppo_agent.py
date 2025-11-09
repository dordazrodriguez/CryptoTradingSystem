"""
PPO agent wrapper for trading decisions.
Integrates Stable-Baselines3 PPO with the trading environment.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging
import os

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    logging.warning("stable-baselines3 not available. Install with: pip install stable-baselines3")

from trading_engine.ppo_env import TradingEnv, TradingEnvConfig

logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            if episode_info:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        
        return True


class PPOAgent:
    """
    PPO agent for trading decisions.
    
    Wraps Stable-Baselines3 PPO algorithm for cryptocurrency trading.
    """
    
    def __init__(self, 
                 env: TradingEnv,
                 model_path: Optional[str] = None,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 verbose: int = 1):
        """
        Initialize PPO agent.
        
        Args:
            env: Trading environment
            model_path: Path to load existing model (optional)
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            verbose: Verbosity level
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3[extra]")
        
        self.env = env
        self.model_path = model_path
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Load existing model or create new one
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading PPO model from {model_path}")
            self.model = PPO.load(model_path, env=self.vec_env, verbose=verbose)
        else:
            logger.info("Creating new PPO model")
            self.model = PPO(
                policy='MlpPolicy',
                env=self.vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=verbose,
                tensorboard_log="./logs/ppo_tensorboard/"
            )
    
    def train(self, 
              total_timesteps: int = 100000,
              callback: Optional[BaseCallback] = None,
              log_interval: int = 10) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of training timesteps
            callback: Training callback
            log_interval: Logging interval
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("PPO training completed")
        
        return {
            'total_timesteps': total_timesteps,
            'status': 'completed'
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """
        Predict action given observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and action probabilities
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, _states
    
    def predict_with_prob(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Predict action with probability.
        
        Args:
            observation: Current state observation
            
        Returns:
            Action and probability
        """
        action, _states = self.model.predict(observation, deterministic=False)
        # For discrete actions, get action probabilities
        action_probs = self.model.policy.get_distribution(self.model.policy.obs_to_tensor(observation)[0]).distribution.probs
        action_probs = action_probs.detach().numpy()[0]
        
        probability = float(action_probs[action])
        
        return int(action), probability
    
    def save(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"PPO model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = PPO.load(filepath, env=self.vec_env)
        logger.info(f"PPO model loaded from {filepath}")


class HybridPPOStrategy:
    """
    Hybrid strategy combining Trend Following + ML + PPO.
    
    This is the integration layer that combines:
    1. Trend filters (EMA, ADX, ATR)
    2. ML predictions (LightGBM probability)
    3. PPO agent decisions (when to enter/exit/size)
    """
    
    def __init__(self,
                 ppo_agent: PPOAgent,
                 ml_model: Optional[Any] = None,
                 trend_filter_enabled: bool = True,
                 ml_filter_enabled: bool = True):
        """
        Initialize hybrid strategy.
        
        Args:
            ppo_agent: Trained PPO agent
            ml_model: Trained ML model (LightGBM)
            trend_filter_enabled: Enable trend filtering
            ml_filter_enabled: Enable ML filtering
        """
        self.ppo_agent = ppo_agent
        self.ml_model = ml_model
        self.trend_filter_enabled = trend_filter_enabled
        self.ml_filter_enabled = ml_filter_enabled
        
        logger.info("Hybrid PPO strategy initialized")
    
    def get_decision(self, 
                    state: np.ndarray,
                    current_price: float,
                    indicators: Dict[str, float],
                    ml_probability: float = 0.5) -> Dict[str, Any]:
        """
        Get trading decision from hybrid strategy.
        
        Args:
            state: Current state observation
            indicators: Technical indicators (EMA, ADX, ATR, etc.)
            ml_probability: ML prediction probability
            current_price: Current asset price
            
        Returns:
            Decision dictionary with action, confidence, and reasoning
        """
        # Get PPO action
        ppo_action, ppo_prob = self.ppo_agent.predict_with_prob(state)
        
        # Apply trend filter
        trend_signal = self._get_trend_signal(indicators)
        
        # Apply ML filter
        ml_signal = self._get_ml_signal(ml_probability)
        
        # Final decision
        final_action = ppo_action
        confidence = ppo_prob
        reasoning = []
        
        # Trend filter check
        if self.trend_filter_enabled:
            if trend_signal == 'no_trend' and ppo_action != 0:
                # No strong trend, override PPO to hold
                final_action = 0
                reasoning.append("Trend filter: No strong trend detected, holding")
                confidence *= 0.7
            elif trend_signal == 'uptrend' and ppo_action == 2:
                # Uptrend detected but PPO wants to short - override
                final_action = 0
                reasoning.append("Trend filter: Uptrend detected, preventing short")
                confidence *= 0.8
            elif trend_signal == 'downtrend' and ppo_action == 1:
                # Downtrend detected but PPO wants to long - override
                final_action = 0
                reasoning.append("Trend filter: Downtrend detected, preventing long")
                confidence *= 0.8
        
        # ML filter check
        if self.ml_filter_enabled and self.ml_model:
            if ml_signal == 'bearish' and final_action == 1:
                # ML predicts down but PPO wants long - reduce confidence
                final_action = 0
                reasoning.append("ML filter: Bearish prediction, holding")
                confidence *= 0.6
            elif ml_signal == 'bullish' and final_action == 2:
                # ML predicts up but PPO wants short - reduce confidence
                final_action = 0
                reasoning.append("ML filter: Bullish prediction, preventing short")
                confidence *= 0.7
        
        # Build reasoning
        if not reasoning:
            reasoning.append(f"PPO action: {['Hold', 'Long', 'Short'][ppo_action]}")
            if trend_signal != 'no_trend':
                reasoning.append(f"Trend: {trend_signal}")
            if ml_signal != 'neutral':
                reasoning.append(f"ML: {ml_signal}")
        
        return {
            'action': final_action,
            'action_name': ['Hold', 'Long', 'Short'][final_action],
            'confidence': float(confidence),
            'ppo_action': ppo_action,
            'ppo_probability': float(ppo_prob),
            'trend_signal': trend_signal,
            'ml_signal': ml_signal,
            'reasoning': ' | '.join(reasoning)
        }
    
    def _get_trend_signal(self, indicators: Dict[str, float]) -> str:
        """
        Get trend signal from indicators.
        
        Args:
            indicators: Dictionary of indicator values
            
        Returns:
            'uptrend', 'downtrend', or 'no_trend'
        """
        adx = indicators.get('adx', 0)
        ema_12 = indicators.get('ema_12', 0)
        ema_26 = indicators.get('ema_26', 0)
        
        # Need strong trend (ADX > 25)
        if adx < 25:
            return 'no_trend'
        
        # Uptrend: fast EMA above slow EMA
        if ema_12 > ema_26:
            return 'uptrend'
        
        # Downtrend: fast EMA below slow EMA
        if ema_12 < ema_26:
            return 'downtrend'
        
        return 'no_trend'
    
    def _get_ml_signal(self, ml_probability: float) -> str:
        """
        Get ML signal from probability.
        
        Args:
            ml_probability: ML prediction probability (0-1)
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if ml_probability > 0.6:
            return 'bullish'
        elif ml_probability < 0.4:
            return 'bearish'
        else:
            return 'neutral'

