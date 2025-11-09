"""YAML configuration loader with environment variable overrides."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from . import CONFIG_FILE


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, with environment variable overrides.
    
    Environment variables override YAML values using dot notation.
    Example: PPO_GAMMA=0.95 overrides config['ppo']['gamma']
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = CONFIG_FILE
    
    config = {}
    
    # Load YAML config if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            config = {}
    else:
        print(f"Config file not found at {config_path}, using defaults and environment variables")
    
    # Override with environment variables
    # PPO_GAMMA -> config['ppo']['gamma']
    # REWARD_TRANSACTION_COST -> config['reward']['transaction_cost_pct']
    env_prefixes = {
        'PPO_': 'ppo',
        'REWARD_': 'reward',
        'FEATURE_': 'features',
        'EXPERIENCE_': 'experience_storage',
        'CHECKPOINT_': 'checkpoints',
        'ML_': 'ml',
        'TRADING_': 'trading',
        'RISK_': 'risk',
        'LOGGING_': 'logging',
        'STRATEGY': 'strategy'
    }
    
    # Handle simple scalar overrides
    for key, value in os.environ.items():
        # Strategy override
        if key == 'TRADING_STRATEGY':
            config['strategy'] = value.lower()
        
        # PPO settings
        elif key.startswith('PPO_'):
            ppo_key = key[4:].lower()
            if '_' in ppo_key:
                # Handle nested keys like PPO_NETWORK_HIDDEN_SIZE
                parts = ppo_key.split('_')
                if parts[0] == 'network':
                    if 'ppo' not in config:
                        config['ppo'] = {}
                    if 'network' not in config['ppo']:
                        config['ppo']['network'] = {}
                    network_key = '_'.join(parts[1:])
                    config['ppo']['network'][network_key] = _convert_type(value)
                else:
                    if 'ppo' not in config:
                        config['ppo'] = {}
                    config['ppo'][ppo_key] = _convert_type(value)
            else:
                if 'ppo' not in config:
                    config['ppo'] = {}
                config['ppo'][ppo_key] = _convert_type(value)
        
        # Reward settings
        elif key.startswith('REWARD_'):
            reward_key = key[7:].lower()
            if 'reward' not in config:
                config['reward'] = {}
            # Map transaction_cost to transaction_cost_pct
            if reward_key == 'transaction_cost':
                config['reward']['transaction_cost_pct'] = _convert_type(value)
            else:
                config['reward'][reward_key] = _convert_type(value)
        
        # Feature settings
        elif key.startswith('FEATURE_'):
            feature_key = key[8:].lower()
            if 'features' not in config:
                config['features'] = {}
            config['features'][feature_key] = _convert_type(value)
        
        # Experience storage
        elif key.startswith('EXPERIENCE_STORAGE_'):
            exp_key = key[19:].lower()
            if 'experience_storage' not in config:
                config['experience_storage'] = {}
            config['experience_storage'][exp_key] = _convert_type(value)
        
        # ML settings
        elif key.startswith('ML_MODEL_PATH'):
            if 'ml' not in config:
                config['ml'] = {}
            config['ml']['model_path'] = value
        elif key == 'ENABLE_ML':
            if 'ml' not in config:
                config['ml'] = {}
            config['ml']['enable_ml'] = value.lower() in ('true', '1', 'yes')
        
        # Checkpoint settings
        elif key.startswith('CHECKPOINT_'):
            ckpt_key = key[11:].lower()
            if 'checkpoints' not in config:
                config['checkpoints'] = {}
            config['checkpoints'][ckpt_key] = _convert_type(value)
    
    return config


def _convert_type(value: str) -> Any:
    """Convert string to appropriate type (int, float, bool, or str)."""
    # Try bool first
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    
    # Try int
    try:
        if '.' not in value:
            return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def get_ppo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PPO configuration with defaults."""
    ppo_config = config.get('ppo', {})
    
    # Merge with defaults
    defaults = {
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'learning_rate': 3e-4,
        'update_epochs': 10,
        'batch_size': 64,
        'update_interval': 100,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'use_gae': True,
        'gae_lambda': 0.95,
        'action_mode': 'discrete',
        'network': {
            'hidden_layers': 2,
            'hidden_size': 128,
            'activation': 'relu'
        }
    }
    
    result = defaults.copy()
    result.update(ppo_config)
    
    # Merge network config
    if 'network' in ppo_config:
        result['network'].update(ppo_config['network'])
    
    return result


def merge_with_env_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config with environment variables, using config as base."""
    return load_config()

