"""
Test script to verify PPO/RL integration.
Run this from the CryptoTradingBot directory:
    python test_ppo_integration.py
"""

import sys
import os

# Ensure we're in the right directory
if not os.path.exists('trading_engine'):
    print("❌ Error: Please run this script from the CryptoTradingBot directory")
    print("   Current directory:", os.getcwd())
    sys.exit(1)

print("=" * 60)
print("Testing PPO/RL Integration")
print("=" * 60)

# Test 1: Import RL components
print("\n1. Testing imports...")
try:
    from trading_engine.rl import PPOAgent, ExperienceBuffer, RewardCalculator
    print("   [OK] PPOAgent imported")
    print("   [OK] ExperienceBuffer imported")
    print("   [OK] RewardCalculator imported")
except ImportError as e:
    print(f"   [ERROR] Import failed: {e}")
    sys.exit(1)

# Test 2: Import feature pipeline
print("\n2. Testing feature pipeline import...")
try:
    from ml_models.rl_feature_pipeline import RLFeaturePipeline
    print("   [OK] RLFeaturePipeline imported")
except ImportError as e:
    print(f"   [ERROR] Import failed: {e}")
    sys.exit(1)

# Test 3: Import experience storage
print("\n3. Testing experience storage import...")
try:
    from trading_engine.rl.experience.experience_storage import PersistentExperienceStorage
    print("   [OK] PersistentExperienceStorage imported")
except ImportError as e:
    print(f"   [ERROR] Import failed: {e}")
    sys.exit(1)

# Test 4: Check PyTorch availability
print("\n4. Testing PyTorch availability...")
try:
    import torch
    print(f"   [OK] PyTorch version: {torch.__version__}")
    print(f"   [OK] CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("   [ERROR] PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

# Test 5: Create PPO agent (minimal test)
print("\n5. Testing PPO agent creation...")
try:
    import numpy as np
    
    # Create a simple config
    config = {
        'ppo': {
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'learning_rate': 3e-4,
            'update_epochs': 10,
            'batch_size': 64,
            'update_interval': 100
        },
        'action_mode': 'discrete'
    }
    
    state_dim = 50  # Example state dimension
    action_dim = 3  # Buy, Hold, Sell
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_mode='discrete',
        config=config
    )
    print(f"   [OK] PPO agent created (state_dim={state_dim}, action_dim={action_dim})")
    
    # Test action selection
    test_state = np.random.randn(state_dim).astype(np.float64)
    action, log_prob, info = agent.act(test_state, deterministic=False, return_probs=True)
    print(f"   [OK] Action selection works (action={action}, log_prob={log_prob:.4f})")
    
except Exception as e:
    print(f"   [ERROR] Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test experience buffer
print("\n6. Testing experience buffer...")
try:
    buffer = ExperienceBuffer(max_size=100)
    buffer.add(
        state=np.array([1.0, 2.0, 3.0]),
        action=1,
        reward=0.5,
        next_state=np.array([1.1, 2.1, 3.1]),
        done=False,
        action_log_prob=-1.5
    )
    print(f"   [OK] Experience buffer works (size={len(buffer)})")
except Exception as e:
    print(f"   [ERROR] Experience buffer failed: {e}")
    sys.exit(1)

# Test 7: Test reward calculator
print("\n7. Testing reward calculator...")
try:
    reward_calc = RewardCalculator(config={
        'reward': {'transaction_cost_pct': 0.001},
        'trading': {'short_selling_enabled': False}
    })
    reward = reward_calc.calculate_step_reward(
        previous_portfolio_value=10000.0,
        current_portfolio_value=10100.0,
        previous_position_size=0.0,
        current_position_size=0.1,
        previous_action=1,
        current_action=0,
        price_change_pct=0.01
    )
    print(f"   [OK] Reward calculator works (reward={reward:.6f})")
except Exception as e:
    print(f"   [ERROR] Reward calculator failed: {e}")
    sys.exit(1)

# Test 8: Test feature pipeline (requires pandas and indicators)
print("\n8. Testing feature pipeline...")
try:
    import pandas as pd
    from trading_engine.indicators import TechnicalIndicators
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='1min')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(200) * 100,
        'high': 50100 + np.random.randn(200) * 100,
        'low': 49900 + np.random.randn(200) * 100,
        'close': 50000 + np.random.randn(200) * 100,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    pipeline = RLFeaturePipeline(config={
        'features': {'lookback_window': 100, 'normalization_method': 'z-score'}
    })
    
    state = pipeline.compute_features(
        df=sample_df,
        portfolio_value=10000.0,
        cash=5000.0,
        position_size=0.1,
        last_action=1,
        ml_prediction=0.6
    )
    
    print(f"   [OK] Feature pipeline works (state_dim={len(state)})")
    print(f"   [OK] State dimension: {len(state)}")
    
except Exception as e:
    print(f"   [ERROR] Feature pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed! PPO/RL integration is working correctly.")
print("=" * 60)
print("\nTo use PPO strategy:")
print("  1. Set environment variable: export TRADING_STRATEGY=ppo_rl")
print("  2. Run: python main.py --mode run --symbol BTC/USD --provider alpaca")
print("=" * 60)

