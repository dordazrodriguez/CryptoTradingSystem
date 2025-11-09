"""Checkpoint management with cleanup and validation."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil


class CheckpointManager:
    """Manages PPO agent checkpoints with cleanup and validation."""
    
    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory where checkpoints are stored
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def get_checkpoint_files(self) -> List[Path]:
        """Get all checkpoint files sorted by modification time (newest first)."""
        pattern = "ppo_agent*.pt"
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        return checkpoints
    
    def cleanup_old_checkpoints(self) -> int:
        """
        Remove old checkpoints, keeping only the most recent N.
        
        Returns:
            Number of checkpoints removed
        """
        checkpoints = self.get_checkpoint_files()
        
        if len(checkpoints) <= self.keep_last_n:
            return 0
        
        removed_count = 0
        for checkpoint in checkpoints[self.keep_last_n:]:
            try:
                checkpoint.unlink()
                removed_count += 1
            except Exception:
                pass
        
        return removed_count
    
    def validate_checkpoint(self, checkpoint_path: Path, state_dim: Optional[int] = None, action_dim: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint file
            state_dim: Expected state dimension (optional)
            action_dim: Expected action dimension (optional)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "checkpoint_data": None
        }
        
        if not checkpoint_path.exists():
            result["errors"].append(f"Checkpoint file does not exist: {checkpoint_path}")
            return result
        
        try:
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['policy_net', 'value_net', 'optimizer']
            for key in required_keys:
                if key not in checkpoint:
                    result["errors"].append(f"Missing required key: {key}")
            
            # Validate state dicts exist
            if 'policy_net' in checkpoint:
                if not isinstance(checkpoint['policy_net'], dict):
                    result["errors"].append("policy_net is not a state dict")
            
            if 'value_net' in checkpoint:
                if not isinstance(checkpoint['value_net'], dict):
                    result["errors"].append("value_net is not a state dict")
            
            # Optional: validate dimensions if provided
            if state_dim and 'policy_net' in checkpoint:
                # Try to infer input dimension from first layer
                policy_state = checkpoint['policy_net']
                if 'shared_layers.0.weight' in policy_state:
                    inferred_state_dim = policy_state['shared_layers.0.weight'].shape[1]
                    if inferred_state_dim != state_dim:
                        result["warnings"].append(
                            f"State dimension mismatch: checkpoint has {inferred_state_dim}, expected {state_dim}"
                        )
            
            result["checkpoint_data"] = checkpoint
            result["valid"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Failed to load checkpoint: {str(e)}")
        
        return result
    
    def get_best_checkpoint(self, state_dim: Optional[int] = None, action_dim: Optional[int] = None) -> Optional[Path]:
        """
        Get the most recent valid checkpoint.
        
        Args:
            state_dim: Expected state dimension (optional)
            action_dim: Expected action dimension (optional)
            
        Returns:
            Path to best checkpoint or None if none valid
        """
        checkpoints = self.get_checkpoint_files()
        
        for checkpoint in checkpoints:
            validation = self.validate_checkpoint(checkpoint, state_dim, action_dim)
            if validation["valid"]:
                return checkpoint
        
        return None

