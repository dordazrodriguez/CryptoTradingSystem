"""Experience buffer for storing and retrieving training experiences."""

from typing import List, Dict, Any
import numpy as np


class ExperienceBuffer:
    """Buffer for storing experiences for PPO training."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_log_prob: float
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            action_log_prob: Log probability of action
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'action_log_prob': action_log_prob
        }
        
        self.buffer.append(experience)
        
        # Remove oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all experiences in buffer."""
        return self.buffer.copy()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.max_size

