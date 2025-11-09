"""PPO (Proximal Policy Optimization) reinforcement learning agent - Standalone PyTorch implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
from .experience.experience_buffer import ExperienceBuffer


class PolicyNetwork(nn.Module):
    """Policy network for PPO agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_mode: str = "discrete",
        hidden_layers: int = 2,
        hidden_size: int = 128,
        activation: str = "relu"
    ):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (1 for discrete, 1 for continuous)
            action_mode: 'discrete' or 'continuous'
            hidden_layers: Number of hidden layers
            hidden_size: Number of neurons per hidden layer
            activation: Activation function ('relu', 'tanh', etc.)
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_mode = action_mode
        
        # Build network layers
        layers = []
        input_size = state_dim
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            input_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layers
        if action_mode == "discrete":
            # Discrete actions: output logits for each action
            self.policy_head = nn.Linear(hidden_size, action_dim)
        else:
            # Continuous actions: output mean and std for normal distribution
            self.policy_mean = nn.Linear(hidden_size, action_dim)
            self.policy_std = nn.Linear(hidden_size, action_dim)
            self.policy_std.weight.data.fill_(0.0)  # Initialize std to small values
            self.policy_std.bias.data.fill_(-1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (action_logits/means, value_estimate)
        """
        x = self.shared_layers(state)
        
        if self.action_mode == "discrete":
            action_logits = self.policy_head(x)
            return action_logits, x  # Return logits and hidden state for value head
        else:
            mean = self.policy_mean(x)
            std = torch.clamp(torch.exp(self.policy_std(x)), min=1e-6, max=1.0)
            return mean, std, x


class ValueNetwork(nn.Module):
    """Value network for PPO agent."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_layers: int = 2,
        hidden_size: int = 128,
        activation: str = "relu"
    ):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_layers: Number of hidden layers
            hidden_size: Number of neurons per hidden layer
            activation: Activation function
        """
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_size = state_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Value estimate tensor of shape (batch_size, 1)
        """
        return self.network(state)


class PPOAgent:
    """PPO (Proximal Policy Optimization) agent for trading."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_mode: str = "discrete",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_mode: 'discrete' or 'continuous'
            config: Optional configuration dict with PPO settings.
                   Expected structure: {'ppo': {...}, 'action_mode': '...'}
            device: PyTorch device ('cpu' or 'cuda'). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if config is None:
            config = {}
        
        ppo_config = config.get("ppo", {})
        self.action_mode = action_mode or config.get("action_mode", "discrete")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters (convert to proper types)
        self.gamma = float(ppo_config.get("gamma", 0.99))
        self.clip_epsilon = float(ppo_config.get("clip_epsilon", 0.2))
        self.learning_rate = float(ppo_config.get("learning_rate", 3e-4))
        self.update_epochs = int(ppo_config.get("update_epochs", 10))
        self.batch_size = int(ppo_config.get("batch_size", 64))
        self.update_interval = int(ppo_config.get("update_interval", 100))
        self.value_coef = float(ppo_config.get("value_coef", 0.5))
        self.entropy_coef = float(ppo_config.get("entropy_coef", 0.01))
        
        # Advanced features
        self.use_gae = ppo_config.get("use_gae", True)
        self.gae_lambda = float(ppo_config.get("gae_lambda", 0.95))
        self.value_clipping = ppo_config.get("value_clipping", True)
        self.gradient_monitoring = ppo_config.get("gradient_monitoring", True)
        self.max_grad_norm = float(ppo_config.get("max_grad_norm", 10.0))
        self.early_stopping = ppo_config.get("early_stopping", True)
        self.adaptive_clip = ppo_config.get("adaptive_clip", False)
        self.base_clip_epsilon = self.clip_epsilon  # Store base for adaptive clipping
        
        network_config = ppo_config.get("network", {})
        hidden_layers = network_config.get("hidden_layers", 2)
        hidden_size = network_config.get("hidden_size", 128)
        activation = network_config.get("activation", "relu")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(
            state_dim, action_dim, self.action_mode,
            hidden_layers, hidden_size, activation
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim, hidden_layers, hidden_size, activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.learning_rate
        )
        
        # Learning rate scheduler
        lr_scheduler_type = ppo_config.get("lr_scheduler", "plateau")
        if lr_scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, 
                min_lr=1e-6
            )
        elif lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
        else:
            self.scheduler = None
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training tracking
        self.step_count = 0
        self.episode_rewards = deque(maxlen=100)
    
    def act(self, state: np.ndarray, deterministic: bool = False, return_probs: bool = False) -> Tuple[int, float, Optional[Dict[str, Any]]]:
        """
        Select action given current state.
        
        Args:
            state: Current state vector
            deterministic: If True, select best action. If False, sample from policy.
            return_probs: If True, return action probabilities and entropy for logging.
            
        Returns:
            Tuple of (action, action_log_prob, optional_info_dict)
            If return_probs=True, optional_info_dict contains:
            - action_probs: List of probabilities for each action
            - entropy: Policy entropy value
        """
        # Ensure state is properly typed float64 array
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float64)
        else:
            state = state.astype(np.float64)
        
        # Handle any NaN or inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            if self.action_mode == "discrete":
                action_logits = self.policy_net(state_tensor)[0]
                
                action_dist = torch.distributions.Categorical(logits=action_logits)
                
                if deterministic:
                    action = torch.argmax(action_logits, dim=1).item()
                    action_log_prob = torch.log_softmax(action_logits, dim=1)[0, action].item()
                else:
                    action_tensor = action_dist.sample()
                    action = action_tensor.item()
                    action_log_prob = action_dist.log_prob(action_tensor).item()
                
                # Extract probabilities and entropy if requested
                optional_info = None
                if return_probs:
                    probs = torch.softmax(action_logits, dim=1)[0].cpu().numpy()
                    entropy = action_dist.entropy().item()
                    optional_info = {
                        "action_probs": probs.tolist(),
                        "entropy": entropy
                    }
                
                return action, action_log_prob, optional_info
            
            else:  # continuous
                mean, std, _ = self.policy_net(state_tensor)
                action_dist = torch.distributions.Normal(mean, std)
                
                if deterministic:
                    action = mean[0].item()
                    action = np.clip(action, -1.0, 1.0)
                    action_log_prob = action_dist.log_prob(mean[0]).item()
                else:
                    action_tensor = action_dist.sample()
                    action = action_tensor[0].item()
                    action = np.clip(action, -1.0, 1.0)
                    # Convert back to tensor for log_prob, then clip tensor
                    action_tensor_clipped = torch.clamp(action_tensor[0], -1.0, 1.0)
                    action_log_prob = action_dist.log_prob(action_tensor_clipped).item()
                
                # Extract entropy if requested
                optional_info = None
                if return_probs:
                    entropy = action_dist.entropy().item()
                    optional_info = {
                        "action_mean": mean[0].item(),
                        "action_std": std[0].item(),
                        "entropy": entropy
                    }
                
                return action, action_log_prob, optional_info
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_log_prob: float
    ):
        """
        Store experience in buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            action_log_prob: Log probability of action
        """
        self.experience_buffer.add(
            state, action, reward, next_state, done, action_log_prob
        )
        self.step_count += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.experience_buffer) < self.batch_size:
            return {"error": "Insufficient experiences for update"}
        
        # Get experiences
        experiences = self.experience_buffer.get_all()
        states = np.array([e['state'] for e in experiences])
        actions = np.array([e['action'] for e in experiences])
        rewards = np.array([e['reward'] for e in experiences])
        next_states = np.array([e['next_state'] for e in experiences])
        dones = np.array([e['done'] for e in experiences])
        old_log_probs = np.array([e['action_log_prob'] for e in experiences])
        
        # Compute value estimates
        states_tensor = torch.FloatTensor(states).to(self.device)
        values = self.value_net(states_tensor).squeeze().cpu().numpy()
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        next_values = self.value_net(next_states_tensor).squeeze().cpu().numpy()
        
        # Compute advantages - use GAE if enabled, otherwise simple TD
        if self.use_gae:
            # Use GAE (Generalized Advantage Estimation)
            # Pad next_values for last step
            if len(next_values) > len(values):
                next_values = next_values[:len(values)]
            elif len(next_values) < len(values):
                # Last state: use current value estimate or bootstrap
                next_values = np.append(next_values, values[-1])
            
            # Prepare lists for GAE computation
            values_list = values.tolist()
            next_values_list = next_values.tolist()[:len(rewards)]
            rewards_list = rewards.tolist()
            dones_list = dones.tolist()
            
            # Compute GAE
            advantages_list = []
            returns_list = []
            gae = 0
            
            for i in reversed(range(len(rewards_list))):
                if dones_list[i]:
                    gae = 0
                    next_val = 0
                else:
                    next_val = next_values_list[i] if i < len(next_values_list) else values_list[i]
                
                delta = rewards_list[i] + self.gamma * next_val - values_list[i]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages_list.insert(0, gae)
                returns_list.insert(0, gae + values_list[i])
            
            advantages = np.array(advantages_list)
            returns = np.array(returns_list)
        else:
            # Simple TD advantages (original method)
            target_values = rewards + self.gamma * next_values * (1 - dones)
            advantages = target_values - values
            returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions) if self.action_mode == "discrete" else torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        self.policy_net.train()
        self.value_net.train()
        
        # Store old values for value clipping
        old_values_tensor = torch.FloatTensor(values).to(self.device)
        
        previous_policy_loss = None
        kl_divergences = []
        
        # Initialize gradient monitoring variables
        policy_grad_norm = 0.0
        value_grad_norm = 0.0
        total_grad_norm = 0.0
        
        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(states)).to(self.device)
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # Compute new action probabilities
                if self.action_mode == "discrete":
                    action_logits = self.policy_net(batch_states)[0]
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    new_log_probs = action_dist.log_prob(batch_actions)
                    entropy = action_dist.entropy().mean()
                else:
                    mean, std, _ = self.policy_net(batch_states)
                    action_dist = torch.distributions.Normal(mean, std)
                    new_log_probs = action_dist.log_prob(batch_actions).sum(dim=1)
                    entropy = action_dist.entropy().sum(dim=1).mean()
                
                # Compute value estimate
                values = self.value_net(batch_states).squeeze()
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with optional clipping
                if self.value_clipping:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon
                    )
                    value_loss = torch.max(
                        (values - batch_returns).pow(2),
                        (value_pred_clipped - batch_returns).pow(2)
                    ).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Track KL divergence for adaptive clipping
                kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                kl_divergences.append(kl_div)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Early stopping check
                if self.early_stopping and previous_policy_loss is not None:
                    if policy_loss.item() > previous_policy_loss * 1.15:  # 15% increase
                        break  # Stop early if loss increased significantly
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient monitoring and clipping
                policy_grad_norm = 0.0
                value_grad_norm = 0.0
                
                if self.gradient_monitoring:
                    # Compute gradient norms
                    for p in self.policy_net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            policy_grad_norm += param_norm.item() ** 2
                    policy_grad_norm = policy_grad_norm ** (1. / 2)
                    
                    for p in self.value_net.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            value_grad_norm += param_norm.item() ** 2
                    value_grad_norm = value_grad_norm ** (1. / 2)
                    
                    total_grad_norm = (policy_grad_norm ** 2 + value_grad_norm ** 2) ** 0.5
                    
                    # Skip update if gradient explosion detected
                    if total_grad_norm > self.max_grad_norm:
                        # Don't step, but continue to next batch
                        continue
                
                # Clip gradients (always done, monitoring is separate)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                previous_policy_loss = policy_loss.item()
        
        # Adaptive clipping (if enabled)
        if self.adaptive_clip and len(kl_divergences) > 0:
            mean_kl = np.mean(kl_divergences)
            if mean_kl > self.clip_epsilon * 1.5:
                self.clip_epsilon = min(self.clip_epsilon * 1.1, 0.3)
            elif mean_kl < self.clip_epsilon * 0.5:
                self.clip_epsilon = max(self.clip_epsilon * 0.95, 0.1)
        
        # Update learning rate scheduler (if using plateau scheduler)
        if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
            avg_value_loss = total_value_loss / max((len(states) // self.batch_size) * self.update_epochs, 1)
            self.scheduler.step(avg_value_loss)
        elif self.scheduler is not None and isinstance(self.scheduler, CosineAnnealingLR):
            self.scheduler.step()
        
        # Clear buffer
        self.experience_buffer.clear()
        
        # Compute metrics
        n_batches = (len(states) // self.batch_size) * self.update_epochs
        if n_batches == 0:
            n_batches = 1
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "mean_advantage": float(advantages.mean()),
            "mean_return": float(returns.mean()),
            "learning_rate": current_lr,
            "clip_epsilon": self.clip_epsilon,
        }
        
        # Add gradient metrics if monitoring
        if self.gradient_monitoring:
            metrics["policy_grad_norm"] = policy_grad_norm
            metrics["value_grad_norm"] = value_grad_norm
            metrics["total_grad_norm"] = total_grad_norm
        
        # Add KL divergence metric
        if len(kl_divergences) > 0:
            metrics["mean_kl_divergence"] = float(np.mean(kl_divergences))
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent to disk."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)

