"""Centralized logging service for the trading bot."""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class TradingLogger:
    """Centralized logger for trading bot operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger.
        
        Args:
            config: Optional configuration dict with logging settings
        """
        if config is None:
            logging_config = {}
        else:
            logging_config = config.get("logging", {})
        
        self.log_level = logging_config.get("level", "INFO")
        self.log_dir = Path(logging_config.get("log_dir", "logs"))
        self.log_trades = logging_config.get("log_trades", True)
        self.log_training = logging_config.get("log_training", True)
        self.log_errors = logging_config.get("log_errors", True)
        
        # Create log directories
        self.trades_log_dir = self.log_dir / "trades"
        self.training_log_dir = self.log_dir / "training"
        self.errors_log_dir = self.log_dir / "errors"
        
        for log_dir in [self.trades_log_dir, self.training_log_dir, self.errors_log_dir]:
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different loggers for different purposes."""
        # Main logger
        self.logger = logging.getLogger("trading_bot")
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # Common formatter with module name
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Trade logger
        if self.log_trades:
            trade_handler = logging.FileHandler(
                self.trades_log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
            )
            trade_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            self.trade_logger = logging.getLogger("trades")
            self.trade_logger.addHandler(trade_handler)
            self.trade_logger.setLevel(logging.INFO)
            self.trade_logger.propagate = False
        else:
            self.trade_logger = None
        
        # Training logger
        if self.log_training:
            training_handler = logging.FileHandler(
                self.training_log_dir / f"training_{datetime.now().strftime('%Y%m%d')}.log"
            )
            training_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            self.training_logger = logging.getLogger("training")
            self.training_logger.addHandler(training_handler)
            self.training_logger.setLevel(logging.INFO)
            self.training_logger.propagate = False
        else:
            self.training_logger = None
        
        # Error logger
        if self.log_errors:
            error_handler = logging.FileHandler(
                self.errors_log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
            )
            error_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            self.error_logger = logging.getLogger("errors")
            self.error_logger.addHandler(error_handler)
            self.error_logger.setLevel(logging.ERROR)
            self.error_logger.propagate = False
        else:
            self.error_logger = None
        
        # Console handler with module name
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(log_format, datefmt=date_format)
        )
        self.logger.addHandler(console_handler)
    
    def get_module_logger(self, module_path: str) -> logging.Logger:
        """
        Get a logger for a specific module/component.
        
        Args:
            module_path: Module path like 'data.data_feeder', 'trading_engine.continuous_service', 'rl.ppo_agent'
            
        Returns:
            Logger instance configured with same handlers
        """
        logger = logging.getLogger(module_path)
        
        # Don't add handlers if already configured
        if logger.handlers:
            return logger
        
        # Use same formatter as main logger
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(log_format, datefmt=date_format)
        )
        logger.addHandler(console_handler)
        
        logger.setLevel(self.logger.level)
        logger.propagate = False
        
        return logger
    
    def log_trade(
        self,
        step: int,
        action: int,
        execution_result: Dict[str, Any],
        portfolio_value: float,
        reward: Optional[float] = None
    ):
        """Log a trade execution."""
        if self.trade_logger:
            action_names = {0: "Buy", 1: "Hold", 2: "Sell"}
            action_name = action_names.get(action, f"Action_{action}")
            reward_str = f", Reward: {reward:.6f}" if reward is not None else ""
            self.trade_logger.info(
                f"Step {step} - Action: {action_name} ({action}), "
                f"Success: {execution_result.get('success', False)}, "
                f"Qty: {execution_result.get('executed_quantity', 0):.6f}, "
                f"Price: ${execution_result.get('execution_price', 0):.2f}, "
                f"Portfolio: ${portfolio_value:,.2f}{reward_str}"
            )
    
    def log_training(
        self,
        step: int,
        metrics: Dict[str, Any],
        episode_reward: Optional[float] = None
    ):
        """Log training metrics."""
        if self.training_logger:
            log_msg = f"Step {step} - "
            
            # PPO metrics
            if metrics.get('policy_loss') is not None:
                log_msg += f"Policy Loss: {metrics.get('policy_loss', 0):.6f}, "
                log_msg += f"Value Loss: {metrics.get('value_loss', 0):.6f}, "
                log_msg += f"Entropy: {metrics.get('entropy', 0):.6f}"
                
                # Additional PPO metrics if available
                if 'learning_rate' in metrics:
                    log_msg += f", LR: {metrics['learning_rate']:.2e}"
                if 'mean_advantage' in metrics:
                    log_msg += f", Mean Adv: {metrics['mean_advantage']:.4f}"
                if 'mean_kl_divergence' in metrics:
                    log_msg += f", KL Div: {metrics['mean_kl_divergence']:.4f}"
            
            # ML metrics
            if metrics.get('ml_validation_accuracy') is not None:
                log_msg += f"ML Validation Accuracy: {metrics.get('ml_validation_accuracy', 0):.4f}"
                if metrics.get('ml_train_accuracy') is not None:
                    log_msg += f", ML Train Accuracy: {metrics.get('ml_train_accuracy', 0):.4f}"
            
            if episode_reward is not None:
                log_msg += f", Episode Reward: {episode_reward:.2f}"
            
            self.training_logger.info(log_msg)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error."""
        if self.error_logger:
            self.error_logger.error(
                f"{context} - {type(error).__name__}: {str(error)}"
            )
        self.logger.error(f"Error: {context} - {str(error)}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        if self.error_logger:
            self.error_logger.error(message)

