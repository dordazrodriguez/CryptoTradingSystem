"""
Model evaluation metrics and analysis for cryptocurrency trading models.
Implements precision, recall, F1, confusion matrix, and trading performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for trading models."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results = {}
    
    def evaluate_classification_model(self, y_true: pd.Series, y_pred: pd.Series, 
                                    y_prob: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        metrics['per_class_metrics'] = self._calculate_per_class_metrics(y_true, y_pred)
        
        # ROC AUC if probabilities available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['roc_curve'] = roc_curve(y_true, y_prob)
                metrics['precision_recall_curve'] = precision_recall_curve(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        return metrics
    
    def evaluate_regression_model(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Regression metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': self._calculate_mape(y_true, y_pred),
            'smape': self._calculate_smape(y_true, y_pred)
        }
        
        # Directional accuracy (for price prediction)
        if len(y_true) > 1:
            true_direction = (y_true > 0).astype(int)
            pred_direction = (y_pred > 0).astype(int)
            metrics['directional_accuracy'] = accuracy_score(true_direction, pred_direction)
        
        return metrics
    
    def evaluate_trading_performance(self, predictions: List[float], actuals: List[float], 
                                   prices: List[float], transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Evaluate trading performance based on predictions.
        
        Args:
            predictions: Model predictions
            actuals: Actual price changes
            prices: Price levels
            transaction_cost: Transaction cost per trade
            
        Returns:
            Trading performance metrics
        """
        if len(predictions) != len(actuals) or len(predictions) != len(prices):
            raise ValueError("All input lists must have the same length")
        
        # Convert to pandas Series for easier manipulation
        pred_series = pd.Series(predictions)
        actual_series = pd.Series(actuals)
        price_series = pd.Series(prices)
        
        # Generate trading signals
        signals = self._generate_trading_signals(pred_series)
        
        # Calculate returns
        returns = actual_series
        signal_returns = signals * returns - abs(signals) * transaction_cost
        
        # Calculate performance metrics
        total_return = signal_returns.sum()
        cumulative_return = (1 + signal_returns).cumprod()
        
        # Sharpe ratio (simplified)
        sharpe_ratio = signal_returns.mean() / signal_returns.std() * np.sqrt(252) if signal_returns.std() > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(cumulative_return)
        
        # Win rate
        winning_trades = (signal_returns > 0).sum()
        total_trades = (signals != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = signal_returns[signal_returns > 0].sum()
        gross_loss = abs(signal_returns[signal_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade = signal_returns.mean()
        
        # Trade statistics
        trade_stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor
        }
        
        # Performance metrics
        performance_metrics = {
            'total_return': total_return,
            'cumulative_return': cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 1.0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': signal_returns.std() * np.sqrt(252),
            'trade_stats': trade_stats
        }
        
        return performance_metrics
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_results: Dictionary of model_name -> results
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Extract metrics for comparison
        for model_name, results in model_results.items():
            if 'classification_metrics' in results:
                metrics = results['classification_metrics']
                comparison[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0)
                }
            elif 'regression_metrics' in results:
                metrics = results['regression_metrics']
                comparison[model_name] = {
                    'r2_score': metrics.get('r2_score', 0),
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0)
                }
        
        # Find best model for each metric
        best_models = {}
        if comparison:
            metrics_to_compare = list(comparison[list(comparison.keys())[0]].keys())
            
            for metric in metrics_to_compare:
                best_model = max(comparison.keys(), 
                               key=lambda x: comparison[x].get(metric, -float('inf')))
                best_models[metric] = best_model
        
        return {
            'model_comparison': comparison,
            'best_models': best_models
        }
    
    def generate_evaluation_report(self, model_name: str, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        report = f"\n{'='*60}\n"
        report += f"MODEL EVALUATION REPORT: {model_name}\n"
        report += f"{'='*60}\n"
        report += f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Classification metrics
        if 'classification_metrics' in results:
            metrics = results['classification_metrics']
            report += "CLASSIFICATION METRICS:\n"
            report += f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
            report += f"  Precision: {metrics.get('precision', 0):.4f}\n"
            report += f"  Recall:    {metrics.get('recall', 0):.4f}\n"
            report += f"  F1-Score:  {metrics.get('f1_score', 0):.4f}\n"
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                report += f"  ROC AUC:   {metrics['roc_auc']:.4f}\n"
            report += "\n"
        
        # Regression metrics
        if 'regression_metrics' in results:
            metrics = results['regression_metrics']
            report += "REGRESSION METRICS:\n"
            report += f"  R² Score:  {metrics.get('r2_score', 0):.4f}\n"
            report += f"  RMSE:      {metrics.get('rmse', 0):.4f}\n"
            report += f"  MAE:       {metrics.get('mae', 0):.4f}\n"
            report += f"  MAPE:      {metrics.get('mape', 0):.4f}%\n"
            report += "\n"
        
        # Trading performance
        if 'trading_performance' in results:
            perf = results['trading_performance']
            report += "TRADING PERFORMANCE:\n"
            report += f"  Total Return:     {perf.get('total_return', 0):.4f}\n"
            report += f"  Sharpe Ratio:      {perf.get('sharpe_ratio', 0):.4f}\n"
            report += f"  Max Drawdown:      {perf.get('max_drawdown', 0):.4f}\n"
            report += f"  Volatility:        {perf.get('volatility', 0):.4f}\n"
            
            if 'trade_stats' in perf:
                stats = perf['trade_stats']
                report += f"  Total Trades:     {stats.get('total_trades', 0)}\n"
                report += f"  Win Rate:         {stats.get('win_rate', 0):.4f}\n"
                report += f"  Profit Factor:    {stats.get('profit_factor', 0):.4f}\n"
            report += "\n"
        
        # Feature importance
        if 'feature_importance' in results:
            report += "TOP 10 FEATURES BY IMPORTANCE:\n"
            importance = results['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance_score in sorted_features:
                report += f"  {feature}: {importance_score:.4f}\n"
            report += "\n"
        
        report += f"{'='*60}\n"
        
        return report
    
    def _calculate_per_class_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        classes = sorted(y_true.unique())
        per_class = {}
        
        for cls in classes:
            precision = precision_score(y_true, y_pred, labels=[cls], average='micro')
            recall = recall_score(y_true, y_pred, labels=[cls], average='micro')
            f1 = f1_score(y_true, y_pred, labels=[cls], average='micro')
            
            per_class[str(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return per_class
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _calculate_smape(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    def _generate_trading_signals(self, predictions: pd.Series, threshold: float = 0.5) -> pd.Series:
        """Generate trading signals from predictions."""
        # Simple threshold-based signals
        signals = pd.Series(0, index=predictions.index)
        signals[predictions > threshold] = 1  # Buy
        signals[predictions < -threshold] = -1  # Sell
        
        return signals
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None, 
                            title: str = "Confusion Matrix"):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: pd.Series, y_prob: pd.Series, title: str = "ROC Curve"):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: pd.Series, y_prob: pd.Series, 
                                  title: str = "Precision-Recall Curve"):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_trading_performance(self, cumulative_returns: pd.Series, 
                               title: str = "Trading Performance"):
        """Plot cumulative returns."""
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the model evaluator
    evaluator = ModelEvaluator()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Classification test
    y_true_class = pd.Series(np.random.randint(0, 2, n_samples))
    y_pred_class = pd.Series(np.random.randint(0, 2, n_samples))
    y_prob_class = pd.Series(np.random.rand(n_samples))
    
    class_metrics = evaluator.evaluate_classification_model(y_true_class, y_pred_class, y_prob_class)
    print("Classification metrics:", class_metrics['accuracy'])
    
    # Regression test
    y_true_reg = pd.Series(np.random.randn(n_samples))
    y_pred_reg = pd.Series(np.random.randn(n_samples))
    
    reg_metrics = evaluator.evaluate_regression_model(y_true_reg, y_pred_reg)
    print("Regression metrics:", reg_metrics['r2_score'])
    
    # Trading performance test
    predictions = np.random.randn(n_samples)
    actuals = np.random.randn(n_samples)
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    
    trading_perf = evaluator.evaluate_trading_performance(predictions, actuals, prices)
    print("Trading performance:", trading_perf['total_return'])
    
    print("Model evaluator test completed!")
