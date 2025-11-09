"""
Multi-algorithm ML model for cryptocurrency price prediction.
Supports Random Forest, XGBoost, and LightGBM.
Implements walk-forward validation and model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Initialize logger early (before optional imports that might use it)
logger = logging.getLogger(__name__)

# Optional imports for advanced algorithms
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Optional imports for hyperparameter tuning and feature importance
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Hyperparameter tuning disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Advanced feature importance analysis disabled.")

from ml_models.features import MLFeatureEngineer
from data.db import get_db_manager


class CryptoPredictionModel:
    """Multi-algorithm ML model for cryptocurrency price prediction."""
    
    def __init__(self, 
                 algorithm: str = 'random_forest',
                 model_type: str = 'classifier', 
                 n_estimators: int = 100, 
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the prediction model with algorithm selection.
        
        Args:
            algorithm: 'random_forest', 'xgboost', or 'lightgbm'
            model_type: 'classifier' or 'regressor'
            n_estimators: Number of trees/estimators
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (for XGBoost/LightGBM)
            random_state: Random state for reproducibility
            **kwargs: Additional algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Validate algorithm availability
        if self.algorithm == 'xgboost' and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to Random Forest")
            self.algorithm = 'random_forest'
        elif self.algorithm == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to Random Forest")
            self.algorithm = 'random_forest'
        
        # Initialize model based on algorithm
        if self.algorithm == 'random_forest':
            self.model = self._create_random_forest(model_type, **kwargs)
        elif self.algorithm == 'xgboost':
            self.model = self._create_xgboost(model_type, **kwargs)
        elif self.algorithm == 'lightgbm':
            self.model = self._create_lightgbm(model_type, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose: 'random_forest', 'xgboost', 'lightgbm'")
        
        self.feature_engineer = MLFeatureEngineer()
        self.db_manager = get_db_manager()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Only log if not suppressing (suppress_init_log is used when loading existing model)
        if not kwargs.get('suppress_init_log', False):
            logger.info(f"Initialized {self.algorithm} {model_type} with {n_estimators} estimators")
    
    def _create_random_forest(self, model_type: str, **kwargs) -> Any:
        """Create Random Forest model."""
        common_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': kwargs.get('min_samples_split', 20),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 10),
            'max_features': kwargs.get('max_features', 'sqrt'),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if model_type == 'classifier':
            common_params['class_weight'] = kwargs.get('class_weight', 'balanced')
            return RandomForestClassifier(**common_params)
        else:
            return RandomForestRegressor(**common_params)
    
    def _create_xgboost(self, model_type: str, **kwargs) -> Any:
        """Create XGBoost model."""
        common_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 0.1),
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        if model_type == 'classifier':
            common_params['eval_metric'] = 'logloss'
            return XGBClassifier(**common_params)
        else:
            common_params['eval_metric'] = 'rmse'
            return XGBRegressor(**common_params)
    
    def _create_lightgbm(self, model_type: str, **kwargs) -> Any:
        """Create LightGBM model."""
        common_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.1),
            'reg_lambda': kwargs.get('reg_lambda', 0.1),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if model_type == 'classifier':
            common_params['objective'] = 'binary'
            common_params['metric'] = 'binary_logloss'
            common_params['is_unbalance'] = True
            return LGBMClassifier(**common_params)
        else:
            common_params['objective'] = 'regression'
            common_params['metric'] = 'rmse'
            return LGBMRegressor(**common_params)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close', 
                    prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with OHLCV data
            target_column: Target column for prediction
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            Tuple of (features, target)
        """
        # Create all features
        featured_data = self.feature_engineer.create_all_features(df)
        
        if len(featured_data) == 0:
            raise ValueError("No data available after feature engineering")
        
        # Create target variable
        if self.model_type == 'classifier':
            # Binary classification: price goes up (1) or down (0)
            target = (featured_data[target_column].shift(-prediction_horizon) > 
                     featured_data[target_column]).astype(int)
        else:
            # Regression: predict actual price change
            target = featured_data[target_column].shift(-prediction_horizon) - featured_data[target_column]
        
        # Remove rows with NaN target
        valid_idx = ~target.isna()
        # Drop timestamp column if it exists (might be 'timestamp' or 'ts')
        columns_to_drop = [target_column]
        if 'timestamp' in featured_data.columns:
            columns_to_drop.append('timestamp')
        elif 'ts' in featured_data.columns:
            columns_to_drop.append('ts')
        
        X = featured_data[valid_idx].drop(columns=columns_to_drop)
        y = target[valid_idx]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Prepared data: {len(X)} samples, {len(self.feature_names)} features")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results
        """
        # Split data chronologically
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        if self.model_type == 'classifier':
            train_metrics = self._calculate_classification_metrics(y_train, train_pred)
            val_metrics = self._calculate_classification_metrics(y_val, val_pred)
        else:
            train_metrics = self._calculate_regression_metrics(y_train, train_pred)
            val_metrics = self._calculate_regression_metrics(y_val, val_pred)
        
        self.is_trained = True
        
        # Store feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'model_type': self.model_type
        }
        
        logger.info(f"Model training completed. Validation accuracy: {val_metrics.get('accuracy', val_metrics.get('r2_score', 'N/A'))}")
        
        return results
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50, 
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna Bayesian optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of optimization trials
            validation_split: Fraction of data to use for validation
            
        Returns:
            Best hyperparameters and optimization results
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping hyperparameter tuning.")
            return {}
        
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        # Split data chronologically
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        def objective(trial):
            """Objective function for Optuna."""
            if self.algorithm == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                if self.model_type == 'classifier':
                    model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **params)
                else:
                    model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **params)
            
            elif self.algorithm == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                if self.model_type == 'classifier':
                    model = XGBClassifier(random_state=self.random_state, n_jobs=-1, verbosity=0, **params)
                else:
                    model = XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0, **params)
            
            elif self.algorithm == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                if self.model_type == 'classifier':
                    params['is_unbalance'] = True
                    model = LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbosity=-1, **params)
                else:
                    model = LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbosity=-1, **params)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Train and evaluate
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            
            # Calculate score (maximize)
            if self.model_type == 'classifier':
                score = accuracy_score(y_val, val_pred)
            else:
                score = -mean_squared_error(y_val, val_pred)  # Negative because Optuna maximizes
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Update model with best parameters
        if self.algorithm == 'random_forest':
            self.model = self._create_random_forest(self.model_type, **best_params)
        elif self.algorithm == 'xgboost':
            self.model = self._create_xgboost(self.model_type, **best_params)
        elif self.algorithm == 'lightgbm':
            self.model = self._create_lightgbm(self.model_type, **best_params)
        
        logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials,
            'study': study
        }
    
    def filter_features_by_importance(self, feature_importance: Dict[str, float], 
                                     top_k: Optional[int] = None, 
                                     min_importance: float = 0.001) -> List[str]:
        """
        Filter features based on importance scores.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_k: Number of top features to keep (if None, uses min_importance)
            min_importance: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            # Select top K features
            selected = [feat for feat, imp in sorted_features[:top_k] if imp >= min_importance]
        else:
            # Select features above threshold
            selected = [feat for feat, imp in sorted_features if imp >= min_importance]
        
        logger.info(f"Selected {len(selected)} features out of {len(feature_importance)} "
                   f"(top_k={top_k}, min_importance={min_importance})")
        
        return selected
    
    def get_shap_importance(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                           sample_size: int = 100) -> Dict[str, float]:
        """
        Get SHAP-based feature importance (more accurate than tree-based importance).
        
        Args:
            X: Feature matrix
            y: Target variable (optional, for explanation)
            sample_size: Number of samples to use for SHAP computation
            
        Returns:
            Dictionary mapping feature names to SHAP importance scores
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Using tree-based feature importance.")
            if self.is_trained:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return {}
        
        if not self.is_trained:
            logger.warning("Model not trained. Cannot compute SHAP importance.")
            return {}
        
        logger.info("Computing SHAP feature importance...")
        
        # Sample data for faster computation
        if len(X) > sample_size:
            X_sample = X.sample(n=min(sample_size, len(X)), random_state=self.random_state)
        else:
            X_sample = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_sample)
        
        # Create SHAP explainer
        if self.algorithm in ['random_forest', 'xgboost', 'lightgbm']:
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(self.model.predict, X_scaled[:50])  # Use subset for kernel
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_scaled)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for binary classification
        
        # Calculate mean absolute SHAP values as importance
        if len(shap_values.shape) > 1:
            shap_importance = np.abs(shap_values).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values)
        
        # Map to feature names
        feature_shap_importance = dict(zip(self.feature_names, shap_importance))
        
        logger.info(f"SHAP importance computed for {len(feature_shap_importance)} features")
        
        return feature_shap_importance
    
    def walk_forward_validation(self, df: pd.DataFrame, window_size: int = 500, 
                               step_size: int = 50) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Size of training window
            step_size: Step size for moving window
            
        Returns:
            Walk-forward validation results
        """
        logger.info(f"Starting walk-forward validation with window size {window_size}")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        if len(X) < window_size + step_size:
            raise ValueError("Insufficient data for walk-forward validation")
        
        # Initialize results
        predictions = []
        actuals = []
        feature_importances = []
        
        # Walk-forward validation
        for start_idx in range(0, len(X) - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Training data
            X_train = X.iloc[start_idx:end_idx]
            y_train = y.iloc[start_idx:end_idx]
            
            # Test data (next step_size samples)
            test_end = min(end_idx + step_size, len(X))
            X_test = X.iloc[end_idx:test_end]
            y_test = y.iloc[end_idx:test_end]
            
            if len(X_test) == 0:
                break
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with same algorithm and hyperparameters
            if self.algorithm == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                ) if self.model_type == 'classifier' else RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif self.algorithm == 'xgboost':
                model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                ) if self.model_type == 'classifier' else XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                )
            elif self.algorithm == 'lightgbm':
                model = LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1
                ) if self.model_type == 'classifier' else LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            test_pred = model.predict(X_test_scaled)
            
            # Store results
            predictions.extend(test_pred)
            actuals.extend(y_test.values)
            feature_importances.append(model.feature_importances_)
            
            logger.info(f"Walk-forward step {start_idx//step_size + 1}: "
                       f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Calculate overall metrics
        if self.model_type == 'classifier':
            overall_metrics = self._calculate_classification_metrics(
                pd.Series(actuals), pd.Series(predictions)
            )
        else:
            overall_metrics = self._calculate_regression_metrics(
                pd.Series(actuals), pd.Series(predictions)
            )
        
        # Calculate average feature importance
        avg_feature_importance = np.mean(feature_importances, axis=0)
        feature_importance_dict = dict(zip(self.feature_names, avg_feature_importance))
        
        results = {
            'overall_metrics': overall_metrics,
            'feature_importance': feature_importance_dict,
            'predictions': predictions,
            'actuals': actuals,
            'num_windows': len(feature_importances)
        }
        
        logger.info(f"Walk-forward validation completed. Overall accuracy: {overall_metrics.get('accuracy', overall_metrics.get('r2_score', 'N/A'))}")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        featured_data = self.feature_engineer.create_all_features(df)
        
        if len(featured_data) == 0:
            return {'error': 'No data available for prediction'}
        
        # Get latest features
        latest_features = featured_data.iloc[-1:].drop(columns=['timestamp', 'close'])
        numeric_columns = latest_features.select_dtypes(include=[np.number]).columns
        latest_features = latest_features[numeric_columns]
        
        # Handle missing features
        for feature in self.feature_names:
            if feature not in latest_features.columns:
                latest_features[feature] = 0
        
        # Select only trained features
        X = latest_features[self.feature_names]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Get prediction probability (for classifier)
        if self.model_type == 'classifier':
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0  # For regression, we don't have probability
        
        # Store prediction in database
        self.db_manager.insert_ml_prediction(
            symbol='BTC/USD',  # Default symbol
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=f'{self.algorithm}_{self.model_type}',
            prediction=float(prediction),
            confidence=float(confidence),
            features=X.iloc[0].to_dict()
        )
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_type': self.model_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'feature_engineer': self.feature_engineer,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath} (algorithm: {self.algorithm})")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Restore algorithm and model structure
        self.algorithm = model_data.get('algorithm', 'random_forest')
        self.model_type = model_data['model_type']
        self.n_estimators = model_data.get('n_estimators', 100)
        self.max_depth = model_data.get('max_depth', 6)
        self.learning_rate = model_data.get('learning_rate', 0.1)
        self.random_state = model_data.get('random_state', 42)
        
        # Load model and other components
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_engineer = model_data.get('feature_engineer', MLFeatureEngineer())
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath} (algorithm: {self.algorithm})")
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_count': len(self.feature_names),
            'is_trained': self.is_trained,
            'feature_names': self.feature_names[:10]
        }
