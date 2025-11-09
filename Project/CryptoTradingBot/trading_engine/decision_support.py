"""
Decision support system combining ML predictions with technical indicators.
Generates trade signals, risk assessments, and portfolio recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timezone
from enum import Enum

from ml_models.predictor import CryptoPredictionModel
from ml_models.evaluation import ModelEvaluator
from trading_engine.indicators import TechnicalIndicators
from trading_engine.portfolio import PortfolioManager
from trading_engine.risk_manager import RiskManager
from data.db import get_db_manager

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength enumeration."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class TradeSignal(Enum):
    """Trade signal enumeration."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class DecisionSupportSystem:
    """Combines ML predictions with technical indicators for trading decisions."""
    
    def __init__(self, ml_model: CryptoPredictionModel = None, 
                 risk_manager: RiskManager = None):
        """
        Initialize decision support system.
        
        Args:
            ml_model: Trained ML model for predictions
            risk_manager: Risk management system
        """
        self.ml_model = ml_model
        self.risk_manager = risk_manager or RiskManager()
        self.indicators = TechnicalIndicators()
        self.evaluator = ModelEvaluator()
        self.db_manager = get_db_manager()
        
        # Signal weights (can be adjusted based on backtesting)
        self.signal_weights = {
            'ml_prediction': 0.4,
            'technical_indicators': 0.3,
            'risk_assessment': 0.2,
            'market_regime': 0.1
        }
        
        # Thresholds for signal generation
        self.signal_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.2,
            'strong_sell': 0.0
        }
        
        logger.info("Decision support system initialized")
    
    def analyze_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using technical indicators and ML model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Comprehensive market analysis
        """
        if len(df) < 50:  # Need minimum data for indicators
            return {'error': 'Insufficient data for analysis'}
        
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points': len(df),
            'symbol': df.get('symbol', 'BTC/USD').iloc[0] if 'symbol' in df.columns else 'BTC/USD'
        }
        
        # Technical indicators analysis
        try:
            indicators_data = self.indicators.calculate_all_indicators(df)
            technical_signals = self.indicators.get_trading_signals(indicators_data)
            
            # Get latest signals
            latest_signals = technical_signals.iloc[-1]
            
            analysis['technical_indicators'] = {
                'rsi': latest_signals.get('rsi', 0),
                'macd': latest_signals.get('macd', 0),
                'macd_signal': latest_signals.get('macd_signal', 0),
                'sma_20': latest_signals.get('sma_20', 0),
                'sma_50': latest_signals.get('sma_50', 0),
                'bb_position': latest_signals.get('bb_position', 0),
                'bb_width': latest_signals.get('bb_width', 0),
                'stoch_k': latest_signals.get('stoch_k', 0),
                'stoch_d': latest_signals.get('stoch_d', 0)
            }
            
            analysis['technical_signals'] = {
                'rsi_oversold': latest_signals.get('rsi_oversold', 0),
                'rsi_overbought': latest_signals.get('rsi_overbought', 0),
                'macd_bullish': latest_signals.get('macd_bullish', 0),
                'macd_bearish': latest_signals.get('macd_bearish', 0),
                'sma_crossover_bull': latest_signals.get('sma_crossover_bull', 0),
                'sma_crossover_bear': latest_signals.get('sma_crossover_bear', 0),
                'bb_squeeze': latest_signals.get('bb_squeeze', 0),
                'bb_breakout_up': latest_signals.get('bb_breakout_up', 0),
                'bb_breakout_down': latest_signals.get('bb_breakout_down', 0),
                'bullish_signals': latest_signals.get('bullish_signals', 0),
                'bearish_signals': latest_signals.get('bearish_signals', 0),
                'signal_strength': latest_signals.get('signal_strength', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            analysis['technical_indicators'] = {}
            analysis['technical_signals'] = {}
        
        # ML prediction analysis
        if self.ml_model and self.ml_model.is_trained:
            try:
                ml_prediction = self.ml_model.predict(df)
                analysis['ml_prediction'] = ml_prediction
            except Exception as e:
                logger.error(f"Error getting ML prediction: {e}")
                analysis['ml_prediction'] = {'error': str(e)}
        else:
            analysis['ml_prediction'] = {'error': 'ML model not trained'}
        
        # Market regime analysis
        analysis['market_regime'] = self._analyze_market_regime(df)
        
        return analysis
    
    def generate_trade_signal(self, market_analysis: Dict[str, Any], 
                            current_price: float) -> Dict[str, Any]:
        """
        Generate trade signal based on comprehensive analysis.
        
        Args:
            market_analysis: Market analysis results
            current_price: Current market price
            
        Returns:
            Trade signal with confidence and reasoning
        """
        signal_score = 0.0
        reasoning = []
        
        # ML prediction contribution
        if 'ml_prediction' in market_analysis and 'error' not in market_analysis['ml_prediction']:
            ml_pred = market_analysis['ml_prediction']
            ml_confidence = ml_pred.get('confidence', 0.5)
            ml_prediction = ml_pred.get('prediction', 0.5)
            
            # Convert ML prediction to signal score
            if self.ml_model.model_type == 'classifier':
                ml_signal_score = (ml_prediction - 0.5) * 2 * ml_confidence  # -1 to 1
            else:
                # For regression, normalize the prediction
                ml_signal_score = np.tanh(ml_prediction) * ml_confidence
            
            signal_score += ml_signal_score * self.signal_weights['ml_prediction']
            reasoning.append(f"ML prediction: {ml_signal_score:.3f} (confidence: {ml_confidence:.3f})")
        
        # Technical indicators contribution
        if 'technical_signals' in market_analysis:
            tech_signals = market_analysis['technical_signals']
            
            # Calculate technical signal score
            bullish_signals = tech_signals.get('bullish_signals', 0)
            bearish_signals = tech_signals.get('bearish_signals', 0)
            tech_signal_score = (bullish_signals - bearish_signals) / 5.0  # Normalize to -1 to 1
            
            signal_score += tech_signal_score * self.signal_weights['technical_indicators']
            reasoning.append(f"Technical signals: {tech_signal_score:.3f} (bullish: {bullish_signals}, bearish: {bearish_signals})")
        
        # Risk assessment contribution
        risk_score = self._calculate_risk_score(market_analysis, current_price)
        signal_score += risk_score * self.signal_weights['risk_assessment']
        reasoning.append(f"Risk assessment: {risk_score:.3f}")
        
        # Market regime contribution
        market_regime = market_analysis.get('market_regime', {})
        regime_score = self._calculate_regime_score(market_regime)
        signal_score += regime_score * self.signal_weights['market_regime']
        reasoning.append(f"Market regime: {regime_score:.3f}")
        
        # Normalize signal score to 0-1 range
        normalized_score = (signal_score + 1) / 2
        
        # Determine trade signal
        trade_signal = self._score_to_signal(normalized_score)
        
        # Calculate confidence
        confidence = self._calculate_signal_confidence(market_analysis, normalized_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(trade_signal, confidence, current_price, market_analysis)
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'signal': trade_signal.value,
            'signal_score': normalized_score,
            'confidence': confidence,
            'reasoning': reasoning,
            'recommendation': recommendation,
            'current_price': current_price,
            'analysis_summary': {
                'ml_contribution': market_analysis.get('ml_prediction', {}),
                'technical_contribution': market_analysis.get('technical_signals', {}),
                'market_regime': market_regime
            }
        }
        
        # Store signal in database
        self._store_signal(result)
        
        logger.info(f"Generated signal: {trade_signal.value} (score: {normalized_score:.3f}, confidence: {confidence:.3f})")
        
        return result
    
    def get_portfolio_recommendations(self, portfolio_manager: PortfolioManager, 
                                    current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get portfolio optimization recommendations.
        
        Args:
            portfolio_manager: Portfolio manager instance
            current_prices: Current market prices
            
        Returns:
            Portfolio recommendations
        """
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
        positions = portfolio_value['positions']
        
        recommendations = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_portfolio': portfolio_value,
            'recommendations': []
        }
        
        # Analyze each position
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Get market analysis for this symbol
                # For now, we'll use a simplified analysis
                position_analysis = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'position_value': position['quantity'] * current_price,
                    'unrealized_pnl': position['quantity'] * (current_price - position['avg_cost']),
                    'unrealized_pnl_percent': (current_price - position['avg_cost']) / position['avg_cost']
                }
                
                # Generate recommendation
                if position_analysis['unrealized_pnl_percent'] > 0.1:  # 10% profit
                    recommendation = {
                        'action': 'consider_taking_profit',
                        'reason': f"Position showing {position_analysis['unrealized_pnl_percent']:.2%} profit",
                        'suggested_action': 'sell_partial'
                    }
                elif position_analysis['unrealized_pnl_percent'] < -0.05:  # 5% loss
                    recommendation = {
                        'action': 'consider_stop_loss',
                        'reason': f"Position showing {position_analysis['unrealized_pnl_percent']:.2%} loss",
                        'suggested_action': 'sell'
                    }
                else:
                    recommendation = {
                        'action': 'hold',
                        'reason': 'Position within acceptable range',
                        'suggested_action': 'hold'
                    }
                
                recommendations['recommendations'].append({
                    'symbol': symbol,
                    'analysis': position_analysis,
                    'recommendation': recommendation
                })
        
        # Overall portfolio recommendations
        total_return = portfolio_value['total_return']
        if total_return > 0.05:  # 5% total return
            recommendations['overall_recommendation'] = {
                'action': 'consider_rebalancing',
                'reason': f'Portfolio showing strong performance ({total_return:.2%})',
                'suggested_action': 'rebalance'
            }
        elif total_return < -0.03:  # 3% loss
            recommendations['overall_recommendation'] = {
                'action': 'risk_reduction',
                'reason': f'Portfolio showing losses ({total_return:.2%})',
                'suggested_action': 'reduce_exposure'
            }
        else:
            recommendations['overall_recommendation'] = {
                'action': 'maintain_strategy',
                'reason': 'Portfolio performing within expected range',
                'suggested_action': 'continue_current_strategy'
            }
        
        return recommendations
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime."""
        if len(df) < 20:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Calculate trend
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        current_price = df['close'].iloc[-1]
        
        # Determine regime
        if volatility > returns.std() * 1.5:
            regime = 'high_volatility'
        elif volatility < returns.std() * 0.5:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = 'uptrend'
        elif current_price < sma_20 < sma_50:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        return {
            'regime': regime,
            'trend': trend,
            'volatility': volatility,
            'confidence': 0.7  # Simplified confidence
        }
    
    def _calculate_risk_score(self, market_analysis: Dict[str, Any], current_price: float) -> float:
        """Calculate risk-based signal score."""
        # High volatility = lower confidence in signals
        market_regime = market_analysis.get('market_regime', {})
        volatility = market_regime.get('volatility', 0.02)
        
        # Normalize volatility (assuming 2% is normal)
        volatility_score = -min(volatility / 0.02, 2.0)  # Penalize high volatility
        
        return volatility_score
    
    def _calculate_regime_score(self, market_regime: Dict[str, Any]) -> float:
        """Calculate market regime-based signal score."""
        regime = market_regime.get('regime', 'normal_volatility')
        trend = market_regime.get('trend', 'sideways')
        
        score = 0.0
        
        # Volatility regime score
        if regime == 'low_volatility':
            score += 0.2
        elif regime == 'high_volatility':
            score -= 0.2
        
        # Trend score
        if trend == 'uptrend':
            score += 0.3
        elif trend == 'downtrend':
            score -= 0.3
        
        return score
    
    def _score_to_signal(self, score: float) -> TradeSignal:
        """Convert signal score to trade signal."""
        if score >= self.signal_thresholds['strong_buy']:
            return TradeSignal.STRONG_BUY
        elif score >= self.signal_thresholds['buy']:
            return TradeSignal.BUY
        elif score >= self.signal_thresholds['hold']:
            return TradeSignal.HOLD
        elif score >= self.signal_thresholds['sell']:
            return TradeSignal.SELL
        else:
            return TradeSignal.STRONG_SELL
    
    def _calculate_signal_confidence(self, market_analysis: Dict[str, Any], score: float) -> float:
        """Calculate confidence in the signal."""
        confidence = 0.5  # Base confidence
        
        # ML confidence contribution
        if 'ml_prediction' in market_analysis and 'error' not in market_analysis['ml_prediction']:
            ml_confidence = market_analysis['ml_prediction'].get('confidence', 0.5)
            confidence += ml_confidence * 0.3
        
        # Technical signals agreement
        if 'technical_signals' in market_analysis:
            tech_signals = market_analysis['technical_signals']
            signal_strength = abs(tech_signals.get('signal_strength', 0))
            confidence += min(signal_strength / 5.0, 0.2)  # Max 0.2 contribution
        
        # Market regime stability
        market_regime = market_analysis.get('market_regime', {})
        if market_regime.get('regime') == 'low_volatility':
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_recommendation(self, signal: TradeSignal, confidence: float, 
                               current_price: float, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed trading recommendation."""
        recommendation = {
            'action': signal.value,
            'confidence_level': self._confidence_to_strength(confidence),
            'price_target': self._calculate_price_target(signal, current_price),
            'stop_loss': self._calculate_stop_loss(signal, current_price),
            'position_size': self._calculate_position_size(confidence, current_price),
            'risk_assessment': self._assess_trade_risk(signal, confidence, market_analysis)
        }
        
        return recommendation
    
    def _confidence_to_strength(self, confidence: float) -> str:
        """Convert confidence to strength description."""
        if confidence >= 0.8:
            return 'very_high'
        elif confidence >= 0.6:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        elif confidence >= 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_price_target(self, signal: TradeSignal, current_price: float) -> float:
        """Calculate price target based on signal."""
        if signal in [TradeSignal.STRONG_BUY, TradeSignal.BUY]:
            return current_price * 1.05  # 5% upside target
        elif signal in [TradeSignal.STRONG_SELL, TradeSignal.SELL]:
            return current_price * 0.95  # 5% downside target
        else:
            return current_price
    
    def _calculate_stop_loss(self, signal: TradeSignal, current_price: float) -> float:
        """Calculate stop-loss price."""
        if signal in [TradeSignal.STRONG_BUY, TradeSignal.BUY]:
            return current_price * 0.95  # 5% stop-loss
        elif signal in [TradeSignal.STRONG_SELL, TradeSignal.SELL]:
            return current_price * 1.05  # 5% stop-loss
        else:
            return current_price
    
    def _calculate_position_size(self, confidence: float, current_price: float) -> float:
        """Calculate recommended position size."""
        # Base position size (1% of portfolio per $100k)
        base_size = 0.01
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        return base_size * confidence_multiplier
    
    def _assess_trade_risk(self, signal: TradeSignal, confidence: float, 
                          market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for the trade."""
        risk_level = 'medium'
        
        if confidence < 0.3:
            risk_level = 'high'
        elif confidence > 0.7:
            risk_level = 'low'
        
        market_regime = market_analysis.get('market_regime', {})
        volatility = market_regime.get('volatility', 0.02)
        
        if volatility > 0.03:  # High volatility
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'volatility_risk': volatility,
            'confidence_risk': 1 - confidence,
            'recommendation': 'reduce_position_size' if risk_level == 'high' else 'normal_position_size'
        }
    
    def _store_signal(self, signal_result: Dict[str, Any]):
        """Store signal result in database."""
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            serializable_data = convert_to_serializable(signal_result)
            
            self.db_manager.log_system_event(
                level='INFO',
                message=f"Trade signal generated: {signal_result['signal']}",
                module='decision_support',
                data=serializable_data
            )
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
    
    def update_signal_weights(self, new_weights: Dict[str, float]):
        """Update signal weights based on backtesting results."""
        if sum(new_weights.values()) != 1.0:
            logger.warning("Signal weights do not sum to 1.0, normalizing...")
            total = sum(new_weights.values())
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        self.signal_weights = new_weights
        logger.info(f"Updated signal weights: {self.signal_weights}")


# Example usage and testing
if __name__ == "__main__":
    # Test the decision support system
    dss = DecisionSupportSystem()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50000 + np.random.randn(100) * 100 + 50,
        'low': 50000 + np.random.randn(100) * 100 - 50,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': 'BTC/USD'
    })
    
    # Analyze market data
    analysis = dss.analyze_market_data(sample_data)
    print("Market analysis completed")
    
    # Generate trade signal
    current_price = sample_data['close'].iloc[-1]
    signal = dss.generate_trade_signal(analysis, current_price)
    print(f"Trade signal: {signal['signal']} (confidence: {signal['confidence']:.3f})")
    
    print("Decision support system test completed!")
