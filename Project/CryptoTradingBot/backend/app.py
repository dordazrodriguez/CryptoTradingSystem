"""Flask REST API for the cryptocurrency trading bot."""

from flask import Flask, request, jsonify, g, Response
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import sys
from functools import wraps
from flask import stream_with_context
import json
import time
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db import get_db_manager
from trading_engine.portfolio import PortfolioManager
from trading_engine.risk_manager import RiskManager
from trading_engine.decision_support import DecisionSupportSystem
from trading_engine.indicators import TechnicalIndicators
from ml_models.predictor import CryptoPredictionModel
from data.collector import AlpacaDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['RATE_LIMIT_STORAGE_URL'] = 'memory://'
DISABLE_RATE_LIMIT = os.getenv('DISABLE_RATE_LIMIT', 'false').lower() in ('1', 'true', 'yes')

rate_limit_storage = {}

db_manager = get_db_manager()
portfolio_manager = PortfolioManager()
risk_manager = RiskManager()
indicators = TechnicalIndicators()
decision_support = DecisionSupportSystem()

ml_model = None
try:
    ml_model = CryptoPredictionModel()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'trained_model.pkl')
    if os.path.exists(model_path):
        ml_model.load_model(model_path)
        decision_support.ml_model = ml_model
except Exception as e:
    logger.warning(f"Could not initialize ML model: {e}")

data_collector = None
try:
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    if api_key and secret_key:
        data_collector = AlpacaDataCollector(api_key, secret_key, paper_trading=True)
except Exception as e:
    logger.warning(f"Could not initialize data collector: {e}")


def rate_limit(max_requests=100, window=3600):
    """Rate limiting decorator. If DISABLE_RATE_LIMIT=true, this becomes a no-op."""
    if DISABLE_RATE_LIMIT:
        def passthrough_decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                return f(*args, **kwargs)
            return decorated_function
        return passthrough_decorator

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = (
                request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
                or request.headers.get('X-Real-IP')
                or request.remote_addr
            )
            endpoint = request.endpoint or request.path
            key = f"{client_ip}:{endpoint}"
            current_time = time.time()

            if key in rate_limit_storage:
                rate_limit_storage[key] = [
                    req_time for req_time in rate_limit_storage[key]
                    if current_time - req_time < window
                ]
            else:
                rate_limit_storage[key] = []

            if len(rate_limit_storage[key]) >= max_requests:
                response = jsonify({'error': 'Rate limit exceeded'})
                response.headers['Retry-After'] = str(int(window))
                return response, 429

            rate_limit_storage[key].append(current_time)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_json(f):
    """Validate JSON request decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.is_json:
            try:
                request.get_json()
            except Exception as e:
                return jsonify({'error': 'Invalid JSON'}), 400
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def before_request():
    g.start_time = time.time()
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")


@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    logger.info(f"Response {response.status_code} - {duration:.3f}s")
    return response


@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0',
        'components': {
            'database': 'healthy',
            'ml_model': 'healthy' if ml_model and ml_model.is_trained else 'not_available',
            'data_collector': 'healthy' if data_collector else 'not_available'
        }
    }
    
    try:
        db_manager.execute_query("SELECT 1")
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['components']['database'] = f'error: {str(e)}'
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


def _record_portfolio_metric_if_needed():
    """Record a portfolio metric snapshot if one hasn't been recorded recently."""
    try:
        cutoff_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        query = "SELECT COUNT(*) as count FROM portfolio_metrics WHERE timestamp >= ?"
        result = db_manager.execute_query(query, (cutoff_time,))
        count = result[0]['count'] if result else 0
        
        if count == 0:
            try:
                api_key = os.getenv('ALPACA_API_KEY')
                api_secret = os.getenv('ALPACA_SECRET_KEY')
                
                total_value = 100000.0
                cash_balance = 100000.0
                position_qty = 0.0
                position_avg_price = 0.0
                
                if api_key and api_secret:
                    try:
                        base_url = "https://paper-api.alpaca.markets"
                        headers = {
                            "APCA-API-KEY-ID": api_key,
                            "APCA-API-SECRET-KEY": api_secret
                        }
                        account_response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=5)
                        if account_response.status_code == 200:
                            account = account_response.json()
                            equity_str = account.get('equity')
                            cash_str = account.get('cash')
                            total_value = float(equity_str) if equity_str else 100000.0
                            cash_balance = float(cash_str) if cash_str else 100000.0
                    except Exception:
                        pass
                
                timestamp_ms = int(time.time() * 1000)
                db_manager.insert_portfolio_metric(
                    timestamp_ms,
                    'BTC/USD',
                    total_value,
                    cash_balance,
                    position_qty,
                    position_avg_price
                )
                logger.info(f"Recorded initial portfolio metric: value=${total_value:,.2f}, cash=${cash_balance:,.2f}")
            except Exception as e:
                logger.warning(f"Failed to record portfolio metric: {e}")
    except Exception as e:
        logger.debug(f"Error checking/recording portfolio metric: {e}")


@app.route('/api/portfolio', methods=['GET'])
@rate_limit(max_requests=60, window=3600)
def get_portfolio():
    """Get current portfolio status - syncs with Alpaca if available."""
    try:
        _record_portfolio_metric_if_needed()
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and api_secret:
            try:
                base_url = "https://paper-api.alpaca.markets"
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret
                }
                
                account_response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
                positions_response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                
                if account_response.status_code != 200:
                    logger.error(f"Alpaca account API failed: status={account_response.status_code}")
                    raise Exception(f"Alpaca account API returned {account_response.status_code}")
                
                if positions_response.status_code != 200:
                    logger.warning(f"Alpaca positions API failed: status={positions_response.status_code}")
                    positions = []
                else:
                    positions = positions_response.json()
                
                account = account_response.json()
                
                equity_str = account.get('equity')
                if equity_str:
                    equity = float(equity_str)
                else:
                    equity = float(account.get('portfolio_value', 0) or account.get('last_equity', 0))
                    logger.warning(f"Equity field not found, using fallback: {equity}")
                
                cash_str = account.get('cash')
                if cash_str:
                    cash = float(cash_str)
                else:
                    cash = float(account.get('buying_power', 0))
                    logger.warning(f"Cash field not found, using buying_power: {cash}")
                
                if data_collector:
                    current_prices = data_collector.get_latest_prices()
                else:
                    current_prices = {'BTC/USD': float(account.get('last_equity', equity)) / 1.0}
                
                unrealized_pl = sum(float(pos.get('unrealized_pl', 0)) for pos in positions)
                
                db_stats = db_manager.get_database_stats()
                realized_pnl_from_db = db_stats.get('realized_pnl', 0) or 0.0
                total_pnl = unrealized_pl + realized_pnl_from_db
                
                total_trades = db_stats.get('total_trades', 0) or len(positions)
                
                win_rate = 0.0
                if total_trades > 0:
                    wins = sum(1 for pos in positions if float(pos.get('unrealized_pl', 0)) > 0)
                    win_rate = (wins / len(positions)) * 100 if positions else 0.0
                
                positions_dict = {}
                for pos in positions:
                    symbol = pos.get('symbol', '')
                    qty = float(pos.get('qty', 0))
                    if qty != 0:
                        positions_dict[symbol] = {
                            'quantity': abs(qty),
                            'avg_cost': float(pos.get('avg_entry_price', 0))
                        }
                
                invested_value = equity - cash
                
                starting_equity = float(account.get('last_equity', 100000) or account.get('initial_margin', 100000) or 100000)
                total_return_pct = ((equity - starting_equity) / starting_equity * 100) if starting_equity > 0 else 0.0
                
                portfolio_value = {
                    'total_value': equity,
                    'cash_balance': cash,
                    'invested_value': invested_value,
                    'unrealized_pnl': total_pnl,
                    'realized_pnl': realized_pnl_from_db,
                    'total_return': total_return_pct,
                    'positions': positions_dict
                }
                
                performance_metrics = {
                    'total_value': equity,
                    'total_return': total_return_pct,
                    'total_pnl': total_pnl,
                    'cash_balance': cash,
                    'invested_value': invested_value,
                    'total_trades': total_trades,
                    'winning_trades': len([p for p in positions if float(p.get('unrealized_pl', 0)) > 0]),
                    'losing_trades': len([p for p in positions if float(p.get('unrealized_pl', 0)) < 0]),
                    'win_rate': win_rate,
                    'positions_count': len(positions)
                }
                
                logger.info(f"Portfolio synced from Alpaca: equity=${equity:,.2f}, cash=${cash:,.2f}, positions={len(positions)}, P&L=${total_pnl:.2f}")
                
                return jsonify({
                    'portfolio': portfolio_value,
                    'performance': performance_metrics,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'alpaca'
                })
            except Exception as alpaca_err:
                logger.error(f"Could not fetch from Alpaca API, falling back to local: {alpaca_err}", exc_info=True)
        
        logger.warning("Using local portfolio manager - Alpaca sync failed or unavailable")
        if data_collector:
            current_prices = data_collector.get_latest_prices()
        else:
            current_prices = {'BTC/USD': 50000, 'ETH/USD': 3000}
        
        portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
        performance_metrics = portfolio_manager.get_performance_metrics(current_prices)
        
        return jsonify({
            'portfolio': portfolio_value,
            'performance': performance_metrics,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'local',
            'warning': 'Alpaca sync unavailable - showing local portfolio data'
        })
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/positions', methods=['GET'])
@rate_limit(max_requests=60, window=3600)
def get_positions():
    """Get current positions - syncs with Alpaca if available."""
    try:
        # Try to fetch live positions from Alpaca first
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and api_secret:
            try:
                base_url = "https://paper-api.alpaca.markets"
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret
                }
                
                positions_response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                
                if positions_response.status_code == 200:
                    positions = positions_response.json()
                    
                    positions_dict = {}
                    for pos in positions:
                        symbol = pos.get('symbol', '')
                        qty = float(pos.get('qty', 0))
                        if qty != 0:
                            positions_dict[symbol] = {
                                'quantity': abs(qty),
                                'avg_cost': float(pos.get('avg_entry_price', 0)),
                                'current_price': float(pos.get('current_price', 0)),
                                'market_value': float(pos.get('market_value', 0)),
                                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                                'unrealized_plpc': float(pos.get('unrealized_plpc', 0))
                            }
                    
                    return jsonify({
                        'positions': positions_dict,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'source': 'alpaca'
                    })
            except Exception as alpaca_err:
                logger.warning(f"Could not fetch positions from Alpaca API, falling back to local: {alpaca_err}")
        
        positions = portfolio_manager.positions
        return jsonify({
            'positions': positions,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'local'
        })
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades', methods=['GET'])
@rate_limit(max_requests=100, window=3600)
def get_trades():
    """Get trade history with optional filtering."""
    try:
        limit = request.args.get('limit', 100, type=int)
        symbol = request.args.get('symbol')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        trades = db_manager.execute_query(query, tuple(params))
        
        logger.info(f"Fetched {len(trades)} trades from database")
        
        if len(trades) == 0:
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and api_secret:
                try:
                    base_url = "https://paper-api.alpaca.markets"
                    headers = {
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": api_secret
                    }
                    
                    # Fetch filled orders from Alpaca (these are completed trades)
                    params_alpaca = {
                        'status': 'closed',  # Get filled orders
                        'limit': min(limit, 500),
                        'direction': 'desc'
                    }
                    
                    if symbol:
                        # Normalize symbol for Alpaca (BTC/USD -> BTCUSD)
                        alpaca_symbol = symbol.replace('/', '') if '/' in symbol else symbol
                        params_alpaca['symbols'] = alpaca_symbol
                    
                    response = requests.get(
                        f"{base_url}/v2/orders",
                        headers=headers,
                        params=params_alpaca,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        orders = response.json()
                        
                        for order in orders:
                            if order.get('status') == 'filled' and order.get('filled_qty'):
                                try:
                                    filled_qty = float(order.get('filled_qty', 0))
                                    filled_avg_price = float(order.get('filled_avg_price', 0))
                                    
                                    commission_rate = 0.001
                                    commission = filled_qty * filled_avg_price * commission_rate
                                    net_amount = (filled_qty * filled_avg_price) - commission
                                    
                                    side = 'buy' if order.get('side', '').lower() == 'buy' else 'sell'
                                    
                                    trade_symbol = order.get('symbol', symbol or 'BTC/USD')
                                    if '/' not in trade_symbol and len(trade_symbol) > 3:
                                        if trade_symbol.endswith('USD'):
                                            trade_symbol = trade_symbol[:-3] + '/USD'
                                        elif trade_symbol.endswith('USDT'):
                                            trade_symbol = trade_symbol[:-4] + '/USDT'
                                    
                                    trade = {
                                        'trade_id': order.get('id', f"alpaca_{order.get('client_order_id', 'unknown')}"),
                                        'order_id': order.get('id', ''),
                                        'symbol': trade_symbol,
                                        'side': side,
                                        'quantity': filled_qty,
                                        'price': filled_avg_price,
                                        'commission': commission,
                                        'net_amount': net_amount,
                                        'timestamp': order.get('filled_at', order.get('created_at', datetime.now(timezone.utc).isoformat())),
                                        'strategy': order.get('order_class', 'market')
                                    }
                                    
                                    trades.append(trade)
                                except (ValueError, KeyError) as e:
                                    logger.warning(f"Skipping invalid order: {e}")
                                    continue
                        
                        trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                        trades = trades[:limit]
                        
                        logger.info(f"Fetched {len(trades)} filled orders from Alpaca as trades")
                        
                except Exception as e:
                    logger.warning(f"Could not fetch trades from Alpaca: {e}")
        
        if len(trades) > 0:
            logger.info(f"Sample trade: {trades[0]}")
        
        return jsonify({
            'trades': trades,
            'count': len(trades),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'database' if len(trades) > 0 and trades[0].get('trade_id', '').startswith('trade_') else 'alpaca'
        })
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/<trade_id>', methods=['GET'])
@rate_limit(max_requests=100, window=3600)
def get_trade(trade_id):
    """Get specific trade details."""
    try:
        query = "SELECT * FROM trades WHERE trade_id = ?"
        trades = db_manager.execute_query(query, (trade_id,))
        
        if not trades:
            return jsonify({'error': 'Trade not found'}), 404
        
        return jsonify({
            'trade': trades[0],
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting trade {trade_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orders', methods=['GET'])
@rate_limit(max_requests=100, window=3600)
def get_orders():
    """Get recent orders from Alpaca with optional filtering."""
    try:
        limit = request.args.get('limit', 100, type=int)
        status = request.args.get('status', 'all')
        symbol = request.args.get('symbol')
        side = request.args.get('side')
        
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not api_secret:
            logger.warning("Alpaca API credentials not configured, returning empty orders")
            return jsonify({
                'orders': [],
                'count': 0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'alpaca',
                'error': 'Alpaca API credentials not configured'
            })
        
        base_url = "https://paper-api.alpaca.markets"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        
        params = {
            'limit': min(limit, 500),
            'direction': 'desc',
            'nested': 'true'
        }
        
        original_status = status.lower() if status != 'all' else None
        
        if status != 'all':
            status_map = {
                'filled': 'closed',
                'partially_filled': 'closed',
                'closed': 'closed',
                'cancelled': 'closed',
                'canceled': 'closed',
                'open': 'open',
                'pending': 'open',
                'pending_new': 'open',
                'new': 'open',
            }
            api_status = status_map.get(status.lower(), 'all')
            params['status'] = api_status
        else:
            params['status'] = 'all'
        
        if symbol:
            normalized_symbol = symbol.replace('/', '') if '/' in symbol else symbol
            params['symbols'] = normalized_symbol
        
        logger.debug(f"Fetching orders from Alpaca with params: {params}")
        response = requests.get(
            f"{base_url}/v2/orders",
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else 'No error message'
            logger.error(f"Failed to fetch Alpaca orders: status={response.status_code}, response={error_text}")
            return jsonify({
                'orders': [],
                'count': 0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'alpaca',
                'error': f"Alpaca API returned {response.status_code}: {error_text}"
            }), response.status_code
        
        try:
            orders = response.json()
            if not isinstance(orders, list):
                logger.warning(f"Alpaca orders response is not a list: {type(orders)}, converting...")
                orders = [orders] if orders else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Alpaca orders response as JSON: {e}, response: {response.text[:200]}")
            return jsonify({
                'orders': [],
                'count': 0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'alpaca',
                'error': f"Invalid JSON response from Alpaca API: {str(e)}"
            }), 500
        
        if original_status and original_status in ['filled', 'partially_filled', 'canceled', 'cancelled']:
            orders = [o for o in orders if o.get('status', '').lower() == original_status.replace('cancelled', 'canceled')]
            logger.debug(f"Filtered orders by status '{original_status}': {len(orders)} orders")
        
        if side:
            orders = [o for o in orders if o.get('side', '').lower() == side.lower()]
        
        formatted_orders = []
        for order in orders:
            try:
                order_time = order.get('submitted_at') or order.get('created_at') or order.get('updated_at', '')
                if order_time:
                    try:
                        time_str = order_time.replace('Z', '+00:00')
                        if '+' not in time_str and 'T' in time_str:
                            time_str = time_str + '+00:00'
                        order_dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    except Exception as parse_err:
                        logger.warning(f"Failed to parse timestamp '{order_time}': {parse_err}")
                        order_dt = datetime.now(timezone.utc)
                else:
                    order_dt = datetime.now(timezone.utc)
                
                qty = float(order.get('qty', 0) or order.get('quantity', 0))
                filled_qty = float(order.get('filled_qty', 0) or order.get('filled_quantity', 0))
                
                order_type = (order.get('order_type') or order.get('type') or 'market').lower()
                
                limit_price = order.get('limit_price')
                stop_price = order.get('stop_price')
                filled_avg_price = order.get('filled_avg_price')
                
                limit_price = float(limit_price) if limit_price is not None else None
                stop_price = float(stop_price) if stop_price is not None else None
                filled_avg_price = float(filled_avg_price) if filled_avg_price is not None else None
                
                formatted_orders.append({
                    'order_id': order.get('id', ''),
                    'client_order_id': order.get('client_order_id', ''),
                    'symbol': order.get('symbol', ''),
                    'side': (order.get('side') or '').lower(),
                    'type': order_type,
                    'status': (order.get('status') or '').lower(),
                    'quantity': qty,
                    'filled_quantity': filled_qty,
                    'limit_price': limit_price,
                    'stop_price': stop_price,
                    'filled_avg_price': filled_avg_price,
                    'time_in_force': (order.get('time_in_force') or '').lower(),
                    'timestamp': order_dt.isoformat(),
                    'submitted_at': order.get('submitted_at', ''),
                    'updated_at': order.get('updated_at', ''),
                    'created_at': order.get('created_at', ''),
                    'source': 'alpaca'
                })
            except Exception as e:
                logger.warning(f"Error formatting order {order.get('id', 'unknown')}: {e}", exc_info=True)
                continue
        
        formatted_orders.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        logger.info(f"Fetched {len(formatted_orders)} orders from Alpaca (from {len(orders)} raw orders)")
        if len(formatted_orders) > 0:
            logger.debug(f"Sample order: {formatted_orders[0]}")
        
        return jsonify({
            'orders': formatted_orders,
            'count': len(formatted_orders),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'alpaca'
        })
    except Exception as e:
        logger.error(f"Error getting orders from Alpaca: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/market/prices', methods=['GET'])
@rate_limit(max_requests=120, window=3600)
def get_current_prices():
    """Get current market prices."""
    try:
        if data_collector:
            prices = data_collector.get_latest_prices()
        else:
            # Fallback to placeholder prices
            prices = {'BTC/USD': 50000, 'ETH/USD': 3000, 'ADA/USD': 0.5}
        
        return jsonify({
            'prices': prices,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        return jsonify({'error': str(e)}), 500


# Server-Sent Events: live price stream
@app.route('/stream/prices', methods=['GET'])
def stream_prices():
    """Stream live prices via SSE (1 update/sec)."""
    @stream_with_context
    def event_stream():
        while True:
            try:
                if data_collector:
                    prices = data_collector.get_latest_prices()
                else:
                    prices = {'BTC/USD': 50000, 'ETH/USD': 3000, 'ADA/USD': 0.5}
                payload = {
                    'prices': prices,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1)
            except GeneratorExit:
                break
            except Exception as e:
                logger.warning(f"SSE stream error: {e}")
                time.sleep(1)
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }
    return Response(event_stream(), headers=headers, mimetype='text/event-stream')


@app.route('/api/market/history/<symbol>', methods=['GET'])
@rate_limit(max_requests=60, window=3600)
def get_price_history(symbol):
    """Get price history for a symbol."""
    try:
        limit = request.args.get('limit', 100, type=int)
        timeframe = request.args.get('timeframe', '1m')
        
        query = """
        SELECT * FROM price_data 
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        history = db_manager.execute_query(query, (symbol, timeframe, limit))
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'history': history,
            'count': len(history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting price history for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/indicators/<symbol>', methods=['GET'])
@rate_limit(max_requests=60, window=3600)
def get_indicators(symbol):
    """Get technical indicators for a symbol."""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        query = """
        SELECT * FROM indicators 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        indicators_data = db_manager.execute_query(query, (symbol, limit))
        
        return jsonify({
            'symbol': symbol,
            'indicators': indicators_data,
            'count': len(indicators_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/<symbol>', methods=['GET'])
@rate_limit(max_requests=30, window=3600)
def get_predictions(symbol):
    """Get ML predictions for a symbol."""
    try:
        if not ml_model or not ml_model.is_trained:
            return jsonify({'error': 'ML model not available'}), 503
        
        limit = request.args.get('limit', 10, type=int)
        
        query = """
        SELECT * FROM ml_predictions 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        predictions = db_manager.execute_query(query, (symbol, limit))
        
        return jsonify({
            'symbol': symbol,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals/<symbol>', methods=['GET'])
@rate_limit(max_requests=30, window=3600)
def get_trading_signals(symbol):
    """Get trading signals for a symbol."""
    try:
        query = """
        SELECT * FROM price_data 
        WHERE symbol = ? 
        ORDER BY timestamp DESC 
        LIMIT 100
        """
        
        price_data = db_manager.execute_query(query, (symbol,))
        
        if not price_data:
            return jsonify({'error': 'No price data available'}), 404
        
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        analysis = decision_support.analyze_market_data(df)
        
        current_price = df['close'].iloc[-1]
        signal = decision_support.generate_trade_signal(analysis, current_price)
        
        return jsonify({
            'symbol': symbol,
            'signal': signal,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk/assessment', methods=['GET'])
@rate_limit(max_requests=30, window=3600)
def get_risk_assessment():
    """Get current risk assessment."""
    try:
        positions = portfolio_manager.positions
        
        if data_collector:
            current_prices = data_collector.get_latest_prices()
        else:
            current_prices = {'BTC/USD': 50000, 'ETH/USD': 3000}
        
        risk_report = risk_manager.get_risk_report(positions, current_prices, 100000)
        
        return jsonify({
            'risk_assessment': risk_report,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting risk assessment: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/portfolio-history', methods=['GET'])
@rate_limit(max_requests=60, window=3600)
def get_portfolio_history():
    """Get portfolio value history over time from database."""
    try:
        _record_portfolio_metric_if_needed()
        
        limit = request.args.get('limit', 100, type=int)
        hours = request.args.get('hours', None, type=int)
        
        if hours:
            cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            query = """
                SELECT timestamp, total_value, cash_balance, total_pnl, total_return
                FROM portfolio_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                LIMIT ?
            """
            params = (cutoff_time, limit)
        else:
            query = """
                SELECT timestamp, total_value, cash_balance, total_pnl, total_return
                FROM portfolio_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (limit,)
        
        metrics = db_manager.execute_query(query, params)
        
        metrics.reverse()
        
        chart_data = []
        for metric in metrics:
            try:
                ts = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
                chart_data.append({
                    'time': ts.strftime('%H:%M'),
                    'timestamp': metric['timestamp'],
                    'value': float(metric['total_value']) if metric['total_value'] else 0,
                    'cash': float(metric['cash_balance']) if metric['cash_balance'] else 0,
                    'pnl': float(metric['total_pnl']) if metric['total_pnl'] else 0,
                    'return': float(metric['total_return']) if metric['total_return'] else 0,
                })
            except Exception as e:
                logger.warning(f"Error parsing metric timestamp: {e}, metric: {metric}")
                continue
        
        if len(chart_data) == 0:
            try:
                api_key = os.getenv('ALPACA_API_KEY')
                api_secret = os.getenv('ALPACA_SECRET_KEY')
                
                total_value = 100000.0
                cash_balance = 100000.0
                
                if api_key and api_secret:
                    try:
                        base_url = "https://paper-api.alpaca.markets"
                        headers = {
                            "APCA-API-KEY-ID": api_key,
                            "APCA-API-SECRET-KEY": api_secret
                        }
                        account_response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=5)
                        if account_response.status_code == 200:
                            account = account_response.json()
                            equity_str = account.get('equity')
                            cash_str = account.get('cash')
                            total_value = float(equity_str) if equity_str else 100000.0
                            cash_balance = float(cash_str) if cash_str else 100000.0
                    except Exception:
                        pass
                
                now = datetime.now(timezone.utc)
                chart_data.append({
                    'time': now.strftime('%H:%M'),
                    'timestamp': now.isoformat(),
                    'value': total_value,
                    'cash': cash_balance,
                    'pnl': 0.0,
                    'return': 0.0,
                })
            except Exception as e:
                logger.warning(f"Failed to create default chart entry: {e}")
        
        logger.info(f"Fetched {len(chart_data)} portfolio metrics for history chart")
        
        return jsonify({
            'history': chart_data,
            'count': len(chart_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/performance', methods=['GET'])
@rate_limit(max_requests=30, window=3600)
def get_performance_metrics():
    """Get performance metrics - syncs with Alpaca if available."""
    try:
        _record_portfolio_metric_if_needed()
        
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and api_secret:
            try:
                base_url = "https://paper-api.alpaca.markets"
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret
                }
                
                account_response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
                positions_response = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
                
                if account_response.status_code == 200:
                    account = account_response.json()
                    positions = positions_response.json() if positions_response.status_code == 200 else []
                    
                    equity_str = account.get('equity')
                    cash_str = account.get('cash')
                    equity = float(equity_str) if equity_str else 100000.0
                    cash = float(cash_str) if cash_str else 100000.0
                    
                    unrealized_pl = sum(float(pos.get('unrealized_pl', 0)) for pos in positions)
                    
                    db_stats = db_manager.get_database_stats()
                    realized_pnl_from_db = db_stats.get('realized_pnl', 0) or 0.0
                    total_pnl = unrealized_pl + realized_pnl_from_db
                    
                    total_trades = db_stats.get('trades_count', 0) or 0
                    
                    win_rate = 0.0
                    winning_count = 0
                    losing_count = 0
                    if positions:
                        for pos in positions:
                            unrealized_pl = float(pos.get('unrealized_pl', 0))
                            if unrealized_pl > 0:
                                winning_count += 1
                            elif unrealized_pl < 0:
                                losing_count += 1
                        if winning_count + losing_count > 0:
                            win_rate = (winning_count / (winning_count + losing_count)) * 100
                    elif total_trades > 0:
                        win_rate = 0.0
                    
                    starting_equity = float(account.get('last_equity', 100000) or account.get('initial_margin', 100000) or 100000)
                    total_return_pct = ((equity - starting_equity) / starting_equity * 100) if starting_equity > 0 else 0.0
                    
                    invested_value = equity - cash
                    
                    portfolio_metrics = {
                        'total_value': equity,
                        'total_return': total_return_pct,
                        'total_pnl': total_pnl,
                        'cash_balance': cash,
                        'invested_value': invested_value,
                        'unrealized_pnl': unrealized_pl,
                        'realized_pnl': realized_pnl_from_db,
                        'total_trades': total_trades,
                        'winning_trades': winning_count,
                        'losing_trades': losing_count,
                        'win_rate': win_rate,
                        'positions_count': len(positions),
                        'total_fees': db_stats.get('trades_count', 0) * 0.001 * 1000,
                        'sharpe_ratio': None,
                        'max_drawdown': None
                    }
                    
                    return jsonify({
                        'portfolio_metrics': portfolio_metrics,
                        'database_stats': db_stats,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'source': 'alpaca'
                    })
            except Exception as alpaca_err:
                logger.warning(f"Could not fetch from Alpaca API for performance metrics, falling back to local: {alpaca_err}")
        
        if data_collector:
            current_prices = data_collector.get_latest_prices()
        else:
            current_prices = {'BTC/USD': 50000, 'ETH/USD': 3000}
        portfolio_metrics = portfolio_manager.get_performance_metrics(current_prices)
        
        db_stats = db_manager.get_database_stats()
        
        return jsonify({
            'portfolio_metrics': portfolio_metrics,
            'database_stats': db_stats,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'local'
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['GET'])
@rate_limit(max_requests=30, window=3600)
def get_system_logs():
    """Get system logs."""
    try:
        level = request.args.get('level', 'INFO')
        limit = request.args.get('limit', 100, type=int)
        
        query = """
        SELECT * FROM system_logs 
        WHERE level = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        logs = db_manager.execute_query(query, (level, limit))
        
        return jsonify({
            'logs': logs,
            'count': len(logs),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
