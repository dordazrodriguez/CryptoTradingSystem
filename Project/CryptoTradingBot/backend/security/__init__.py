"""
Security features for the cryptocurrency trading bot API.
Implements input validation, rate limiting, and secure logging.
"""

import os
import re
import hashlib
import hmac
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from functools import wraps
import jwt
from flask import request, jsonify, g
import pandas as pd

logger = logging.getLogger(__name__)


class SecurityManager:
    """Manages security features for the trading bot."""
    
    def __init__(self):
        """Initialize security manager."""
        self.secret_key = os.getenv('JWT_SECRET_KEY', 'dev-secret-key')
        self.rate_limit_storage = {}
        self.blocked_ips = set()
        self.failed_attempts = {}
        
        # Security configuration
        self.max_failed_attempts = 5
        self.block_duration = 3600  # 1 hour
        self.rate_limit_window = 3600  # 1 hour
        self.max_requests_per_hour = 1000
        
        logger.info("Security manager initialized")
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False
        
        # In production, validate against database or external service
        # For now, check against environment variable
        valid_key = os.getenv('API_KEY')
        if not valid_key:
            return True  # No API key required in development
        
        return hmac.compare_digest(api_key, valid_key)
    
    def generate_jwt_token(self, user_id: str, expires_in: int = 3600) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User identifier
            expires_in: Token expiration time in seconds
            
        Returns:
            JWT token
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def check_rate_limit(self, client_ip: str, endpoint: str = None) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: Client IP address
            endpoint: Specific endpoint (optional)
            
        Returns:
            True if within limits, False if exceeded
        """
        current_time = time.time()
        key = f"{client_ip}:{endpoint}" if endpoint else client_ip
        
        # Clean old entries
        if key in self.rate_limit_storage:
            self.rate_limit_storage[key] = [
                req_time for req_time in self.rate_limit_storage[key]
                if current_time - req_time < self.rate_limit_window
            ]
        else:
            self.rate_limit_storage[key] = []
        
        # Check rate limit
        if len(self.rate_limit_storage[key]) >= self.max_requests_per_hour:
            return False
        
        # Add current request
        self.rate_limit_storage[key].append(current_time)
        return True
    
    def is_ip_blocked(self, client_ip: str) -> bool:
        """
        Check if IP is blocked.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if blocked, False otherwise
        """
        if client_ip in self.blocked_ips:
            # Check if block has expired
            if client_ip in self.failed_attempts:
                last_attempt = self.failed_attempts[client_ip]['last_attempt']
                if time.time() - last_attempt > self.block_duration:
                    self.blocked_ips.remove(client_ip)
                    del self.failed_attempts[client_ip]
                    return False
            return True
        return False
    
    def record_failed_attempt(self, client_ip: str):
        """
        Record failed authentication attempt.
        
        Args:
            client_ip: Client IP address
        """
        current_time = time.time()
        
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = {
                'count': 0,
                'last_attempt': current_time
            }
        
        self.failed_attempts[client_ip]['count'] += 1
        self.failed_attempts[client_ip]['last_attempt'] = current_time
        
        # Block IP if too many failed attempts
        if self.failed_attempts[client_ip]['count'] >= self.max_failed_attempts:
            self.blocked_ips.add(client_ip)
            logger.warning(f"IP {client_ip} blocked due to {self.max_failed_attempts} failed attempts")
    
    def validate_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against schema.
        
        Args:
            data: Input data to validate
            schema: Validation schema
            
        Returns:
            Validation result with errors if any
        """
        errors = []
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Required field check
            if rules.get('required', False) and value is None:
                errors.append(f"{field} is required")
                continue
            
            if value is None:
                continue  # Skip validation for optional fields
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"{field} must be of type {expected_type.__name__}")
                continue
            
            # String validation
            if isinstance(value, str):
                min_length = rules.get('min_length')
                max_length = rules.get('max_length')
                
                if min_length and len(value) < min_length:
                    errors.append(f"{field} must be at least {min_length} characters")
                
                if max_length and len(value) > max_length:
                    errors.append(f"{field} must be at most {max_length} characters")
                
                # Pattern validation
                pattern = rules.get('pattern')
                if pattern and not re.match(pattern, value):
                    errors.append(f"{field} format is invalid")
            
            # Numeric validation
            if isinstance(value, (int, float)):
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None and value < min_val:
                    errors.append(f"{field} must be at least {min_val}")
                
                if max_val is not None and value > max_val:
                    errors.append(f"{field} must be at most {max_val}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def sanitize_input(self, data: Any) -> Any:
        """
        Sanitize input data to prevent injection attacks.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            # Remove potentially dangerous characters
            data = re.sub(r'[<>"\']', '', data)
            data = data.strip()
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        return data
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using secure method.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = os.urandom(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return salt.hex() + password_hash.hex()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            salt = bytes.fromhex(hashed_password[:64])
            password_hash = bytes.fromhex(hashed_password[64:])
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return hmac.compare_digest(password_hash, new_hash)
        except Exception:
            return False
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = 'INFO'):
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Log severity level
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'severity': severity,
            'client_ip': request.remote_addr if request else 'unknown',
            'user_agent': request.headers.get('User-Agent') if request else 'unknown',
            'details': details
        }
        
        # Log to file (in production, use proper logging service)
        logger.info(f"Security event: {log_entry}")
        
        # Store in database
        try:
            from data.db import get_db_manager
            db_manager = get_db_manager()
            db_manager.log_system_event(
                level=severity,
                message=f"Security event: {event_type}",
                module='security',
                data=log_entry
            )
        except Exception as e:
            logger.error(f"Failed to log security event to database: {e}")


# Security decorators
def require_auth(f):
    """Require authentication decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            token = auth_header.split(' ')[1]  # Bearer <token>
        except IndexError:
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        # Validate token
        from backend.security import SecurityManager
        security_manager = SecurityManager()
        payload = security_manager.validate_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        g.user_id = payload['user_id']
        return f(*args, **kwargs)
    
    return decorated_function


def require_api_key(f):
    """Require API key decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        from backend.security import SecurityManager
        security_manager = SecurityManager()
        
        if not security_manager.validate_api_key(api_key):
            security_manager.record_failed_attempt(request.remote_addr)
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def rate_limit(max_requests=100, window=3600):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            
            # Check if IP is blocked
            from backend.security import SecurityManager
            security_manager = SecurityManager()
            
            if security_manager.is_ip_blocked(client_ip):
                return jsonify({'error': 'IP address blocked'}), 403
            
            # Check rate limit
            if not security_manager.check_rate_limit(client_ip):
                security_manager.log_security_event(
                    'rate_limit_exceeded',
                    {'client_ip': client_ip, 'endpoint': request.endpoint},
                    'WARNING'
                )
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_input(schema: Dict[str, Any]):
    """Input validation decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()
            
            from backend.security import SecurityManager
            security_manager = SecurityManager()
            
            # Sanitize input
            data = security_manager.sanitize_input(data)
            
            # Validate input
            validation_result = security_manager.validate_input(data, schema)
            
            if not validation_result['valid']:
                return jsonify({
                    'error': 'Input validation failed',
                    'details': validation_result['errors']
                }), 400
            
            g.validated_data = data
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Input validation schemas
VALIDATION_SCHEMAS = {
    'trade_order': {
        'symbol': {
            'type': str,
            'required': True,
            'pattern': r'^[A-Z]+/[A-Z]+$',
            'max_length': 10
        },
        'side': {
            'type': str,
            'required': True,
            'pattern': r'^(buy|sell)$'
        },
        'quantity': {
            'type': float,
            'required': True,
            'min': 0.0001,
            'max': 1000
        },
        'price': {
            'type': float,
            'required': False,
            'min': 0.01,
            'max': 1000000
        },
        'order_type': {
            'type': str,
            'required': True,
            'pattern': r'^(market|limit|stop)$'
        }
    },
    'portfolio_query': {
        'limit': {
            'type': int,
            'required': False,
            'min': 1,
            'max': 1000
        },
        'symbol': {
            'type': str,
            'required': False,
            'pattern': r'^[A-Z]+/[A-Z]+$',
            'max_length': 10
        }
    },
    'date_range': {
        'start_date': {
            'type': str,
            'required': False,
            'pattern': r'^\d{4}-\d{2}-\d{2}$'
        },
        'end_date': {
            'type': str,
            'required': False,
            'pattern': r'^\d{4}-\d{2}-\d{2}$'
        }
    }
}


# Example usage and testing
if __name__ == "__main__":
    # Test the security manager
    security_manager = SecurityManager()
    
    # Test password hashing
    password = "test_password"
    hashed = security_manager.hash_password(password)
    print(f"Password hashed: {hashed[:20]}...")
    
    # Test password verification
    is_valid = security_manager.verify_password(password, hashed)
    print(f"Password verification: {is_valid}")
    
    # Test input validation
    test_data = {
        'symbol': 'BTC/USD',
        'side': 'buy',
        'quantity': 0.1,
        'price': 50000
    }
    
    validation_result = security_manager.validate_input(test_data, VALIDATION_SCHEMAS['trade_order'])
    print(f"Input validation: {validation_result}")
    
    # Test JWT token
    token = security_manager.generate_jwt_token('user123')
    print(f"JWT token generated: {token[:20]}...")
    
    payload = security_manager.validate_jwt_token(token)
    print(f"JWT validation: {payload}")
    
    print("Security manager test completed!")
