"""
Simple trade execution engine from the completed project.
Provides simulated trade execution for the trading bot.
"""

from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    side: str
    price: float
    qty: float
    fee: float
    
    def __repr__(self) -> str:
        return (f"ExecutionResult(side={self.side}, qty={self.qty:.6f}, "
                f"price=${self.price:.2f}, fee=${self.fee:.4f})")


class SimulatorExecutor:
    """
    Simulator for trade execution.
    Executes trades in a simulated environment without real money.
    """
    
    def __init__(self, fee_bps: float = 10.0) -> None:
        """
        Initialize simulator executor.
        
        Args:
            fee_bps: Fee in basis points (default: 10 bps = 0.10%)
        """
        self.fee_bps = fee_bps
        logger.info(f"SimulatorExecutor initialized with {fee_bps} bps fee")

    def market(self, side: str, price: float, notional_usd: float, symbol: str | None = None) -> ExecutionResult:
        """
        Execute a market order in simulation.
        
        Args:
            side: "buy" or "sell"
            price: Current market price
            notional_usd: Trade size in USD
            
        Returns:
            ExecutionResult with trade details
            
        Raises:
            ValueError: If price or notional is invalid
        """
        # Validate inputs
        if price <= 0:
            raise ValueError("price must be > 0")
        if notional_usd <= 0:
            raise ValueError("notional_usd must be > 0")
        
        # Calculate quantity and fee
        qty = notional_usd / price
        fee = (self.fee_bps / 10000.0) * notional_usd
        
        result = ExecutionResult(side=side, price=price, qty=qty, fee=fee)
        
        logger.info(f"Executed {side} order: qty={qty:.6f}, price=${price:.2f}, fee=${fee:.4f}")
        
        return result
    
    def calculate_fee(self, notional_usd: float) -> float:
        """
        Calculate fee for a trade.
        
        Args:
            notional_usd: Trade size in USD
            
        Returns:
            Fee amount in USD
        """
        return (self.fee_bps / 10000.0) * notional_usd
    
    def calculate_qty(self, notional_usd: float, price: float) -> float:
        """
        Calculate quantity from notional value.
        
        Args:
            notional_usd: Trade size in USD
            price: Current market price
            
        Returns:
            Quantity
        """
        return notional_usd / price


class AlpacaExecutor:
    """
    Real executor using Alpaca via CCXT for paper trading.
    """
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True, fee_bps: float = 10.0) -> None:
        import ccxt  # local import to keep optional dependency boundary similar
        self.exchange = ccxt.alpaca({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        self.fee_bps = fee_bps
        # Load markets to enable order creation
        try:
            self.exchange.load_markets()
            logger.info(f"AlpacaExecutor initialized (sandbox={sandbox}), loaded {len(self.exchange.markets)} markets")
        except Exception as e:
            logger.warning(f"Could not load Alpaca markets: {e}. Order creation may fail.")

    def market(self, side: str, price: float, notional_usd: float, symbol: str) -> ExecutionResult:
        if price <= 0:
            raise ValueError("price must be > 0")
        if notional_usd <= 0:
            raise ValueError("notional_usd must be > 0")
        if not symbol:
            raise ValueError("symbol is required for AlpacaExecutor")

        # Calculate quantity; preserve precision for crypto (don't round down unnecessarily)
        # Use higher precision to avoid truncation errors
        qty = notional_usd / price
        # Round to 9 decimal places (BTC precision) but preserve exact value if possible
        qty = float(f"{qty:.9f}")

        try:
            # Check if symbol exists in markets, try alternative formats if needed
            if symbol not in self.exchange.markets:
                # Try with slash format
                alt_symbol = symbol.replace('USD', '/USD') if not '/' in symbol else symbol
                if alt_symbol in self.exchange.markets:
                    symbol = alt_symbol
                    logger.info(f"Using alternative symbol format: {alt_symbol}")
                else:
                    # List available crypto symbols for debugging
                    crypto_symbols = [s for s in self.exchange.markets.keys() if 'BTC' in s.upper() or 'USD' in s.upper()]
                    logger.error(f"Symbol '{symbol}' not found. Available BTC/USD related symbols: {crypto_symbols[:10]}")
                    raise ValueError(f"Symbol '{symbol}' not available in Alpaca. Try one of: {crypto_symbols[:5]}")
            
            order = self.exchange.create_order(symbol, type='market', side=side, amount=qty)
            fee = (self.fee_bps / 10000.0) * notional_usd
            logger.info(f"Alpaca market {side} placed: id={order.get('id')} qty={qty:.8f} {symbol} @ ~${price:.2f}")
            return ExecutionResult(side=side, price=price, qty=qty, fee=fee)
        except Exception as e:
            logger.error(f"Failed to place Alpaca order: {e}")
            raise
