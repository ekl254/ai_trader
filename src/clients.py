"""
Shared API client factory.

Provides singleton instances of API clients to avoid creating
multiple connections and improve testability.
"""

from functools import lru_cache
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from config.config import config
from src.logger import logger


@lru_cache(maxsize=1)
def get_trading_client() -> TradingClient:
    """
    Get a singleton TradingClient instance.
    
    Uses paper trading by default. Set ALPACA_BASE_URL to
    'https://api.alpaca.markets' for live trading.
    """
    is_paper = "paper" in config.alpaca.base_url.lower()
    
    logger.info(
        "trading_client_initialized",
        paper_mode=is_paper,
        base_url=config.alpaca.base_url,
    )
    
    return TradingClient(
        config.alpaca.api_key,
        config.alpaca.secret_key,
        paper=is_paper,
    )


@lru_cache(maxsize=1)
def get_data_client() -> StockHistoricalDataClient:
    """Get a singleton StockHistoricalDataClient instance."""
    return StockHistoricalDataClient(
        config.alpaca.api_key,
        config.alpaca.secret_key,
    )


class CircuitBreaker:
    """
    Circuit breaker for API calls.
    
    Opens after `failure_threshold` consecutive failures.
    Stays open for `reset_timeout` seconds before allowing retry.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: int = 300,  # 5 minutes
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
    
    def record_success(self) -> None:
        """Record a successful API call."""
        self.failures = 0
        self.is_open = False
    
    def record_failure(self) -> None:
        """Record a failed API call."""
        import time
        
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.error(
                "circuit_breaker_opened",
                failures=self.failures,
                threshold=self.failure_threshold,
                reset_timeout=self.reset_timeout,
            )
    
    def can_proceed(self) -> bool:
        """Check if a request can proceed."""
        import time
        
        if not self.is_open:
            return True
        
        # Check if reset timeout has passed
        if self.last_failure_time is not None:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(
                    "circuit_breaker_reset",
                    elapsed_seconds=elapsed,
                )
                self.is_open = False
                self.failures = 0
                return True
        
        logger.warning(
            "circuit_breaker_blocked",
            failures=self.failures,
            is_open=self.is_open,
        )
        return False
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        import time
        
        time_until_reset = None
        if self.is_open and self.last_failure_time is not None:
            elapsed = time.time() - self.last_failure_time
            time_until_reset = max(0, self.reset_timeout - elapsed)
        
        return {
            "is_open": self.is_open,
            "failures": self.failures,
            "threshold": self.failure_threshold,
            "time_until_reset": time_until_reset,
        }


# Global circuit breaker for trading API
trading_circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=300)
