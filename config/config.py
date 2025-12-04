"""Configuration management for the trading system."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()


class AlpacaConfig(BaseModel):
    """Alpaca API configuration."""

    api_key: str = Field(...)
    secret_key: str = Field(...)
    base_url: str = Field(default="https://paper-api.alpaca.markets")


class TradingConfig(BaseModel):
    """Trading strategy configuration."""

    max_positions: int = Field(default=10)
    risk_per_trade: float = Field(default=0.02)
    max_position_size: float = Field(default=0.10)  # 10% max per position
    portfolio_value: float = Field(default=100000.0)

    # Scoring thresholds (optimized based on performance data)
    min_composite_score: float = Field(
        default=72.5
    )  # Raised for higher quality entries
    min_factor_score: float = Field(default=40.0)

    # Score weights for composite calculation (must sum to 1.0)
    weight_technical: float = Field(default=0.50)
    weight_sentiment: float = Field(default=0.50)
    weight_fundamental: float = Field(default=0.00)  # Currently unused

    # Technical indicator settings
    rsi_period: int = Field(default=14)
    rsi_oversold: float = Field(default=30.0)
    rsi_overbought: float = Field(default=70.0)

    # Risk management
    stop_loss_pct: float = Field(
        default=0.03
    )  # 3% stop loss (optimized via 5-year backtest)
    take_profit_pct: float = Field(
        default=0.08
    )  # 8% take profit (optimized via 5-year backtest)
    # NOTE: trailing_stop_pct removed - feature not implemented

    # Position management
    min_volume: int = Field(default=1_000_000)  # Minimum daily volume

    # Rebalancing (replace weak positions with better opportunities)
    enable_rebalancing: bool = Field(default=True)  # Auto-replace weak positions
    rebalance_score_diff: float = Field(
        default=10.0
    )  # Replace if new candidate scores 10+ points higher
    rebalance_min_hold_time: int = Field(
        default=30
    )  # Minutes - don't replace positions held < 30 min
    rebalance_cooldown_minutes: int = Field(
        default=60
    )  # Minimum minutes between rebalancing swaps
    rebalance_partial_sell_pct: float = Field(
        default=0.0
    )  # Percentage of position to sell (0.0 = full position, 0.5 = 50%)

    @field_validator("risk_per_trade")
    @classmethod
    def validate_risk(cls, v: float) -> float:
        """Ensure risk per trade is reasonable."""
        if not 0.001 <= v <= 0.05:
            raise ValueError("Risk per trade must be between 0.1% and 5%")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trading.log")
    log_format: str = Field(default="json")


class Config(BaseModel):
    """Main configuration object."""

    alpaca: AlpacaConfig
    trading: TradingConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required. "
                "Please set them in your .env file or environment."
            )

        return cls(
            alpaca=AlpacaConfig(
                api_key=api_key,
                secret_key=secret_key,
                base_url=os.getenv(
                    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
                ),
            ),
            trading=TradingConfig(
                max_positions=int(os.getenv("MAX_POSITIONS", "10")),
                risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.10")),
                portfolio_value=float(os.getenv("PORTFOLIO_VALUE", "100000")),
                min_composite_score=float(os.getenv("MIN_COMPOSITE_SCORE", "72.5")),
                weight_technical=float(os.getenv("WEIGHT_TECHNICAL", "0.50")),
                weight_sentiment=float(os.getenv("WEIGHT_SENTIMENT", "0.50")),
                weight_fundamental=float(os.getenv("WEIGHT_FUNDAMENTAL", "0.00")),
            ),
            logging=LoggingConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_file=os.getenv("LOG_FILE", "logs/trading.log"),
            ),
        )


# Global config instance
config = Config.from_env()
