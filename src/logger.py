"""Structured logging for the trading system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

from config.config import config


def setup_logging() -> BoundLogger:
    """Configure structured logging."""

    # Create logs directory
    log_dir = Path(config.logging.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, config.logging.log_level))

    # Add file handler for persistent logs
    file_handler = logging.FileHandler(config.logging.log_file)
    file_handler.setLevel(getattr(logging, config.logging.log_level))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)

    # Add stdout handler for Docker/console output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(getattr(logging, config.logging.log_level))
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(stdout_handler)

    bound_logger: BoundLogger = structlog.get_logger()
    return bound_logger


def log_trade_decision(
    logger: BoundLogger,
    symbol: str,
    action: str,
    reasoning: dict[str, Any],
    score: float,
) -> None:
    """Log a trade decision with full reasoning."""
    logger.info(
        "trade_decision",
        symbol=symbol,
        action=action,
        composite_score=score,
        technical_score=reasoning.get("technical", {}),
        fundamental_score=reasoning.get("fundamental", {}),
        sentiment_score=reasoning.get("sentiment", {}),
        timestamp=datetime.now().isoformat(),
    )


logger = setup_logging()
