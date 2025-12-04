"""
Watchdog module for monitoring and recovering from hangs.

Provides:
- Heartbeat tracking to detect stuck processes
- Automatic recovery mechanisms
- Health status file for external monitoring
"""

import json
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logger import logger

# Heartbeat file location
HEARTBEAT_FILE = Path(__file__).parent.parent / "data" / "heartbeat.json"
WATCHDOG_TIMEOUT_SECONDS = 600  # 10 minutes max without heartbeat


class Watchdog:
    """
    Watchdog that monitors the trading bot for hangs and unresponsiveness.

    Features:
    - Heartbeat tracking with timestamps
    - Automatic process restart on hang detection
    - Health file for external monitoring (systemd, Docker, etc.)
    - Configurable timeout thresholds
    """

    def __init__(
        self,
        timeout_seconds: int = WATCHDOG_TIMEOUT_SECONDS,
        check_interval: int = 30,
    ):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.last_heartbeat: datetime = datetime.now()
        self.last_activity: str = "initialized"
        self.is_running = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._restart_callback: Callable[[], None] | None = None
        self._hang_count = 0
        self._max_hangs_before_exit = 3  # Exit after 3 consecutive hangs

        # Ensure data directory exists
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "watchdog_initialized",
            timeout_seconds=timeout_seconds,
            check_interval=check_interval,
        )

    def heartbeat(self, activity: str = "active") -> None:
        """
        Record a heartbeat indicating the bot is still responsive.

        Call this regularly from the main trading loop to prevent
        the watchdog from triggering a restart.

        Args:
            activity: Description of current activity for debugging
        """
        with self._lock:
            self.last_heartbeat = datetime.now()
            self.last_activity = activity
            self._hang_count = 0  # Reset hang count on successful heartbeat

        # Write health file for external monitoring
        self._write_health_file()

    def _write_health_file(self) -> None:
        """Write health status to file for external monitoring."""
        try:
            with self._lock:
                health_data = {
                    "last_heartbeat": self.last_heartbeat.isoformat(),
                    "last_activity": self.last_activity,
                    "timeout_seconds": self.timeout_seconds,
                    "is_healthy": self._is_healthy(),
                    "seconds_since_heartbeat": (
                        datetime.now() - self.last_heartbeat
                    ).total_seconds(),
                    "pid": os.getpid(),
                    "hang_count": self._hang_count,
                }

            with open(HEARTBEAT_FILE, "w") as f:
                json.dump(health_data, f, indent=2)

        except Exception as e:
            logger.warning("failed_to_write_health_file", error=str(e))

    def _is_healthy(self) -> bool:
        """Check if the bot is healthy based on heartbeat timing."""
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < self.timeout_seconds

    def get_status(self) -> dict[str, Any]:
        """Get current watchdog status."""
        with self._lock:
            elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
            return {
                "is_healthy": elapsed < self.timeout_seconds,
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "last_activity": self.last_activity,
                "seconds_since_heartbeat": elapsed,
                "timeout_seconds": self.timeout_seconds,
                "hang_count": self._hang_count,
                "is_monitoring": self.is_running,
            }

    def set_restart_callback(self, callback: Callable[[], None]) -> None:
        """Set a callback to be called when a restart is triggered."""
        self._restart_callback = callback

    def start_monitoring(self) -> None:
        """Start the watchdog monitoring thread."""
        if self.is_running:
            logger.warning("watchdog_already_running")
            return

        self.is_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="WatchdogMonitor",
        )
        self._monitor_thread.start()
        logger.info("watchdog_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop the watchdog monitoring thread."""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("watchdog_monitoring_stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.is_running:
            try:
                time.sleep(self.check_interval)

                with self._lock:
                    elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
                    activity = self.last_activity

                if elapsed >= self.timeout_seconds:
                    self._hang_count += 1
                    logger.error(
                        "watchdog_timeout_detected",
                        elapsed_seconds=elapsed,
                        timeout_seconds=self.timeout_seconds,
                        last_activity=activity,
                        hang_count=self._hang_count,
                    )

                    # Write unhealthy status
                    self._write_health_file()

                    if self._hang_count >= self._max_hangs_before_exit:
                        logger.critical(
                            "watchdog_max_hangs_reached_forcing_exit",
                            hang_count=self._hang_count,
                            max_hangs=self._max_hangs_before_exit,
                        )
                        # Force exit - systemd will restart us
                        os._exit(1)

                    # Try restart callback first
                    if self._restart_callback:
                        try:
                            logger.info("watchdog_triggering_restart_callback")
                            self._restart_callback()
                        except Exception as e:
                            logger.error(
                                "watchdog_restart_callback_failed",
                                error=str(e),
                            )

                elif elapsed >= self.timeout_seconds * 0.75:
                    # Warning at 75% of timeout
                    logger.warning(
                        "watchdog_approaching_timeout",
                        elapsed_seconds=elapsed,
                        timeout_seconds=self.timeout_seconds,
                        last_activity=activity,
                    )

            except Exception as e:
                logger.error("watchdog_monitor_error", error=str(e))


class OperationTimeout:
    """
    Context manager for timing out long-running operations.

    Usage:
        with OperationTimeout(seconds=30, operation="api_call"):
            # Long running operation
            result = api.call()
    """

    def __init__(
        self,
        seconds: int,
        operation: str = "operation",
        raise_on_timeout: bool = True,
    ):
        self.seconds = seconds
        self.operation = operation
        self.raise_on_timeout = raise_on_timeout
        self._timer: threading.Timer | None = None
        self._timed_out = False

    def _timeout_handler(self) -> None:
        """Called when the operation times out."""
        self._timed_out = True
        logger.error(
            "operation_timeout",
            operation=self.operation,
            timeout_seconds=self.seconds,
        )

    def __enter__(self) -> "OperationTimeout":
        self._timer = threading.Timer(self.seconds, self._timeout_handler)
        self._timer.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._timer:
            self._timer.cancel()

        if self._timed_out and self.raise_on_timeout:
            raise TimeoutError(
                f"Operation '{self.operation}' timed out after {self.seconds}s"
            )

        return False

    @property
    def timed_out(self) -> bool:
        """Check if the operation timed out."""
        return self._timed_out


def with_timeout(seconds: int, operation: str = "operation"):
    """
    Decorator to add timeout to a function.

    Usage:
        @with_timeout(30, "fetch_data")
        def fetch_data():
            # Long running operation
            pass
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            completed = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    completed.set()

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()

            if not completed.wait(timeout=seconds):
                logger.error(
                    "function_timeout",
                    function=func.__name__,
                    operation=operation,
                    timeout_seconds=seconds,
                )
                raise TimeoutError(
                    f"Function '{func.__name__}' ({operation}) "
                    f"timed out after {seconds}s"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


def check_health_file() -> dict[str, Any]:
    """
    Read the health file to check bot status.

    Useful for external monitoring scripts.

    Returns:
        Health status dict or error status if file not found/invalid
    """
    try:
        if not HEARTBEAT_FILE.exists():
            return {
                "is_healthy": False,
                "error": "heartbeat_file_not_found",
            }

        with open(HEARTBEAT_FILE) as f:
            data = json.load(f)

        # Validate freshness
        last_heartbeat = datetime.fromisoformat(data["last_heartbeat"])
        elapsed = (datetime.now() - last_heartbeat).total_seconds()

        # Consider unhealthy if no heartbeat in 10 minutes
        data["is_healthy"] = elapsed < 600
        data["seconds_since_heartbeat"] = elapsed

        return data

    except Exception as e:
        return {
            "is_healthy": False,
            "error": str(e),
        }


# Global watchdog instance
watchdog = Watchdog(
    timeout_seconds=WATCHDOG_TIMEOUT_SECONDS,
    check_interval=30,
)
