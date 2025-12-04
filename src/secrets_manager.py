#!/usr/bin/env python3
"""Secrets management for the AI Trading System.

Provides a unified interface for retrieving secrets from multiple backends:
- Environment variables (default)
- Docker secrets
- Kubernetes secrets
- AWS Secrets Manager (optional)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class EnvSecretsBackend(SecretsBackend):
    """Environment variables secrets backend."""

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable."""
        return os.getenv(key)

    def is_available(self) -> bool:
        """Environment variables are always available."""
        return True


class DockerSecretsBackend(SecretsBackend):
    """Docker secrets backend (/run/secrets/)."""

    SECRETS_PATH = Path("/run/secrets")

    def get_secret(self, key: str) -> str | None:
        """Get secret from Docker secrets file."""
        secret_file = self.SECRETS_PATH / key.lower()
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read Docker secret {key}: {e}")
        return None

    def is_available(self) -> bool:
        """Check if Docker secrets directory exists."""
        return self.SECRETS_PATH.exists() and self.SECRETS_PATH.is_dir()


class KubernetesSecretsBackend(SecretsBackend):
    """Kubernetes secrets backend (mounted as files or env vars)."""

    # Default K8s secrets mount path
    SECRETS_PATH = Path("/var/run/secrets/ai-trader")

    def __init__(self, mount_path: str | None = None) -> None:
        """Initialize with optional custom mount path."""
        if mount_path:
            self.SECRETS_PATH = Path(mount_path)

    def get_secret(self, key: str) -> str | None:
        """Get secret from Kubernetes mounted file."""
        secret_file = self.SECRETS_PATH / key.lower()
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read K8s secret {key}: {e}")
        return None

    def is_available(self) -> bool:
        """Check if K8s secrets directory exists."""
        return self.SECRETS_PATH.exists() and self.SECRETS_PATH.is_dir()


class FileSecretsBackend(SecretsBackend):
    """JSON file-based secrets backend for development."""

    def __init__(self, file_path: str | None = None) -> None:
        """Initialize with secrets file path."""
        self.file_path = (
            Path(file_path) if file_path else Path.home() / ".ai_trader_secrets.json"
        )
        self._secrets: dict[str, str] = {}
        self._loaded = False

    def _load_secrets(self) -> None:
        """Load secrets from file."""
        if self._loaded:
            return

        if self.file_path.exists():
            try:
                self._secrets = json.loads(self.file_path.read_text())
                self._loaded = True
            except Exception as e:
                logger.warning(f"Failed to load secrets file: {e}")
                self._secrets = {}

    def get_secret(self, key: str) -> str | None:
        """Get secret from JSON file."""
        self._load_secrets()
        return self._secrets.get(key)

    def is_available(self) -> bool:
        """Check if secrets file exists."""
        return self.file_path.exists()


class SecretsManager:
    """Unified secrets manager with multiple backend support.

    Backends are tried in order of priority:
    1. Kubernetes secrets (if available)
    2. Docker secrets (if available)
    3. Environment variables (always available)
    4. File-based secrets (development fallback)
    """

    def __init__(self) -> None:
        """Initialize secrets manager with available backends."""
        self._backends: list[SecretsBackend] = []
        self._cache: dict[str, str] = {}
        self._setup_backends()

    def _setup_backends(self) -> None:
        """Set up available backends in priority order."""
        # Kubernetes (highest priority in K8s environments)
        k8s_backend = KubernetesSecretsBackend()
        if k8s_backend.is_available():
            self._backends.append(k8s_backend)
            logger.info("Kubernetes secrets backend enabled")

        # Docker secrets
        docker_backend = DockerSecretsBackend()
        if docker_backend.is_available():
            self._backends.append(docker_backend)
            logger.info("Docker secrets backend enabled")

        # Environment variables (always available)
        self._backends.append(EnvSecretsBackend())
        logger.info("Environment variables backend enabled")

        # File-based (development fallback)
        file_backend = FileSecretsBackend()
        if file_backend.is_available():
            self._backends.append(file_backend)
            logger.info("File-based secrets backend enabled")

    def get_secret(
        self,
        key: str,
        default: str | None = None,
        required: bool = False,
        cache: bool = True,
    ) -> str | None:
        """Get a secret value from the first available backend.

        Args:
            key: The secret key to retrieve
            default: Default value if secret not found
            required: If True, raise exception when not found
            cache: Whether to cache the result

        Returns:
            The secret value or default

        Raises:
            ValueError: If required=True and secret not found
        """
        # Check cache first
        if cache and key in self._cache:
            return self._cache[key]

        # Try each backend
        for backend in self._backends:
            value = backend.get_secret(key)
            if value is not None:
                if cache:
                    self._cache[key] = value
                return value

        # Not found in any backend
        if required and default is None:
            raise ValueError(f"Required secret '{key}' not found in any backend")

        return default

    def get_alpaca_credentials(self) -> dict[str, str | bool]:
        """Get Alpaca API credentials."""
        api_key = self.get_secret("ALPACA_API_KEY", required=True)
        secret_key = self.get_secret("ALPACA_SECRET_KEY", required=True)
        paper_str = self.get_secret("ALPACA_PAPER", default="true")
        # All required=True or have defaults, so these are guaranteed non-None
        assert api_key is not None
        assert secret_key is not None
        assert paper_str is not None
        return {
            "api_key": api_key,
            "secret_key": secret_key,
            "paper": paper_str.lower() == "true",
        }

    def get_newsapi_key(self) -> str | None:
        """Get NewsAPI key."""
        return self.get_secret("NEWSAPI_KEY")

    def get_dashboard_credentials(self) -> dict[str, str | None]:
        """Get dashboard authentication credentials."""
        return {
            "username": self.get_secret("DASHBOARD_USERNAME"),
            "password": self.get_secret("DASHBOARD_PASSWORD"),
            "secret_key": self.get_secret("DASHBOARD_SECRET_KEY"),
        }

    def get_ollama_config(self) -> dict[str, str]:
        """Get Ollama LLM configuration."""
        host = self.get_secret("OLLAMA_HOST", default="http://localhost:11434")
        model = self.get_secret("OLLAMA_MODEL", default="llama2")
        # Both have defaults, so guaranteed non-None
        assert host is not None
        assert model is not None
        return {
            "host": host,
            "model": model,
        }

    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._cache.clear()

    def list_backends(self) -> list[str]:
        """List active backends."""
        return [type(b).__name__ for b in self._backends]


# Global instance
secrets_manager = SecretsManager()


# Convenience functions
def get_secret(
    key: str, default: str | None = None, required: bool = False
) -> str | None:
    """Get a secret using the global secrets manager."""
    return secrets_manager.get_secret(key, default=default, required=required)
