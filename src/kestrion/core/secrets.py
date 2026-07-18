"""
Secret Management Protocol.
Provides a secure way to inject API keys and credentials into tools at runtime
without persisting them in the durable CheckpointStore.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class SecretProvider(Protocol):
    """
    Protocol for resolving secrets dynamically.
    Implementations could wrap AWS Secrets Manager, HashiCorp Vault, etc.
    """

    async def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key. Returns None if not found."""
        ...


class EnvVarSecretProvider:
    """
    Default zero-dependency implementation that resolves secrets from
    environment variables.
    """

    async def get_secret(self, key: str) -> str | None:
        return os.environ.get(key)
