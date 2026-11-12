"""Repository-wide pytest safeguards."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import pytest


def _is_loopback_address(sock: socket.socket | None, address: Any) -> bool:
    if sock is not None and sock.family not in (socket.AF_INET, socket.AF_INET6):
        return True
    if not isinstance(address, tuple) or not address:
        return False

    host = str(address[0]).strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every test that attempts a non-loopback network connection."""
    monkeypatch.setenv("OPENAI_API_KEY", "tests-must-not-use-real-credentials")
    monkeypatch.setenv("TOGETHERAI_API_KEY", "tests-must-not-use-real-credentials")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "tests-must-not-use-real-credentials")

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if not _is_loopback_address(sock, address):
            raise AssertionError(f"External network access is disabled in tests: {address!r}")
        return real_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if not _is_loopback_address(sock, address):
            raise AssertionError(f"External network access is disabled in tests: {address!r}")
        return real_connect_ex(sock, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not _is_loopback_address(None, address):
            raise AssertionError(f"External network access is disabled in tests: {address!r}")
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
