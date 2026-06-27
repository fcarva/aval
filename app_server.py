"""Local web server entrypoint for the AVAL MVP.

The entrypoint is factored into small, importable helpers so the host/port
resolution can be unit-tested without ever binding a socket. Configuration is
layered: explicit CLI flags win, then ``AVAL_HOST`` / ``AVAL_PORT`` environment
variables, then the local-only defaults. Running ``python app_server.py`` with no
arguments keeps the original behaviour (127.0.0.1:8000, no reload).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Callable

import uvicorn

APP_PATH = "api:app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def resolve_host(cli_host: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Resolve the bind host: CLI flag, then AVAL_HOST, then the default."""
    environ = os.environ if env is None else env
    if cli_host:
        return cli_host
    return environ.get("AVAL_HOST") or DEFAULT_HOST


def resolve_port(cli_port: int | None = None, env: Mapping[str, str] | None = None) -> int:
    """Resolve the bind port: CLI flag, then AVAL_PORT, then the default."""
    environ = os.environ if env is None else env
    if cli_port is not None:
        port = cli_port
    else:
        raw = environ.get("AVAL_PORT")
        try:
            port = int(raw) if raw not in (None, "") else DEFAULT_PORT
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AVAL_PORT must be an integer, got {raw!r}.") from exc
    if not 0 < port < 65536:
        raise ValueError(f"port must be in 1..65535, got {port}.")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AVAL MVP web server.")
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host (default: AVAL_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: AVAL_PORT or 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload for local development.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    runner: Callable[..., object] = uvicorn.run,
) -> None:
    args = parse_args(argv)
    runner(
        APP_PATH,
        host=resolve_host(args.host),
        port=resolve_port(args.port),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
