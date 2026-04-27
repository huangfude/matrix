"""Matrix Router module.

Provides high-level Python wrapper for the Rust Router, including
configuration utilities and type conversion helpers.
"""

from matrix_cli.router.launch_router import launch_router, parse_router_args
from matrix_cli.router.router import (
    Router,
    policy_from_str,
    backend_from_str,
    history_backend_from_str,
    role_from_str,
    build_control_plane_auth_config,
)
from matrix_cli.router.router_args import RouterArgs

__all__ = [
    "launch_router",
    "parse_router_args",
    "Router",
    "policy_from_str",
    "backend_from_str",
    "history_backend_from_str",
    "role_from_str",
    "build_control_plane_auth_config",
    "RouterArgs",
]
