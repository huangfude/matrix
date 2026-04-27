"""Router launch module.

Provides functionality for launching the Matrix Router with various configuration options.
Supports regular mode, PD (Prefill-Decode) disaggregated mode, and IGW (Inference-Gateway) mode.
"""

from __future__ import annotations

import argparse
import os

import setproctitle

from matrix_cli.log import logger
from matrix_cli.router.router_args import RouterArgs
from matrix_cli.matrix_cli_rs import print_banner

try:
    from matrix_cli.router.router import Router
except ImportError:
    Router = None  # type: ignore[assignment,misc]
    logger.warning("Rust Router is not installed")


def launch_router(args: argparse.Namespace | RouterArgs) -> None:
    """Launch the Matrix Router with the configuration from parsed arguments.

    Args:
        args: Namespace object containing router configuration.
            Can be either raw argparse.Namespace or converted RouterArgs.

    Returns:
        None. Raises on failure.

    Raises:
        RuntimeError: If the Rust Router is not installed.
    """
    setproctitle.setproctitle("matrix::router")

    # Convert to RouterArgs if needed
    router_args = args if isinstance(args, RouterArgs) else RouterArgs.from_cli_args(args)

    if Router is None:
        raise RuntimeError("Rust Router is not installed")

    router_args._validate_router_args()

    # Determine mode for banner display
    if getattr(router_args, "pd_disaggregation", False):
        mode = "PD Disaggregated"
    elif getattr(router_args, "enable_igw", False):
        mode = "IGW"
    else:
        mode = "Regular"

    print_banner(router_args.host, router_args.port, mode)

    router = Router.from_args(router_args)
    router.start()


def parse_router_args(args: list[str]) -> RouterArgs:
    """Parse command line arguments and return RouterArgs instance.

    Args:
        args: List of command line arguments.

    Returns:
        RouterArgs instance with parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description="""Matrix Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  python -m matrix_cli.router.launch_router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  python -m matrix_cli.router.launch_router \\
    --prefill http://prefill1:8000 9000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --policy cache_aware

  # PD mode with optional bootstrap ports
  python -m matrix_cli.router.launch_router \\
    --prefill http://prefill1:8000 9000 \\    # With bootstrap port
    --prefill http://prefill2:8000 none \\    # Explicitly no bootstrap port
    --prefill http://prefill3:8000 \\         # Defaults to no bootstrap port
    --decode http://decode1:8001 --decode http://decode2:8001

  # PD mode with different policies for prefill and decode
  python -m matrix_cli.router.launch_router \\
    --prefill http://prefill1:8000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --prefill-policy cache_aware --decode-policy power_of_two
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    RouterArgs.add_cli_args(parser, use_router_prefix=False)
    return RouterArgs.from_cli_args(parser.parse_args(args), use_router_prefix=False)


def main() -> None:
    """Main entry point for router launch."""
    router_args = parse_router_args(os.sys.argv[1:])
    launch_router(router_args)


if __name__ == "__main__":
    main()
