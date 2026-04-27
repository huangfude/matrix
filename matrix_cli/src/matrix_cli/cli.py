"""Matrix CLI.

Provides convenient command-line interface for launching the router.

Usage:
    matrix/matrix-cli router [args]          # Launch router
    matrix/matrix-cli --help                 # Show help
"""

import argparse
import sys

from matrix_cli.log import logger
from matrix_cli.matrix_cli_rs import get_version_string, get_verbose_version_string


def print_help() -> None:
    """Print help message."""
    print("""Matrix CLI - High-performance inference router

Usage:
    matrix router [args]  OR  matrix-cli router [args]        # Launch router
    matrix vllm [args]    OR  matrix-cli vllm [args]          # Launch vLLM server
    matrix sglang [args]  OR  matrix-cli sglang [args]        # Launch SGLang server
    matrix --help OR matrix-cli --help                        # Show help
    matrix --version OR matrix-cli --version                  # Show version
""")


def _parse_router_args_with_command(argv: list[str]) -> tuple[str | None, list[str]]:
    """Parse command line arguments to extract command and remaining args.

    Args:
        argv: Command line arguments (without script name)

    Returns:
        Tuple of (command, unknown_args)
    """
    parser = argparse.ArgumentParser(
        prog="matrix-cli",
        description="Matrix CLI - High-performance inference router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["router", "vllm", "sglang"],
        help="Command to run",
    )
    args, unknown = parser.parse_known_args(argv)
    return args.command, unknown


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point.

    Args:
        argv: Command line arguments. Defaults to sys.argv[1:].
    """
    argv = argv or sys.argv[1:]

    # Handle version flags
    if argv and argv[0] in ["--version", "-V", "--version-verbose"]:
        version_func = get_verbose_version_string if argv[0] == "--version-verbose" else get_version_string
        logger.info(version_func())
        sys.exit(0)

    # Check for help flag
    if argv and argv[0] in ["-h", "--help"]:
        print_help()
        sys.exit(0)

    # Handle empty command - show help
    if not argv:
        print_help()
        sys.exit(1)

    # Parse command and extract unknown args
    command, unknown = _parse_router_args_with_command(argv)

    if command == "router":
        from matrix_cli.router.launch_router import launch_router, parse_router_args
        router_args = parse_router_args(unknown)
        launch_router(router_args)

    elif command == "vllm":
        from matrix_cli.vllm.vllm_launcher import launch_vllm, parse_vllm_args
        vllm_args = parse_vllm_args(unknown)
        launch_vllm(vllm_args)

    elif command == "sglang":
        from matrix_cli.sglang.sglang_launcher import launch_sglang, parse_sglang_args
        sglang_args = parse_sglang_args(unknown)
        launch_sglang(sglang_args)

    else:
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
