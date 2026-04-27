"""
Unit tests for matrix_cli router module.

This module tests the router functionality with mocked matrix_cli.matrix_cli_rs
to avoid Rust dependencies during Python testing.
The mock is configured in conftest.py.
"""

import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "../../src/matrix_cli")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestLaunchRouter:
    """Test launch_router function with mocked Rust Router."""

    def test_launch_router_with_rust_router_mocked(self):
        """Test launch_router when Rust Router is available."""
        from matrix_cli.router.router_args import RouterArgs

        # Import the module fresh and patch it before Router is imported
        import matrix_cli.matrix_cli_rs as mock_matrix_cli_rs

        # Create a proper mock router class with from_args method
        mock_router_class = MagicMock()
        mock_router_instance = MagicMock()
        mock_router_instance.start = MagicMock()
        mock_router_class.from_args = MagicMock(return_value=mock_router_instance)

        # Patch the Router class in matrix_cli.router.launch_router
        with patch('matrix_cli.router.launch_router.Router', mock_router_class):
            from matrix_cli.router.launch_router import launch_router

            args = RouterArgs(
                worker_urls=["http://worker1:8000"],
                host="127.0.0.1",
                port=30000,
            )

            launch_router(args)

            # Verify from_args was called with the args
            mock_router_class.from_args.assert_called_once()
            call_args = mock_router_class.from_args.call_args[0][0]
            assert call_args.worker_urls == ["http://worker1:8000"]
            assert call_args.host == "127.0.0.1"
            assert call_args.port == 30000

            # Verify start was called
            mock_router_instance.start.assert_called_once()

    def test_launch_router_sets_proctitle(self):
        """Test that launch_router sets the process title."""
        from matrix_cli.router.router_args import RouterArgs

        import matrix_cli.matrix_cli_rs as mock_matrix_cli_rs

        mock_router_class = MagicMock()
        mock_router_instance = MagicMock()
        mock_router_instance.start = MagicMock()
        mock_router_class.from_args = MagicMock(return_value=mock_router_instance)

        with patch('matrix_cli.router.launch_router.Router', mock_router_class):
            with patch('matrix_cli.router.launch_router.setproctitle.setproctitle') as mock_setproctitle:
                from matrix_cli.router.launch_router import launch_router

                args = RouterArgs(worker_urls=["http://worker1:8000"])

                launch_router(args)

                # Verify setproctitle was called (with any argument)
                mock_setproctitle.assert_called_once()
                # Verify the call was with the expected argument
                assert mock_setproctitle.call_args[0][0] == "matrix::router"

    def test_launch_router_disaggregated_mode_banner(self):
        """Test banner display for PD disaggregated mode."""
        from matrix_cli.router.launch_router import launch_router
        from matrix_cli.router.router_args import RouterArgs

        import matrix_cli.matrix_cli_rs as mock_matrix_cli_rs

        mock_router_instance = MagicMock()
        mock_router_instance.start = MagicMock()

        mock_matrix_cli_rs.Router.from_args = MagicMock(return_value=mock_router_instance)

        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
        )

        launch_router(args)

        # Verify print_banner was called with mode
        # Note: print_banner may have been called multiple times from other tests
        # So we check that it was called at least once
        assert mock_matrix_cli_rs.print_banner.call_count >= 1

        # Get the last call
        call_args = mock_matrix_cli_rs.print_banner.call_args
        # The call signature is print_banner(host, port, mode)
        assert call_args[0][2] == "PD Disaggregated"

    def test_launch_router_igw_mode_banner(self):
        """Test banner display for IGW mode."""
        from matrix_cli.router.launch_router import launch_router
        from matrix_cli.router.router_args import RouterArgs

        import matrix_cli.matrix_cli_rs as mock_matrix_cli_rs

        mock_router_instance = MagicMock()
        mock_router_instance.start = MagicMock()

        mock_matrix_cli_rs.Router.from_args = MagicMock(return_value=mock_router_instance)

        args = RouterArgs(
            enable_igw=True,
            worker_urls=["http://worker1:8000"],
        )

        launch_router(args)

        call_args = mock_matrix_cli_rs.print_banner.call_args
        assert call_args[0][2] == "IGW"

    def test_launch_router_regular_mode_banner(self):
        """Test banner display for regular mode."""
        from matrix_cli.router.launch_router import launch_router
        from matrix_cli.router.router_args import RouterArgs

        import matrix_cli.matrix_cli_rs as mock_matrix_cli_rs

        mock_router_instance = MagicMock()
        mock_router_instance.start = MagicMock()

        mock_matrix_cli_rs.Router.from_args = MagicMock(return_value=mock_router_instance)

        args = RouterArgs(
            worker_urls=["http://worker1:8000"],
        )

        launch_router(args)

        call_args = mock_matrix_cli_rs.print_banner.call_args
        assert call_args[0][2] == "Regular"


class TestParseRouterArgs:
    """Test parse_router_args function."""

    def test_parse_router_args_with_worker_urls(self):
        """Test parsing arguments with worker URLs."""
        from matrix_cli.router.launch_router import parse_router_args

        args = parse_router_args([
            "--worker-urls", "http://worker1:8000", "http://worker2:8000",
        ])

        assert args.worker_urls == ["http://worker1:8000", "http://worker2:8000"]
        assert args.host == "0.0.0.0"  # default
        assert args.port == 30000  # default

    def test_parse_router_args_with_custom_host_port(self):
        """Test parsing arguments with custom host and port."""
        from matrix_cli.router.launch_router import parse_router_args

        args = parse_router_args([
            "--worker-urls", "http://worker1:8000",
            "--host", "127.0.0.1",
            "--port", "8080",
        ])

        assert args.worker_urls == ["http://worker1:8000"]
        assert args.host == "127.0.0.1"
        assert args.port == 8080

    def test_parse_router_args_with_policy(self):
        """Test parsing arguments with custom policy."""
        from matrix_cli.router.launch_router import parse_router_args

        args = parse_router_args([
            "--worker-urls", "http://worker1:8000",
            "--policy", "round_robin",
        ])

        assert args.policy == "round_robin"

    def test_parse_router_args_with_pd_disaggregation(self):
        """Test parsing arguments for PD disaggregated mode."""
        from matrix_cli.router.launch_router import parse_router_args

        args = parse_router_args([
            "--pd-disaggregation",
            "--prefill", "http://prefill1:8000", "9000",
            "--decode", "http://decode1:8001",
        ])

        assert args.pd_disaggregation is True
        assert len(args.prefill_urls) == 1
        assert args.prefill_urls[0][0] == "http://prefill1:8000"
        # Note: bootstrap_port is parsed as int, not string
        assert args.prefill_urls[0][1] == 9000
        assert args.decode_urls == ["http://decode1:8001"]

    def test_parse_router_args_with_service_discovery(self):
        """Test parsing arguments with service discovery configuration."""
        from matrix_cli.router.launch_router import parse_router_args

        args = parse_router_args([
            "--worker-urls", "http://worker1:8000",
            "--service-discovery",
            "--selector", "app=worker",
            "--service-discovery-port", "8080",
        ])

        assert args.service_discovery is True
        assert args.selector == {"app": "worker"}
        assert args.service_discovery_port == 8080


class TestRouterArgsFromCliArgs:
    """Test RouterArgs.from_cli_args method."""

    def test_from_cli_args_creates_routerargs(self):
        """Test creating RouterArgs from argparse Namespace."""
        from matrix_cli.router.router_args import RouterArgs

        namespace = argparse.Namespace(
            worker_urls=["http://worker1:8000"],
            host="127.0.0.1",
            port=8080,
            policy="round_robin",
            pd_disaggregation=True,
        )

        args = RouterArgs.from_cli_args(namespace)

        assert args.worker_urls == ["http://worker1:8000"]
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.policy == "round_robin"
        assert args.pd_disaggregation is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
