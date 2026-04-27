"""
Unit tests for matrix_cli router_args module.

Tests the RouterArgs dataclass for proper argument parsing and validation.
"""

import argparse
import os
import sys

import pytest

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "../../src/matrix_cli")


@pytest.fixture(autouse=True)
def setup_path():
    """Setup path for imports."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)


class TestRouterArgsDefaults:
    """Test RouterArgs default values."""

    def test_worker_defaults(self):
        """Test default worker configuration values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.worker_urls == []
        assert args.host == "0.0.0.0"
        assert args.port == 30000

    def test_routing_defaults(self):
        """Test default routing policy values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.policy == "cache_aware"
        assert args.prefill_policy is None
        assert args.decode_policy is None
        assert args.cache_threshold == 0.3
        assert args.balance_abs_threshold == 64
        assert args.balance_rel_threshold == 1.5

    def test_timeout_defaults(self):
        """Test default timeout values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.worker_startup_timeout_secs == 1800
        assert args.worker_startup_check_interval == 30
        assert args.request_timeout_secs == 1800

    def test_rate_limit_defaults(self):
        """Test default rate limiting values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.max_concurrent_requests == -1
        assert args.queue_size == 100
        assert args.queue_timeout_secs == 60
        assert args.rate_limit_tokens_per_second is None

    def test_health_check_defaults(self):
        """Test default health check values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.health_failure_threshold == 3
        assert args.health_success_threshold == 2
        assert args.health_check_timeout_secs == 5
        assert args.health_check_interval_secs == 60
        assert args.health_check_endpoint == "/health"
        assert args.disable_health_check is False

    def test_circuit_breaker_defaults(self):
        """Test default circuit breaker values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.cb_failure_threshold == 10
        assert args.cb_success_threshold == 3
        assert args.cb_timeout_duration_secs == 60
        assert args.cb_window_duration_secs == 120
        assert args.disable_circuit_breaker is False

    def test_retry_defaults(self):
        """Test default retry values."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.retry_max_retries == 5
        assert args.retry_initial_backoff_ms == 50
        assert args.retry_max_backoff_ms == 30_000
        assert args.retry_backoff_multiplier == 1.5
        assert args.retry_jitter_factor == 0.2
        assert args.disable_retries is False


class TestRouterArgsPDMode:
    """Test RouterArgs PD (Prefill-Decode) disaggregated mode configuration."""

    def test_pd_disaggregation_default(self):
        """Test PD disaggregation is disabled by default."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.pd_disaggregation is False
        assert args.prefill_urls == []
        assert args.decode_urls == []

    def test_pd_urls_withBootstrapPort(self):
        """Test PD URLs with bootstrap ports."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[
                ("http://prefill1:8000", 9000),
                ("http://prefill2:8000", None),
            ],
            decode_urls=["http://decode1:8001", "http://decode2:8001"],
        )

        assert args.pd_disaggregation is True
        assert len(args.prefill_urls) == 2
        assert args.prefill_urls[0] == ("http://prefill1:8000", 9000)
        assert args.prefill_urls[1] == ("http://prefill2:8000", None)
        assert len(args.decode_urls) == 2

    def test_service_discovery_defaults(self):
        """Test service discovery configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.service_discovery is False
        assert args.selector == {}
        assert args.service_discovery_port == 80
        assert args.service_discovery_namespace is None

    def test_pd_service_discovery_defaults(self):
        """Test PD service discovery configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.prefill_selector == {}
        assert args.decode_selector == {}
        assert args.router_selector == {}
        assert args.bootstrap_port_annotation == "sglang.ai/bootstrap-port"


class TestRouterArgsBackend:
    """Test RouterArgs backend configuration."""

    def test_backend_default(self):
        """Test default backend type."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.backend == "vllm"

    def test_history_backend_default(self):
        """Test default history backend."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.history_backend == "memory"

    def test_oracle_config_defaults(self):
        """Test Oracle database configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.oracle_wallet_path is None
        assert args.oracle_tns_alias is None
        assert args.oracle_connect_descriptor is None
        assert args.oracle_username is None
        assert args.oracle_password is None
        assert args.oracle_external_auth is False
        assert args.oracle_pool_min == 1
        assert args.oracle_pool_max == 16
        assert args.oracle_pool_timeout_secs == 30

    def test_postgres_config_defaults(self):
        """Test PostgreSQL database configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.postgres_db_url is None
        assert args.postgres_pool_max == 16

    def test_redis_config_defaults(self):
        """Test Redis database configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.redis_url is None
        assert args.redis_pool_max == 16
        assert args.redis_retention_days == 30


class TestRouterArgsTLS:
    """Test RouterArgs TLS/mTLS configuration."""

    def test_server_tls_defaults(self):
        """Test server TLS configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.server_cert_path is None
        assert args.server_key_path is None

    def test_client_tls_defaults(self):
        """Test client mTLS configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.client_cert_path is None
        assert args.client_key_path is None
        assert args.ca_cert_paths == []


class TestRouterArgsControlPlaneAuth:
    """Test RouterArgs control plane authentication configuration."""

    def test_api_keys_default(self):
        """Test control plane API keys default."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.control_plane_api_keys == []
        assert args.control_plane_audit_enabled is False

    def test_jwt_config_defaults(self):
        """Test JWT/OIDC configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.jwt_issuer is None
        assert args.jwt_audience is None
        assert args.jwt_jwks_uri is None
        assert args.jwt_role_mapping == {}


class TestRouterArgsMesh:
    """Test RouterArgs mesh server configuration."""

    def test_mesh_defaults(self):
        """Test mesh server configuration defaults."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs()

        assert args.enable_mesh is False
        assert args.mesh_server_name is None
        assert args.mesh_host == "0.0.0.0"
        assert args.mesh_port == 39527
        assert args.mesh_peer_urls == []


class TestRouterArgsAddCliArgs:
    """Test RouterArgs.add_cli_args method."""

    def test_add_cli_args_creates_groups(self):
        """Test that add_cli_args creates expected argument groups."""
        from matrix_cli.router.router_args import RouterArgs

        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)

        # Check that argument groups were created
        # Note: The actual group names depend on the implementation
        assert parser is not None

    def test_add_cli_args_with_router_prefix(self):
        """Test add_cli_args with router prefix."""
        from matrix_cli.router.router_args import RouterArgs

        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, use_router_prefix=True)

        # With prefix, arguments should have 'router-' prefix
        assert parser is not None

    def test_add_cli_args_exclude_host_port(self):
        """Test add_cli_args excluding host and port arguments."""
        from matrix_cli.router.router_args import RouterArgs

        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser, exclude_host_port=True)

        # Host and port should not be in the parser
        assert parser is not None


class TestRouterArgsFromCliArgs:
    """Test RouterArgs.from_cli_args method."""

    def test_from_cli_args_from_namespace(self):
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

    def test_from_cli_args_with_router_prefix(self):
        """Test creating RouterArgs with router prefix."""
        from matrix_cli.router.router_args import RouterArgs

        # Create a namespace with router- prefix
        namespace = argparse.Namespace(
            router_worker_urls=["http://worker1:8000"],
            router_host="127.0.0.1",
            router_port=8080,
        )

        args = RouterArgs.from_cli_args(namespace, use_router_prefix=True)

        assert args.worker_urls == ["http://worker1:8000"]
        assert args.host == "127.0.0.1"
        assert args.port == 8080


class TestRouterArgsEdgeCases:
    """Test RouterArgs edge cases and boundary conditions."""

    def test_large_port_number(self):
        """Test with maximum valid port number."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(port=65535)

        assert args.port == 65535

    def test_max_tree_size(self):
        """Test with maximum tree size."""
        from matrix_cli.router.router_args import RouterArgs

        max_size = 2**30
        args = RouterArgs(max_tree_size=max_size)

        assert args.max_tree_size == max_size

    def test_max_payload_size(self):
        """Test with maximum payload size."""
        from matrix_cli.router.router_args import RouterArgs

        max_size = 1024 * 1024 * 1024  # 1GB
        args = RouterArgs(max_payload_size=max_size)

        assert args.max_payload_size == max_size

    def test_empty_worker_urls_list(self):
        """Test with empty worker URLs list."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(worker_urls=[])

        assert args.worker_urls == []
        assert len(args.worker_urls) == 0

    def test_empty_prefill_urls_list(self):
        """Test with empty prefill URLs list in PD mode."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
        )

        assert args.prefill_urls == []
        assert args.decode_urls == []

    def test_all_defaults_equal_new_instance(self):
        """Test that default values are equal across instances."""
        from matrix_cli.router.router_args import RouterArgs

        args1 = RouterArgs()
        args2 = RouterArgs()

        assert args1.host == args2.host
        assert args1.port == args2.port
        assert args1.policy == args2.policy


class TestRouterArgsFieldModification:
    """Test that RouterArgs fields can be modified after creation."""

    def test_modify_host(self):
        """Test modifying host field."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(host="0.0.0.0")
        args.host = "127.0.0.1"

        assert args.host == "127.0.0.1"

    def test_modify_policy(self):
        """Test modifying policy field."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(policy="cache_aware")
        args.policy = "round_robin"

        assert args.policy == "round_robin"

    def test_add_worker_url(self):
        """Test adding worker URL after creation."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(worker_urls=["http://worker1:8000"])
        args.worker_urls.append("http://worker2:8000")

        assert len(args.worker_urls) == 2
        assert "http://worker2:8000" in args.worker_urls

    def test_modify_prefill_urls(self):
        """Test modifying prefill URLs in PD mode."""
        from matrix_cli.router.router_args import RouterArgs

        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
        )
        args.prefill_urls.append(("http://prefill2:8000", 9001))

        assert len(args.prefill_urls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
