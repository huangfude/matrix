"""
Pytest configuration for CLI tests.

These tests mock the Rust module to avoid dependencies.
"""

import os
import sys
from unittest.mock import MagicMock

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def pytest_configure(config):
    """Configure pytest markers and mock Rust module."""
    config.addinivalue_line("markers", "unit: mark test as a unit test (no GPU required)")

    # Add src directory to path for imports
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    # Create Router mock class
    MockRouter = MagicMock()

    # Mock the Rust module before any module imports it
    rust_mock = MagicMock()

    # Basic functions
    rust_mock.get_version_string = MagicMock(return_value="matrix-cli/1.0.0")
    rust_mock.get_verbose_version_string = MagicMock(return_value="matrix-cli/1.0.0 (verbose)")
    rust_mock.print_banner = MagicMock()
    rust_mock.get_available_tool_call_parsers = MagicMock(return_value=[])

    # Enum types - these need to be proper classes for isinstance() checks
    class MockEnum:
        """Base class for mock enums."""
        def __new__(cls, value):
            obj = object.__new__(cls)
            obj._value_ = value
            return obj

        def __str__(self):
            return self._value_

        def __repr__(self):
            return f"{self.__class__.__name__}.{self._value_}"

    class PolicyType(MockEnum):
        Random = "Random"
        RoundRobin = "RoundRobin"
        CacheAware = "CacheAware"
        PowerOfTwo = "PowerOfTwo"
        Bucket = "Bucket"
        Manual = "Manual"
        ConsistentHashing = "ConsistentHashing"
        PrefixHash = "PrefixHash"

    class BackendType(MockEnum):
        Sglang = "Sglang"
        Openai = "Openai"
        Anthropic = "Anthropic"
        Vllm = "Vllm"

    class HistoryBackendType(MockEnum):
        Memory = "Memory"
        None_ = "None"
        Oracle = "Oracle"
        Postgres = "Postgres"
        Redis = "Redis"

    # Use setattr to set the alias since None is a keyword
    setattr(HistoryBackendType, "None", HistoryBackendType.None_)

    class PyRole(MockEnum):
        Admin = "Admin"
        User = "User"

    rust_mock.PolicyType = PolicyType
    rust_mock.BackendType = BackendType
    rust_mock.HistoryBackendType = HistoryBackendType
    rust_mock.PyRole = PyRole

    # Config types
    rust_mock.PyApiKeyEntry = MagicMock()
    rust_mock.PyJwtConfig = MagicMock()
    rust_mock.PyControlPlaneAuthConfig = MagicMock()

    # Router class (required by router.py)
    rust_mock.Router = MockRouter

    sys.modules["matrix_cli.matrix_cli_rs"] = rust_mock

    # Also mock setproctitle if not available
    try:
        import setproctitle
    except ImportError:
        setproctitle_mock = MagicMock()
        setproctitle_mock.setproctitle = MagicMock()
        sys.modules["setproctitle"] = setproctitle_mock
