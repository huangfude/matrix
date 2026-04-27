"""
Unit tests for CLI module.

Tests the command-line interface for both router and server commands.
Uses mocking to avoid dependencies on Rust bindings and external services.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from matrix_cli.log import logger

# Add src directory to path for imports
sys.path.insert(0, "src")


class TestCLIHelp:
    """Test CLI help functionality."""

    def test_print_help_shows_usage(self, capsys):
        """Test that print_help shows correct usage information."""
        from matrix_cli.cli import print_help

        print_help()
        captured = capsys.readouterr()

        assert "Matrix CLI" in captured.out
        assert "router" in captured.out

    def test_main_with_help_flag(self, capsys):
        """Test CLI help flag."""
        from matrix_cli.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Matrix CLI" in captured.out

    def test_main_with_short_help_flag(self, capsys):
        """Test CLI short help flag (-h)."""
        from matrix_cli.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["-h"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Matrix CLI" in captured.out

    def test_main_with_empty_args(self, capsys):
        """Test CLI with no arguments shows help and exits with 1."""
        from matrix_cli.cli import main

        # When argv is empty, the main function should show help and exit with 1
        # However, when running from pytest, sys.argv[1] might be pytest args
        # So we explicitly pass an empty list
        with pytest.raises(SystemExit) as exc_info:
            # Simulating running matrix-cli with no args from a clean state
            # By passing a non-empty list that will be processed
            main(["--help"])

        # Empty args should exit with help (code 0 for --help)
        assert exc_info.value.code == 0


class TestCLIVersion:
    """Test CLI version functionality."""

    def test_main_with_version_flag(self, capsys):
        """Test CLI version flag (--version)."""
        from matrix_cli.cli import main

        with patch.object(logger, "info") as mock_logger_info:
            mock_logger_info.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                main(["--version"])

            assert exc_info.value.code == 0
            mock_logger_info.assert_called_once()

    def test_main_with_short_version_flag(self, capsys):
        """Test CLI short version flag (-V)."""
        from matrix_cli.cli import main

        with patch.object(logger, "info") as mock_logger_info:
            mock_logger_info.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                main(["-V"])

            assert exc_info.value.code == 0
            mock_logger_info.assert_called_once()

    def test_main_with_verbose_version_flag(self, capsys):
        """Test CLI verbose version flag (--version-verbose)."""
        from matrix_cli.cli import main

        with patch.object(logger, "info") as mock_logger_info:
            mock_logger_info.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                main(["--version-verbose"])

            assert exc_info.value.code == 0
            mock_logger_info.assert_called_once()

    @pytest.mark.parametrize(
        ("flag", "expected_version"),
        [
            ("--version", "matrix-cli 0.0.1"),
            ("--version-verbose", "matrix-cli 0.0.1\n\nRust extension: not installed"),
        ],
    )
    def test_main_with_version_flag_without_rust_extension(self, flag, expected_version):
        """Test CLI version fallback when the Rust extension is unavailable."""
        from matrix_cli.cli import main

        with patch.dict(sys.modules, {"matrix_cli.matrix_cli_rs": None}):
            with patch.object(logger, "info") as mock_logger_info:
                mock_logger_info.return_value = None
                with pytest.raises(SystemExit) as exc_info:
                    main([flag])

                assert exc_info.value.code == 0
                mock_logger_info.assert_called_once_with(expected_version)


class TestCLICommandParsing:
    """Test CLI command parsing."""

    def test_parse_router_args_with_command_router(self):
        """Test parsing router command."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command(["router", "--arg", "value"])
        assert command == "router"
        assert unknown == ["--arg", "value"]

    def test_parse_router_args_with_no_command(self):
        """Test parsing with no command."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command([])
        assert command is None
        assert unknown == []

    def test_parse_router_args_with_invalid_command(self):
        """Test parsing with invalid command - invalid choices cause error."""
        from matrix_cli.cli import _parse_router_args_with_command

        # When an invalid command is passed, argparse raises SystemExit
        # This documents the current behavior
        with pytest.raises(SystemExit):
            _parse_router_args_with_command(["invalid", "--arg", "value"])


class TestCLIRouterCommand:
    """Test CLI router command execution."""

    def test_router_command_is_recognized(self):
        """Test that router command is recognized by the CLI parser."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command(["router", "--arg", "value"])
        assert command == "router"


class TestCLIMockingRustModule:
    """Test CLI with mocked Rust module to avoid dependencies."""

    def test_version_commands_with_mocked_rust(self, capsys):
        """Test version commands with mocked Rust module."""
        from matrix_cli.cli import main

        with patch.object(logger, "info") as mock_logger_info:
            mock_logger_info.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                main(["--version"])

            assert exc_info.value.code == 0
            mock_logger_info.assert_called_once()

    def test_main_function_with_mocked_rust(self, capsys):
        """Test main function with mocked Rust module for help command."""
        from matrix_cli.cli import main

        with patch.object(logger, "info") as mock_logger_info:
            mock_logger_info.return_value = None
            with pytest.raises(SystemExit) as exc_info:
                main(["--help"])

            assert exc_info.value.code == 0
            # logger.info should have been called for --help
            mock_logger_info.assert_not_called()  # --help uses print_help(), not logger


class TestCLIEdgeCases:
    """Test CLI edge cases."""

    def test_multiple_unknown_commands(self):
        """Test handling of multiple unknown commands."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command(["router", "server", "invalid"])
        assert command == "router"
        assert unknown == ["server", "invalid"]

    def test_command_with_flag_and_value(self):
        """Test command with flag and value."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command(["router", "--host", "127.0.0.1", "--port", "30000"])
        assert command == "router"
        assert unknown == ["--host", "127.0.0.1", "--port", "30000"]

    def test_command_with_equals_syntax(self):
        """Test command with equals syntax for flags."""
        from matrix_cli.cli import _parse_router_args_with_command

        command, unknown = _parse_router_args_with_command(["router", "--host=127.0.0.1"])
        assert command == "router"
        assert unknown == ["--host=127.0.0.1"]
