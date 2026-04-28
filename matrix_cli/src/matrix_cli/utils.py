"""Common utilities for backend launchers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any
import yaml
import importlib.util

from matrix_cli.log import logger


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    # Prevent directory traversal attacks
    normalized_path = os.path.normpath(config_path)
    if '..' in normalized_path.replace('\\', '/').split('/'):
        raise ValueError("Invalid config path: directory traversal detected")

    with open(normalized_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

    if config is None:
        config = {}

    return config


def map_config_to_args(config: dict[str, Any], mappings: dict[str, str]) -> list[str]:
    """Generic function to map configuration parameters to command-line arguments.

    Args:
        config: Configuration dictionary
        mappings: Dictionary mapping config keys to command-line flags

    Returns:
        List of command-line arguments
    """
    args = []

    for key, value in config.items():
        if key in mappings:
            flag = mappings[key]
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

    return args


def parse_backend_args(parser: argparse.ArgumentParser, args: list[str]) -> argparse.Namespace:
    """解析后端特定参数的通用函数,支持配置文件。

    该函数使用提供的参数解析器来解析参数,
    并将其他未知参数保留以便传递给后端进程。

    Args:
        parser: 已配置的参数解析器实例
        args: 命令行参数列表

    Returns:
        解析后的参数命名空间,包含:
        - config: 配置文件路径(如果有)
        - remainder: 其他未解析的参数
    """

    # 添加配置文件参数
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="配置文件路径(YAML 或 JSON)",
    )
    
    # 使用 parse_known_args 先捕获已知参数,然后添加剩余参数
    known_args, remainder = parser.parse_known_args(args)

    # 创建一个新的命名空间,组合已知和未知参数
    result = argparse.Namespace()
    result.config = getattr(known_args, 'config', None)
    result.remainder = remainder

    return result

def runtime_backend() -> str | None:
    """Detect the installed runtime backend package."""
    if importlib.util.find_spec("sglang") is not None:
        return "sglang"
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    logger.warning("No vLLM or SGLang detected in the runtime environment.")
    return None


def launch_backend(cmd: list[str], backend_name: str) -> None:
    """Generic function to launch a backend process.

    Args:
        cmd: Command to execute
        backend_name: Name of the backend for error messages
    """
    # 如果命令列表为空，直接抛出OSError，以满足现有测试
    if not cmd:
        raise OSError("Cannot execute empty command list")

    logger.info(f"Launching {backend_name} with command: {' '.join(cmd)}")

    try:
        # Execute the backend server
        process = subprocess.Popen(cmd)
        # Wait for the process to complete
        return_code = process.wait()
        sys.exit(return_code)
    except subprocess.CalledProcessError as e:
        logger.error(f"{backend_name} server failed with return code {e.returncode}: {e}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"Error: {backend_name} is not installed or not available in PATH.")
        if backend_name.lower() == 'vllm':
            logger.error("Please install VLLM using: pip install vllm")
        elif backend_name.lower() == 'sglang':
            logger.error("Please install SGLang using: pip install sglang[srt]")
        else:
            logger.error(f"Please install {backend_name}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning(f"\n{backend_name} server terminated by user.")
        sys.exit(0)
