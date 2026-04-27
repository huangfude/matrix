"""SGLang launcher module for Matrix CLI.

Provides functionality for launching SGLang server with support for configuration files
and parameter pass-through.
"""

from __future__ import annotations

import argparse
from typing import Any

from matrix_cli.utils import (
    launch_backend,
    load_config,
    map_config_to_args,
    parse_backend_args,
)


def map_config_to_sglang_args(config: dict[str, Any]) -> list[str]:
    """Map configuration parameters to SGLang command-line arguments.

    Args:
        config: Configuration dictionary

    Returns:
        List of command-line arguments for SGLang
    """
    # Map common config parameters to SGLang arguments
    mappings = {
        'model_path': '--model-path',
        'host': '--host',
        'port': '--port',
        'tp': '--tp',  # tensor parallelism
        'mem_fraction_static': '--mem-fraction-static',
        'max_running_requests': '--max-running-requests',
        'max_total_tokens': '--max-total-tokens',
        'schedule_heuristic': '--schedule-heuristic',
        'attention_backend': '--attention-backend',
        'sampling_backend': '--sampling-backend',
        'random_seed': '--random-seed',
        'stream_interval': '--stream-interval',
        'disable_log_stats': '--disable-log-stats',
        'log_level': '--log-level',
        'log_level_http': '--log-level-http',
        'disable_log_requests': '--disable-log-requests',
        'enable_torch_compile': '--enable-torch-compile',
        'enable_memory_pool': '--enable-memory-pool',
        'enable_radix_cache': '--enable-radix-cache',
        'enable_overlap_schedule': '--enable-overlap-schedule',
        'enable_double_sparsity': '--enable-double-sparsity',
        'ds_channel_config_path': '--ds-channel-config-path',
        'ds_channel_config': '--ds-channel-config',
        'ds_mixed_chunks': '--ds-mixed-chunks',
        'json_model_by_weights': '--json-model-by-weights',
        'trust_remote_code': '--trust-remote-code',
        'context_length': '--context-length',
        'model_override_args': '--model-override-args',
        'gate_server_port': '--gate-server-port',
        'token_queue_type': '--token-queue-type',
        'shuffling_scheduler': '--shuffling-scheduler',
        'load_balance_method': '--load-balance-method',
        'random_round_robin': '--random-round-robin',
        'next_batch_sampling_method': '--next-batch-sampling-method',
        'next_batch_gpu_copy_method': '--next-batch-gpu-copy-method',
        'disable_cuda_graph': '--disable-cuda-graph',
        'disable_disk_cache': '--disable-disk-cache',
        'enable_flashinfer': '--enable-flashinfer',
        'enable_p2p_check': '--enable-p2p-check',
        'free_gpu_cache_threshold': '--free-gpu-cache-threshold',
        'cpu_garbage_collect_interval': '--cpu-garbage-collect-interval',
        'disable_fast_attn': '--disable-fast-attn',
        'enable_dp_attention': '--enable-dp-attention',
        'enable_ep_moe': '--enable-ep-moe',
        'enable_mla': '--enable-mla',
        'disable_penalizer': '--disable-penalizer',
        'torchao_config': '--torchao-config',
        'json_config': '--json-config',
        'chunked_prefill_size': '--chunked-prefill-size',
        'cache_report_interval': '--cache-report-interval',
        'skip_tokenizer_init': '--skip-tokenizer-init',
        'enable_low_bit_attn': '--enable-low-bit-attn',
        'low_bit_attn_implementation': '--low-bit-attn-implementation',
        'enable_nan_detection': '--enable-nan-detection',
    }

    return map_config_to_args(config, mappings)


def parse_sglang_args(args: list[str]) -> argparse.Namespace:
    """解析SGLang特定参数,包括配置文件支持。

    Args:
        args: 命令行参数列表

    Returns:
        解析后的参数命名空间,包含:
        - config: 配置文件路径(如果有)
        - remainder: 其他未解析的参数
    """
    # 创建带有冲突处理器的解析器,允许未知参数透传
    parser = argparse.ArgumentParser(
        prog="matrix sglang",
        description="Launch SGLang server with configuration support",
        add_help=False,  # 禁用帮助以避免与SGLang帮助冲突
        conflict_handler='resolve'  # 允许此解析器优先解决冲突
    )

    return parse_backend_args(parser, args)


def launch_sglang(args: argparse.Namespace) -> None:
    """Launch SGLang server process.

    Args:
        args: Parsed arguments namespace
    """
    # SGLang can be launched with different module names depending on the version
    # Try the most common module path for SGLang
    cmd = ["python3", "-m", "sglang.launch_server"]

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        config_args = map_config_to_sglang_args(config)
        cmd.extend(config_args)

    # Add additional arguments from command line (these override config)
    cmd.extend(args.remainder)

    launch_backend(cmd, "SGLang")
