"""VLLM launcher module for Matrix CLI.

Provides functionality for launching VLLM server with support for configuration files
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


def map_config_to_vllm_args(config: dict[str, Any]) -> list[str]:
    """Map configuration parameters to VLLM command-line arguments.

    Args:
        config: Configuration dictionary

    Returns:
        List of command-line arguments for VLLM
    """
    # Map common config parameters to VLLM arguments
    mappings = {
        'model': '--model',
        'host': '--host',
        'port': '--port',
        'tensor_parallel_size': '--tensor-parallel-size',
        'pipeline_parallel_size': '--pipeline-parallel-size',
        'dtype': '--dtype',
        'max_model_len': '--max-model-len',
        'gpu_memory_utilization': '--gpu-memory-utilization',
        'max_num_batched_tokens': '--max-num-batched-tokens',
        'max_num_seqs': '--max-num-seqs',
        'block_size': '--block-size',
        'swap_space': '--swap-space',
        'gpu_prompt_percent': '--gpu-prompt-percent',
        'seed': '--seed',
        'disable_log_requests': '--disable-log-requests',
        'max_log_len': '--max-log-len',
        'trust_remote_code': '--trust-remote-code',
        'quantization': '--quantization',
        'enforce_eager': '--enforce-eager',
        'kv_cache_dtype': '--kv-cache-dtype',
        'max_cpu_loras': '--max-cpu-loras',
        'engine_use_ray': '--engine-use-ray',
        'distributed_executor_backend': '--distributed-executor-backend',
        'load_format': '--load-format',
        'download_dir': '--download-dir',
        'enable_lora': '--enable-lora',
        'max_loras': '--max-loras',
        'max_lora_rank': '--max-lora-rank',
        'lora_extra_vocab_size': '--lora-extra-vocab-size',
        'long_lora_scaling_factors': '--long-lora-scaling-factors',
        'max_logprobs': '--max-logprobs',
        'tokenizer_pool_size': '--tokenizer-pool-size',
        'tokenizer_pool_type': '--tokenizer-pool-type',
        'tokenizer_pool_extra_config': '--tokenizer-pool-extra-config',
        'enable_chunked_prefill': '--enable-chunked-prefill',
        'speculative_model': '--speculative-model',
        'num_speculative_tokens': '--num-speculative-tokens',
        'spec_decoding_acceptance_method': '--spec-decoding-acceptance-method',
        'typical_acceptance_sampler_posterior_threshold': '--typical-acceptance-sampler-posterior-threshold',
        'typical_acceptance_sampler_posterior_alpha': '--typical-acceptance-sampler-posterior-alpha',
        'ngram_prompt_lookup_max': '--ngram-prompt-lookup-max',
        'ngram_prompt_lookup_min': '--ngram-prompt-lookup-min',
        'model_loader_extra_config': '--model-loader-extra-config',
        'preemption_mode': '--preemption-mode',
        'ray_workers_use_nsight': '--ray-workers-use-nsight',
        'num_scheduler_steps': '--num-scheduler-steps',
        'multi_step_stream_outputs': '--multi-step-stream-outputs',
        'enable_prefix_caching': '--enable-prefix-caching',
        'use_v2_block_manager': '--use-v2-block-manager',
        'scheduler_delay_factor': '--scheduler-delay-factor',
        'enable_chunked_prefill_token_size': '--chunked-prefill-token-size',
    }

    return map_config_to_args(config, mappings)


def parse_vllm_args(args: list[str]) -> argparse.Namespace:
    """Parse VLLM-specific arguments including config file support.

    Args:
        args: List of command-line arguments

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="matrix vllm",
        description="Launch VLLM server with configuration support",
        add_help=False,  # 禁用帮助以避免与后端帮助冲突
        conflict_handler='resolve'  # 允许此解析器优先解决冲突
    )

    return parse_backend_args(parser, args)


def launch_vllm(args: argparse.Namespace) -> None:
    """Launch VLLM server process.

    Args:
        args: Parsed arguments namespace
    """
    # Build the command for VLLM OpenAI-compatible API server
    #cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    cmd = ["vllm", "serve"]

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        config_args = map_config_to_vllm_args(config)
        cmd.extend(config_args)

    # Add additional arguments from command line (these override config)
    cmd.extend(args.remainder)

    launch_backend(cmd, "VLLM")