"""
单元测试：utils.py 工具函数

测试配置加载、参数映射和后端启动功能。
使用 mock 和 fixture 避免依赖外部进程和文件系统。
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml

# 添加 src 目录到路径以便导入
sys.path.insert(0, "src")


class TestLoadConfig:
    """测试 load_config 函数：从 YAML 或 JSON 文件加载配置。"""

    @pytest.fixture
    def yaml_config_content(self):
        """提供 YAML 配置内容。"""
        return {
            "host": "127.0.0.1",
            "port": 8000,
            "model": "qwen3",
            "enable_cache": True
        }

    @pytest.fixture
    def json_config_content(self):
        """提供 JSON 配置内容。"""
        return {
            "backend": "vllm",
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.9
        }

    @pytest.fixture
    def temp_yaml_file(self, tmp_path, yaml_config_content):
        """创建临时 YAML 配置文件。"""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config_content, f)
        return str(config_file)

    @pytest.fixture
    def temp_yml_file(self, tmp_path, yaml_config_content):
        """创建临时 .yml 配置文件。"""
        config_file = tmp_path / "config.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config_content, f)
        return str(config_file)

    @pytest.fixture
    def temp_json_file(self, tmp_path, json_config_content):
        """创建临时 JSON 配置文件。"""
        config_file = tmp_path / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(json_config_content, f)
        return str(config_file)

    @pytest.fixture
    def temp_empty_yaml_file(self, tmp_path):
        """创建空的 YAML 文件。"""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding='utf-8')
        return str(config_file)

    def test_load_yaml_config_success(self, temp_yaml_file, yaml_config_content):
        """测试成功加载 YAML 配置文件。"""
        from matrix_cli.utils import load_config

        config = load_config(temp_yaml_file)
        assert config == yaml_config_content
        assert isinstance(config, dict)

    def test_load_yml_extension_success(self, temp_yml_file, yaml_config_content):
        """测试成功加载 .yml 扩展名的配置文件。"""
        from matrix_cli.utils import load_config

        config = load_config(temp_yml_file)
        assert config == yaml_config_content

    def test_load_json_config_success(self, temp_json_file, json_config_content):
        """测试成功加载 JSON 配置文件。"""
        from matrix_cli.utils import load_config

        config = load_config(temp_json_file)
        assert config == json_config_content

    def test_load_empty_yaml_returns_empty_dict(self, temp_empty_yaml_file):
        """测试加载空 YAML 文件返回空字典。"""
        from matrix_cli.utils import load_config

        config = load_config(temp_empty_yaml_file)
        assert config == {}
        assert isinstance(config, dict)

    def test_load_unsupported_format_raises_error(self, tmp_path):
        """测试加载不支持的文件格式时抛出 ValueError。"""
        from matrix_cli.utils import load_config

        unsupported_file = tmp_path / "config.txt"
        unsupported_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(str(unsupported_file))

    def test_load_config_with_directory_traversal_attack(self, tmp_path):
        """测试目录遍历攻击防护。"""
        from matrix_cli.utils import load_config

        # 测试各种目录遍历尝试
        malicious_paths = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "./../../../secrets",
            "config/../../hidden",
        ]

        for path in malicious_paths:
            with pytest.raises(ValueError, match="directory traversal detected"):
                load_config(path)

    def test_load_config_with_invalid_json(self, tmp_path):
        """测试加载无效 JSON 文件时抛出适当错误"""
        from matrix_cli.utils import load_config

        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text("{ invalid json }", encoding='utf-8')

        with pytest.raises(json.JSONDecodeError):
            load_config(str(invalid_json_file))

    def test_load_config_with_invalid_yaml(self, tmp_path):
        """测试加载无效 YAML 文件时抛出适当错误"""
        from matrix_cli.utils import load_config

        invalid_yaml_file = tmp_path / "invalid.yaml"
        invalid_yaml_file.write_text("key:\n  - item1\n  -\n  item2", encoding='utf-8')

        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_yaml_file))

    def test_load_config_with_windows_path_traversal(self, tmp_path):
        """测试 Windows 路径的目录遍历防护。"""
        from matrix_cli.utils import load_config

        # Windows 特定的路径遍历尝试
        with pytest.raises(ValueError, match="directory traversal detected"):
            load_config("..\\..\\config.yaml")

    def test_load_config_with_nonexistent_file(self):
        """测试加载不存在的文件时抛出 FileNotFoundError。"""
        from matrix_cli.utils import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/config.yaml")

    def test_load_yaml_with_complex_structure(self, tmp_path):
        """测试加载包含复杂嵌套结构的 YAML 文件。"""
        from matrix_cli.utils import load_config

        complex_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8888,
                "tls": {
                    "enabled": True,
                    "cert_path": "/path/to/cert.pem"
                }
            },
            "models": [
                {"name": "qwen3", "backend": "vllm"},
                {"name": "glm4", "backend": "sglang"}
            ],
            "settings": {
                "cache": {"enabled": True, "ttl": 3600},
                "rate_limits": {"requests_per_minute": 100}
            }
        }

        config_file = tmp_path / "complex.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(complex_config, f)

        config = load_config(str(config_file))
        assert config == complex_config


class TestMapConfigToArgs:
    """测试 map_config_to_args 函数：将配置映射为命令行参数。"""

    def test_map_simple_string_values(self):
        """测试映射简单的字符串值。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "model": "qwen3",
            "host": "127.0.0.1",
            "port": "8000"
        }
        mappings = {
            "model": "--model",
            "host": "--host",
            "port": "--port"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--model", "qwen3", "--host", "127.0.0.1", "--port", "8000"]

    def test_map_integer_values_converted_to_string(self):
        """测试整数值被转换为字符串。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "tensor_parallel_size": 4,
            "port": 30000
        }
        mappings = {
            "tensor_parallel_size": "--tp",
            "port": "--port"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--tp", "4", "--port", "30000"]
        assert all(isinstance(arg, str) for arg in args)

    def test_map_float_values_converted_to_string(self):
        """测试浮点数值被转换为字符串。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "gpu_memory_utilization": 0.9,
            "temperature": 0.7
        }
        mappings = {
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "temperature": "--temperature"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--gpu-memory-utilization", "0.9", "--temperature", "0.7"]

    def test_map_boolean_true_adds_flag(self):
        """测试布尔值 true 时添加标志。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "enable_cache": True,
            "use_ssl": True
        }
        mappings = {
            "enable_cache": "--cache-enabled",
            "use_ssl": "--ssl"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--cache-enabled", "--ssl"]

    def test_map_boolean_false_omits_flag(self):
        """测试布尔值 false 时不添加标志。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "enable_cache": False,
            "use_ssl": False
        }
        mappings = {
            "enable_cache": "--cache-enabled",
            "use_ssl": "--ssl"
        }

        args = map_config_to_args(config, mappings)
        assert args == []

    def test_map_mixed_types(self):
        """测试混合类型的配置。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "model": "qwen3",
            "port": 8000,
            "enable_cache": True,
            "debug_mode": False,
            "temperature": 0.7
        }
        mappings = {
            "model": "--model",
            "port": "--port",
            "enable_cache": "--cache",
            "debug_mode": "--debug",
            "temperature": "--temp"
        }

        args = map_config_to_args(config, mappings)
        # 注意：False 的标志应该被省略
        assert args == ["--model", "qwen3", "--port", "8000", "--cache", "--temp", "0.7"]

    def test_map_ignores_unmapped_keys(self):
        """测试忽略未在映射中定义的键。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "model": "qwen3",
            "unmapped_key": "should_be_ignored",
            "another_unmapped": 123
        }
        mappings = {
            "model": "--model"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--model", "qwen3"]

    def test_map_empty_config_returns_empty_list(self):
        """测试空配置返回空列表。"""
        from matrix_cli.utils import map_config_to_args

        config = {}
        mappings = {"model": "--model"}

        args = map_config_to_args(config, mappings)
        assert args == []

    def test_map_preserves_order(self):
        """测试保持配置键的顺序。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "first": "value1",
            "second": "value2",
            "third": "value3"
        }
        mappings = {
            "first": "--first",
            "second": "--second",
            "third": "--third"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--first", "value1", "--second", "value2", "--third", "value3"]

    def test_map_with_none_values_converted_to_string(self):
        """测试值为 None 时转换为字符串 'None'。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "model": "qwen3",
            "port": None
        }
        mappings = {
            "model": "--model",
            "port": "--port"
        }

        args = map_config_to_args(config, mappings)
        # None 值会被转换为字符串 "None"
        assert args == ["--model", "qwen3", "--port", "None"]

    def test_map_with_none_values_filtered_out_if_needed(self):
        """测试需要过滤 None 值的情况。"""
        from matrix_cli.utils import map_config_to_args

        # 在过滤掉 None 值的情况下
        config = {
            "model": "qwen3",
            "port": None
        }
        mappings = {
            "model": "--model",
            "port": "--port"
        }

        # 先过滤 None 值再映射
        filtered_config = {k: v for k, v in config.items() if v is not None}
        args = map_config_to_args(filtered_config, mappings)
        assert args == ["--model", "qwen3"]

    def test_map_with_list_values_converted_to_string(self):
        """测试列表值被转换为字符串。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "allowed_models": ["qwen3", "glm4"],
            "ports": [8000, 9000]
        }
        mappings = {
            "allowed_models": "--models",
            "ports": "--ports"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--models", "['qwen3', 'glm4']", "--ports", "[8000, 9000]"]


class TestLaunchBackend:
    """测试 launch_backend 函数：启动后端进程。"""

    @pytest.fixture
    def sample_cmd(self):
        """示例命令。"""
        return ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]

    @pytest.fixture
    def sample_backend_name(self):
        """示例后端名称。"""
        return "vllm"

    @patch('matrix_cli.utils.subprocess.Popen')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_success(self, mock_exit, mock_print, mock_popen, sample_cmd, sample_backend_name):
        """测试成功启动后端进程。"""
        from matrix_cli.utils import launch_backend

        # 模拟进程成功完成（返回码 0）
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        launch_backend(sample_cmd, sample_backend_name)

        # 验证调用了正确的命令
        mock_popen.assert_called_once_with(sample_cmd)
        mock_process.wait.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch('matrix_cli.utils.subprocess.Popen')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_with_nonzero_exit_code(self, mock_exit, mock_print, mock_popen, sample_cmd):
        """测试后端进程以非零退出码终止。"""
        from matrix_cli.utils import launch_backend

        backend_name = "test_backend"

        # 模拟进程以退出码 1 终止
        mock_process = MagicMock()
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        launch_backend(sample_cmd, backend_name)

        mock_exit.assert_called_once_with(1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_file_not_found(self, mock_exit, mock_print, sample_cmd):
        """测试后端程序未找到时显示适当的错误消息。"""
        from matrix_cli.utils import launch_backend

        backend_name = "unknown_backend"

        with patch('matrix_cli.utils.subprocess.Popen', side_effect=FileNotFoundError()):
            launch_backend(sample_cmd, backend_name)

        # 验证显示了错误消息
        assert mock_print.call_count >= 1
        mock_exit.assert_called_once_with(1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_vllm_not_found(self, mock_exit, mock_print, sample_cmd):
        """测试 VLLM 未找到时显示安装提示。"""
        from matrix_cli.utils import launch_backend

        backend_name = "vllm"

        with patch('matrix_cli.utils.subprocess.Popen', side_effect=FileNotFoundError()):
            launch_backend(sample_cmd, backend_name)

        # 验证显示了 VLLM 特定的安装提示
        print_calls = [str(call) for call in mock_print.call_args_list]
        any_install_hint = any("pip install vllm" in call for call in print_calls)
        assert any_install_hint, "应该显示 VLLM 安装提示"
        mock_exit.assert_called_once_with(1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_sglang_not_found(self, mock_exit, mock_print, sample_cmd):
        """测试 SGLang 未找到时显示安装提示。"""
        from matrix_cli.utils import launch_backend

        backend_name = "sglang"

        with patch('matrix_cli.utils.subprocess.Popen', side_effect=FileNotFoundError()):
            launch_backend(sample_cmd, backend_name)

        # 验证显示了 SGLang 特定的安装提示
        print_calls = [str(call) for call in mock_print.call_args_list]
        any_install_hint = any("pip install sglang" in call for call in print_calls)
        assert any_install_hint, "应该显示 SGLang 安装提示"
        mock_exit.assert_called_once_with(1)

    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_keyboard_interrupt(self, mock_exit, mock_print, sample_cmd, sample_backend_name):
        """测试用户通过 Ctrl+C 中断后端进程。"""
        from matrix_cli.utils import launch_backend

        with patch('matrix_cli.utils.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.wait.side_effect = KeyboardInterrupt()
            mock_popen.return_value = mock_process

            launch_backend(sample_cmd, sample_backend_name)

            # 验证显示终止消息并以状态码 0 退出
            print_calls = [str(call) for call in mock_print.call_args_list]
            any_termination_msg = any("terminated by user" in call.lower() for call in print_calls)
            assert any_termination_msg, "应该显示用户终止消息"
            mock_exit.assert_called_once_with(0)

    @patch('matrix_cli.utils.subprocess.Popen')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_prints_launch_message(self, mock_exit, mock_print, mock_popen, sample_cmd, sample_backend_name):
        """测试启动时打印正确的消息。"""
        from matrix_cli.utils import launch_backend

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        launch_backend(sample_cmd, sample_backend_name)

        # 验证打印了启动消息
        print_calls = [str(call) for call in mock_print.call_args_list]
        any_launch_msg = any(
            f"Launching {sample_backend_name}" in call and
            " ".join(sample_cmd) in call
            for call in print_calls
        )
        assert any_launch_msg, "应该打印包含命令的启动消息"

    @patch('matrix_cli.utils.subprocess.Popen')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_launch_backend_with_long_command(self, mock_exit, mock_print, mock_popen):
        """测试处理长命令行参数。"""
        from matrix_cli.utils import launch_backend

        long_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "meta-llama/Llama-2-70b-chat-hf",
            "--tensor-parallel-size", "8",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "4096",
            "--dtype", "float16",
            "--trust-remote-code"
        ]
        backend_name = "vllm"

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        launch_backend(long_cmd, backend_name)

        # 验证命令被正确传递
        mock_popen.assert_called_once_with(long_cmd)


class TestUtilsIntegration:
    """集成测试：测试工具函数的组合使用。"""

    @pytest.fixture
    def sample_vllm_config(self):
        """示例 VLLM 配置。"""
        return {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.9,
            "port": 8000,
            "enable_chunked_prefill": True,
            "disable_log_stats": False
        }

    @pytest.fixture
    def vllm_mappings(self):
        """VLLM 参数映射。"""
        return {
            "model": "--model",
            "tensor_parallel_size": "--tensor-parallel-size",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "port": "--port",
            "enable_chunked_prefill": "--enable-chunked-prefill",
            "disable_log_stats": "--disable-log-stats"
        }

    @pytest.fixture
    def temp_vllm_config_file(self, tmp_path, sample_vllm_config):
        """创建临时 VLLM 配置文件。"""
        import yaml
        config_file = tmp_path / "vllm_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_vllm_config, f)
        return str(config_file)

    def test_load_and_map_config_full_workflow(self, temp_vllm_config_file, vllm_mappings):
        """测试完整的加载和映射工作流程。"""
        from matrix_cli.utils import load_config, map_config_to_args

        # 加载配置
        config = load_config(temp_vllm_config_file)
        assert config is not None
        assert len(config) > 0

        # 映射到参数
        args = map_config_to_args(config, vllm_mappings)
        assert len(args) > 0

        # 验证关键参数存在
        assert "--model" in args
        assert "meta-llama/Llama-2-7b-chat-hf" in args
        assert "--tensor-parallel-size" in args
        assert "4" in args
        assert "--enable-chunked-prefill" in args
        # False 的标志应该被省略
        assert "--disable-log-stats" not in args

    @pytest.fixture
    def sample_sglang_config(self):
        """示例 SGLang 配置。"""
        return {
            "model-path": "lmsys/vicuna-7b-v1.5",
            "port": 30000,
            "tp": 2,
            "mem-fraction-static": 0.7,
            "enable-torch-compile": True
        }

    @pytest.fixture
    def sglang_mappings(self):
        """SGLang 参数映射。"""
        return {
            "model-path": "--model-path",
            "port": "--port",
            "tp": "--tp",
            "mem-fraction-static": "--mem-fraction-static",
            "enable-torch-compile": "--enable-torch-compile"
        }

    @pytest.fixture
    def temp_sglang_config_file(self, tmp_path, sample_sglang_config):
        """创建临时 SGLang 配置文件。"""
        import json
        config_file = tmp_path / "sglang_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_sglang_config, f)
        return str(config_file)

    def test_load_json_and_map_sglang_config(self, temp_sglang_config_file, sglang_mappings):
        """测试加载 JSON 配置并映射到 SGLang 参数。"""
        from matrix_cli.utils import load_config, map_config_to_args

        config = load_config(temp_sglang_config_file)
        args = map_config_to_args(config, sglang_mappings)

        assert "--model-path" in args
        assert "lmsys/vicuna-7b-v1.5" in args
        assert "--port" in args
        assert "30000" in args
        assert "--enable-torch-compile" in args


class TestEdgeCases:
    """边缘情况和异常场景测试。"""

    def test_load_config_with_special_characters_in_yaml(self, tmp_path):
        """测试加载包含特殊字符的 YAML 配置。"""
        from matrix_cli.utils import load_config
        import yaml

        config_with_special = {
            "password": "p@$$w0rd!#",
            "api_key": "sk-1234567890abcdef",
            "description": "模型描述\n多行文本\n特殊字符: <>&\"'"
        }

        config_file = tmp_path / "special.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_with_special, f)

        config = load_config(str(config_file))
        assert config == config_with_special

    def test_load_yaml_with_unicode(self, tmp_path):
        """测试加载包含 Unicode 字符的 YAML。"""
        from matrix_cli.utils import load_config
        import yaml

        unicode_config = {
            "模型名称": "千问3",
            "说明": "中文配置测试",
            "列表": ["选项一", "选项二", "选项三"]
        }

        config_file = tmp_path / "unicode.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)

        config = load_config(str(config_file))
        assert config == unicode_config

    def test_map_config_with_numeric_strings(self):
        """测试映射看起来像数字的字符串值。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "port": "8000",
            "rank": "0"
        }
        mappings = {
            "port": "--port",
            "rank": "--rank"
        }

        args = map_config_to_args(config, mappings)
        # 字符串应该保持为字符串，不会转换为整数
        assert args == ["--port", "8000", "--rank", "0"]

    def test_map_config_preserves_zero_values(self):
        """测试映射值为 0 的情况不会被误判为 False。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "timeout": 0,
            "retry_count": 0
        }
        mappings = {
            "timeout": "--timeout",
            "retry_count": "--retries"
        }

        args = map_config_to_args(config, mappings)
        assert args == ["--timeout", "0", "--retries", "0"]

    def test_map_config_preserves_original_config(self):
        """测试映射过程不修改原始配置。"""
        from matrix_cli.utils import map_config_to_args

        original_config = {
            "model": "qwen3",
            "port": 8000
        }
        original_copy = original_config.copy()
        mappings = {
            "model": "--model",
            "port": "--port"
        }

        args = map_config_to_args(original_config, mappings)

        # 确认原始配置没有被修改
        assert original_config == original_copy
        assert args == ["--model", "qwen3", "--port", "8000"]

    def test_launch_backend_with_special_characters_in_command(self):
        """测试命令中包含特殊字符的处理。"""
        from matrix_cli.utils import launch_backend

        special_cmd = [
            "python", "-c",
            'print("Model: meta-llama/Llama-2-7b-chat-hf"); import sys; sys.exit(0)'
        ]
        backend_name = "test_special"

        with patch('matrix_cli.utils.subprocess.Popen') as mock_popen, \
             patch('builtins.print'), \
             patch('sys.exit') as mock_exit:

            mock_process = MagicMock()
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            launch_backend(special_cmd, backend_name)

            mock_popen.assert_called_once_with(special_cmd)
            mock_exit.assert_called_once_with(0)

    def test_launch_backend_with_empty_command_list(self):
        """测试空命令列表的处理。"""
        from matrix_cli.utils import launch_backend

        with patch('builtins.print'), \
             patch('sys.exit'):
            # 空命令可能会导致 subprocess.Popen 失败
            with pytest.raises(OSError):  # Popen 会抛出 OSError
                launch_backend([], "test_backend")

    def test_map_config_to_args_large_input_performance(self):
        """测试处理大量配置项时的性能"""
        from matrix_cli.utils import map_config_to_args
        import time

        large_config = {f"key_{i}": f"value_{i}" for i in range(1000)}
        mappings = {f"key_{i}": f"--key-{i}" for i in range(1000)}

        start_time = time.time()
        args = map_config_to_args(large_config, mappings)
        elapsed_time = time.time() - start_time

        # 确保处理时间在合理范围内
        assert elapsed_time < 0.5
        assert len(args) == 2000  # 1000 键值对 -> 2000 个参数（键+值）

    def test_map_config_to_args_handles_nested_dicts_as_strings(self):
        """测试处理嵌套字典时将其转换为字符串。"""
        from matrix_cli.utils import map_config_to_args

        config = {
            "nested_dict": {"inner_key": "inner_value"},
            "model": "qwen3"
        }
        mappings = {
            "nested_dict": "--nested",
            "model": "--model"
        }

        args = map_config_to_args(config, mappings)
        # 嵌套字典会被转换为字符串
        assert "--nested" in args
        nested_value_idx = args.index("--nested") + 1
        assert nested_value_idx < len(args)
        assert args[nested_value_idx] == "{'inner_key': 'inner_value'}"
        assert "--model" in args
        assert "qwen3" in args

    def test_load_large_config_file_performance(self, tmp_path):
        """测试加载大型配置文件的性能"""
        from matrix_cli.utils import load_config
        import json
        import time

        # 创建大型配置文件
        large_config = {"item_" + str(i): f"value_{i}" for i in range(5000)}
        config_file = tmp_path / "large_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(large_config, f)

        start_time = time.time()
        config = load_config(str(config_file))
        elapsed_time = time.time() - start_time

        # 确保加载时间在合理范围内（如1秒内）
        assert elapsed_time < 1.0
        assert len(config) == 5000


class TestParseBackendArgs:
    """测试 parse_backend_args 函数：解析后端参数并支持配置文件。"""

    def test_parse_backend_args_without_config(self):
        """测试不带配置文件参数的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--model", "qwen3", "--port", "8000"]
        result = parse_backend_args(parser, args)

        assert result.config is None
        assert result.remainder == args

    def test_parse_backend_args_with_config(self):
        """测试带配置文件参数的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--config", "config.yaml"]
        result = parse_backend_args(parser, args)

        assert result.config == "config.yaml"
        assert result.remainder == []

    def test_parse_backend_args_with_config_and_other_args(self):
        """测试同时包含配置文件和其他参数的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--config", "config.json", "--model", "qwen3", "--port", "8000"]
        result = parse_backend_args(parser, args)

        assert result.config == "config.json"
        assert result.remainder == ["--model", "qwen3", "--port", "8000"]

    def test_parse_backend_args_empty_args(self):
        """测试空参数列表的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = []
        result = parse_backend_args(parser, args)

        assert result.config is None
        assert result.remainder == []

    def test_parse_backend_args_config_last(self):
        """测试配置文件参数在最后的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--model", "qwen3", "--config", "my_config.yaml"]
        result = parse_backend_args(parser, args)

        assert result.config == "my_config.yaml"
        assert result.remainder == ["--model", "qwen3"]

    def test_parse_backend_args_config_with_path(self):
        """测试配置文件参数包含路径的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--config", "/path/to/config.yaml", "--verbose"]
        result = parse_backend_args(parser, args)

        assert result.config == "/path/to/config.yaml"
        assert result.remainder == ["--verbose"]

    def test_parse_backend_args_multiple_unknown_args(self):
        """测试多个未知参数的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--arg1", "value1", "--arg2", "value2", "--arg3", "value3"]
        result = parse_backend_args(parser, args)

        assert result.config is None
        assert result.remainder == args

    def test_parse_backend_args_config_flag_only(self):
        """测试只有配置文件标志的情况。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="test_backend",
            description="Test backend parser",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--config", "test_config.json"]
        result = parse_backend_args(parser, args)

        assert result.config == "test_config.json"
        assert result.remainder == []

    def test_parse_backend_args_custom_parser_description(self):
        """测试使用自定义解析器描述。"""
        from matrix_cli.utils import parse_backend_args
        import argparse

        parser = argparse.ArgumentParser(
            prog="custom_backend",
            description="Custom backend for testing",
            add_help=False,
            conflict_handler='resolve'
        )

        args = ["--model", "test_model"]
        result = parse_backend_args(parser, args)

        assert result.config is None
        assert result.remainder == args
        assert parser.prog == "custom_backend"
        assert parser.description == "Custom backend for testing"
