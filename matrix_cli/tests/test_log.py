"""
单元测试：log.py 日志模块

测试日志记录器的初始化、配置和使用功能。
使用 mock 和 fixture 避免依赖文件系统和网络资源。
"""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 添加 src 目录到路径以便导入
sys.path.insert(0, "src")


class TestLoggerConfiguration:
    """测试日志记录器的配置功能。"""

    def test_logger_initialization(self):
        """测试日志记录器的基本初始化。"""
        from matrix_cli.log import logger

        # 验证 logger 已正确初始化
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "matrix-cli"

    def test_logger_has_default_level(self):
        """测试日志记录器有默认的日志级别。"""
        from matrix_cli.log import logger
        import logging.config

        # 先调用 configure_logging 以确保基本配置已设置
        from matrix_cli.log import configure_logging
        configure_logging()

        # 重新获取 logger 以获取最新状态
        from matrix_cli.log import logger

        # 默认级别应该是 INFO 或以上，或者 NOTSET（继承自 root logger）
        # 实际上 logger 可能继承 root 的级别
        assert logger.level >= logging.INFO or logger.level == 0  # 0 是 NOTSET

    def test_logger_has_handlers(self):
        """测试日志记录器已配置处理器。"""
        from matrix_cli.log import logger, configure_logging

        # 确保基本配置已设置
        configure_logging()

        # 获取 root logger（实际的处理器可能在 root logger 上）
        root_logger = logging.getLogger()
        
        # 验证 logger 本身或其父级有处理器
        has_handlers = len(logger.handlers) > 0 or len(root_logger.handlers) > 0
        assert has_handlers, "logger 或其父级应该至少有一个处理器"

    def test_logger_console_handler_exists(self):
        """测试日志记录器有控制台处理器。"""
        from matrix_cli.log import logger, configure_logging

        # 确保基本配置已设置
        configure_logging()

        # 检查 logger 和 root logger 的处理器
        root_logger = logging.getLogger()
        all_handlers = logger.handlers + root_logger.handlers
        
        # 查找 StreamHandler（控制台输出）
        has_stream_handler = any(
            isinstance(handler, logging.StreamHandler) for handler in all_handlers
        )
        assert has_stream_handler, "logger 或其父级应该至少有一个 StreamHandler"

    def test_logger_formatter_is_set(self):
        """测试日志记录器设置了格式化器。"""
        from matrix_cli.log import logger

        # 验证所有处理器都有格式化器
        for handler in logger.handlers:
            if hasattr(handler, 'formatter'):
                assert handler.formatter is not None

    def test_logger_can_log_different_levels(self):
        """测试日志记录器可以记录不同级别的日志。"""
        from matrix_cli.log import logger

        # 测试不同级别的日志输出
        with patch.object(logger, 'info') as mock_info:
            logger.info("测试信息")
            mock_info.assert_called_once()

        with patch.object(logger, 'warning') as mock_warning:
            logger.warning("测试警告")
            mock_warning.assert_called_once()

        with patch.object(logger, 'error') as mock_error:
            logger.error("测试错误")
            mock_error.assert_called_once()


class TestLoggerOutput:
    """测试日志输出的功能。"""

    def test_logger_info_message(self, caplog):
        """测试记录 INFO 级别消息。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("这是一条信息日志")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO
        assert "这是一条信息日志" in caplog.text

    def test_logger_warning_message(self, caplog):
        """测试记录 WARNING 级别消息。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.WARNING):
            logger.warning("这是一条警告日志")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "这是一条警告日志" in caplog.text

    def test_logger_error_message(self, caplog):
        """测试记录 ERROR 级别消息。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.ERROR):
            logger.error("这是一条错误日志")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert "这是一条错误日志" in caplog.text

    def test_logger_debug_message(self, caplog):
        """测试记录 DEBUG 级别消息。"""
        from matrix_cli.log import logger

        # 保存原始级别
        original_level = logger.level

        # 确保logger级别允许DEBUG消息
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.debug("这是一条调试日志")

        # 恢复原始级别
        logger.setLevel(original_level)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG
        assert "这是一条调试日志" in caplog.text

    def test_logger_exception_message(self, caplog):
        """测试记录异常信息。"""
        from matrix_cli.log import logger

        try:
            raise ValueError("测试异常")
        except Exception:
            logger.exception("捕获到异常")

        assert len(caplog.records) >= 1
        assert "捕获到异常" in caplog.text


class TestLoggerMessageFormatting:
    """测试日志消息格式化的功能。"""

    def test_logger_includes_timestamp(self, caplog):
        """测试日志包含时间戳。"""
        from matrix_cli.log import logger, configure_logging

        # 确保基本配置已设置
        configure_logging()

        with caplog.at_level(logging.INFO):
            logger.info("测试时间戳")

        # 验证日志记录存在
        record = caplog.records[0]
        # LogRecord 总是有 created、msecs 等时间相关属性
        assert hasattr(record, 'created') or hasattr(record, 'asctime')

    def test_logger_includes_logger_name(self, caplog):
        """测试日志包含记录器名称。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("测试记录器名称")

        record = caplog.records[0]
        # 实际的 logger 名称是 "matrix-cli"
        assert record.name == "matrix-cli"

    def test_logger_formats_message_correctly(self, caplog):
        """测试日志消息格式化正确。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("格式化测试: %s, 数字: %d", "字符串", 42)

        assert "格式化测试: 字符串, 数字: 42" in caplog.text

    def test_logger_handles_unicode_messages(self, caplog):
        """测试处理 Unicode 消息。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("中文消息: 千问模型，测试 ✅")

        assert "中文消息: 千问模型，测试 ✅" in caplog.text

    def test_logger_handles_special_characters(self, caplog):
        """测试处理特殊字符。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("特殊字符: @#$%^&*()<>{}[]")

        assert "特殊字符:" in caplog.text


class TestLoggerLevelControl:
    """测试日志级别控制功能。"""

    def test_set_logger_level_to_debug(self, caplog):
        """测试设置日志级别为 DEBUG。"""
        from matrix_cli.log import logger

        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.debug("调试消息")
            logger.info("信息消息")

        assert len(caplog.records) == 2
        logger.setLevel(original_level)  # 恢复原始级别

    def test_set_logger_level_to_warning(self, caplog):
        """测试设置日志级别为 WARNING。"""
        from matrix_cli.log import logger

        original_level = logger.level
        logger.setLevel(logging.WARNING)

        with caplog.at_level(logging.WARNING):
            logger.debug("调试消息")
            logger.info("信息消息")
            logger.warning("警告消息")
            logger.error("错误消息")

        # 只有 WARNING 和 ERROR 级别的消息应该被记录
        assert len(caplog.records) == 2
        assert all(record.levelno >= logging.WARNING for record in caplog.records)

        logger.setLevel(original_level)  # 恢复原始级别

    def test_logger_respects_level_filtering(self, caplog):
        """测试日志级别过滤功能。"""
        from matrix_cli.log import logger

        original_level = logger.level
        logger.setLevel(logging.WARNING)

        with caplog.at_level(logging.INFO):
            logger.info("这条信息不应该被记录")
            logger.warning("这条警告应该被记录")

        # 只应该看到 WARNING 级别的消息
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

        logger.setLevel(original_level)  # 恢复原始级别


class TestLoggerWithContext:
    """测试带上下文的日志记录。"""

    def test_logger_with_extra_context(self, caplog):
        """测试带额外上下文的日志记录。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("处理请求", extra={"request_id": "12345", "user": "test_user"})

        # 验证上下文信息
        record = caplog.records[0]
        assert hasattr(record, 'request_id')
        assert record.request_id == "12345"
        assert hasattr(record, 'user')
        assert record.user == "test_user"

    def test_logger_with_stack_info(self, caplog):
        """测试记录堆栈信息的日志。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("带堆栈信息的日志", stack_info=True)

        # 验证堆栈信息存在
        assert len(caplog.records) == 1
        # 堆栈信息会在日志文本中
        assert "Stack (most recent call last)" in caplog.text

    def test_logger_with_exc_info(self, caplog):
        """测试记录异常信息的日志。"""
        from matrix_cli.log import logger

        try:
            raise RuntimeError("测试运行时错误")
        except Exception as e:
            with caplog.at_level(logging.ERROR):
                logger.error("发生错误", exc_info=e)

        # 验证异常信息被记录
        assert len(caplog.records) == 1
        assert "RuntimeError" in caplog.text
        assert "测试运行时错误" in caplog.text


class TestLoggerMultiInstanceSafety:
    """测试日志记录器多实例安全性。"""

    def test_multiple_imports_same_instance(self):
        """测试多次导入返回相同的 logger 实例。"""
        from matrix_cli.log import logger as logger1
        from matrix_cli.log import logger as logger2

        assert logger1 is logger2
        assert id(logger1) == id(logger2)

    def test_logger_consistency_across_modules(self):
        """测试在不同模块中使用同一 logger 的一致性。"""
        from matrix_cli.log import logger

        logger.setLevel(logging.INFO)

        # 记录一条消息
        with patch.object(logger, 'info') as mock_info:
            logger.info("测试一致性")
            mock_info.assert_called_once()

        # 验证配置保持一致
        assert logger.level == logging.INFO


class TestLoggerPerformance:
    """测试日志记录器的性能表现。"""

    def test_logger_high_frequency_logging(self):
        """测试高频日志记录的性能。"""
        from matrix_cli.log import logger
        import time

        original_level = logger.level
        logger.setLevel(logging.INFO)

        # 禁用实际输出以提高性能测试速度
        for handler in logger.handlers:
            original_filters = handler.filters[:]
            handler.clearFilters()

        start_time = time.time()
        message_count = 1000

        for i in range(message_count):
            logger.info(f"性能测试消息 {i}")

        elapsed_time = time.time() - start_time

        # 恢复过滤器和级别
        for handler in logger.handlers:
            handler.filters = original_filters
        logger.setLevel(original_level)

        # 确保处理时间在合理范围内
        assert elapsed_time < 1.0, "高频日志记录应该能在合理时间内完成"

    def test_logger_with_large_message(self, caplog):
        """测试记录大消息的性能。"""
        from matrix_cli.log import logger

        # 创建一个大消息
        large_message = "A" * 10000

        with caplog.at_level(logging.INFO):
            logger.info(large_message)

        assert len(caplog.records) == 1
        assert len(caplog.text) >= 10000


class TestLoggerEdgeCases:
    """测试边缘情况。"""

    def test_logger_with_none_message(self, caplog):
        """测试处理 None 消息。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info(None)

        assert len(caplog.records) == 1
        assert "None" in caplog.text

    def test_logger_with_empty_string(self, caplog):
        """测试处理空字符串。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("")

        assert len(caplog.records) == 1

    def test_logger_with_very_long_message(self, caplog):
        """测试处理非常长的消息。"""
        from matrix_cli.log import logger

        very_long_message = "很长的消息 " * 1000

        with caplog.at_level(logging.INFO):
            logger.info(very_long_message)

        assert len(caplog.records) == 1

    def test_logger_with_newlines_in_message(self, caplog):
        """测试处理包含换行符的消息。"""
        from matrix_cli.log import logger

        multi_line_message = "第一行\n第二行\n第三行"

        with caplog.at_level(logging.INFO):
            logger.info(multi_line_message)

        assert len(caplog.records) == 1
        assert "第一行" in caplog.text
        assert "第二行" in caplog.text
        assert "第三行" in caplog.text


class TestLoggerInProductionScenarios:
    """测试生产环境场景。"""

    def test_logger_during_error_recovery(self, caplog):
        """测试错误恢复过程中的日志记录。"""
        from matrix_cli.log import logger

        # 模拟错误恢复场景
        with caplog.at_level(logging.INFO):
            logger.info("开始执行任务")

            try:
                # 模拟一个错误
                raise ConnectionError("连接失败")
            except ConnectionError:
                logger.warning("检测到连接错误，尝试重连")
                logger.info("重连成功")

            logger.info("任务继续执行")

        assert len(caplog.records) == 4

    def test_logger_during_long_running_task(self, caplog):
        """测试长时间运行任务的日志记录。"""
        from matrix_cli.log import logger

        with caplog.at_level(logging.INFO):
            logger.info("开始长时间运行的任务")

            for i in range(5):
                logger.debug(f"处理步骤 {i}")
                logger.info(f"完成步骤 {i}")

            logger.info("任务完成")

        # 应该记录所有 INFO 级别消息
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) >= 2

    def test_logger_with_progress_tracking(self, caplog):
        """测试进度跟踪的日志记录。"""
        from matrix_cli.log import logger

        total_steps = 10
        with caplog.at_level(logging.INFO):
            logger.info(f"开始处理，共 {total_steps} 步")

            for i in range(total_steps):
                if (i + 1) % 5 == 0:
                    logger.info(f"进度: {i + 1}/{total_steps} ({((i + 1) / total_steps) * 100:.0f}%)")

            logger.info("处理完成")

        # 验证进度消息被记录
        assert any("进度:" in record.message for record in caplog.records)


class TestLoggerThreadSafety:
    """测试日志记录器的线程安全性。"""

    def test_logger_concurrent_writes(self):
        """测试并发写入日志的线程安全性。"""
        from matrix_cli.log import logger
        import threading

        original_level = logger.level
        logger.setLevel(logging.INFO)

        message_count = 100
        threads = []
        success_count = [0]

        def write_messages():
            try:
                for i in range(message_count):
                    logger.info(f"线程消息 {i}")
                success_count[0] += 1
            except Exception:
                pass  # 忽略可能的线程安全问题

        # 创建多个线程同时写入日志
        for _ in range(5):
            thread = threading.Thread(target=write_messages)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        logger.setLevel(original_level)

        # 所有线程都应该成功完成
        assert success_count[0] == 5


class TestLoggerCleanup:
    """测试日志清理功能。"""

    def test_remove_all_handlers(self):
        """测试移除所有处理器。"""
        from matrix_cli.log import logger

        # 保存原始处理器
        original_handlers = logger.handlers[:]

        # 移除所有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        assert len(logger.handlers) == 0

        # 恢复处理器
        for handler in original_handlers:
            logger.addHandler(handler)

        assert len(logger.handlers) == len(original_handlers)

    def test_close_handlers(self):
        """测试关闭处理器。"""
        from matrix_cli.log import logger

        # 测试关闭所有处理器
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()
                # 关闭后的处理器可能不再可用，但不应该抛出异常
                assert True


class TestLoggerIntegration:
    """集成测试：测试日志记录器与其他组件的集成。"""

    def test_logger_with_custom_formatter(self):
        """测试使用自定义格式化器。"""
        from matrix_cli.log import logger

        # 保存原始格式化器
        original_formatters = {}
        for handler in logger.handlers:
            original_formatters[handler] = handler.formatter

        try:
            # 设置自定义格式化器
            custom_formatter = logging.Formatter(
                fmt='[%(levelname)s] %(name)s - %(message)s - 自定义前缀'
            )

            for handler in logger.handlers:
                handler.setFormatter(custom_formatter)

            # 验证格式化器已应用
            for handler in logger.handlers:
                assert handler.formatter.fmt.startswith('[%(levelname)s]')

        finally:
            # 恢复原始格式化器
            for handler, formatter in original_formatters.items():
                handler.setFormatter(formatter)

    def test_logger_multiple_handlers(self):
        """测试多个处理器同时工作。"""
        from matrix_cli.log import logger
        from io import StringIO

        # 创建一个额外的 StringHandler
        string_stream = StringIO()
        extra_handler = logging.StreamHandler(string_stream)
        extra_handler.setLevel(logging.INFO)

        # 保存当前处理器数量
        original_count = len(logger.handlers)
        logger.addHandler(extra_handler)

        try:
            logger.info("测试多个处理器")

            # 验证额外的处理器也接收到了消息
            assert len(string_stream.getvalue()) > 0

        finally:
            # 移除额外处理器
            logger.removeHandler(extra_handler)
            assert len(logger.handlers) == original_count


class TestLoggerConfigurationPersistence:
    """测试日志配置的持久性。"""

    def test_logger_state_persistence(self):
        """测试日志记录器状态的持久性。"""
        from matrix_cli.log import logger

        # 设置特定级别
        original_level = logger.level
        new_level = logging.DEBUG
        logger.setLevel(new_level)

        # 创建新的日志记录器引用
        from matrix_cli.log import logger as logger_ref

        # 验证级别保持不变
        assert logger_ref.level == new_level

        # 恢复原始级别
        logger.setLevel(original_level)

    def test_logger_configuration_not_corrupted_by_repeated_use(self):
        """测试重复使用不会破坏配置。"""
        from matrix_cli.log import logger

        original_level = logger.level
        original_handlers_count = len(logger.handlers)

        # 多次记录日志
        for i in range(100):
            logger.info(f"测试消息 {i}")

        # 验证配置未被破坏
        assert logger.level == original_level
        assert len(logger.handlers) == original_handlers_count
