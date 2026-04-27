# Matrix Router

轻量级高性能 AI 推理路由组件，支持多后端负载均衡、认证鉴权、可观测性和流量管理。

## 项目简介

Matrix Router 是一个基于 [smg](https://github.com/lightseekorg/smg) 框架构建的推理路由服务，专为 AI 模型推理场景设计。它能够智能地将请求路由到多个推理后端（如 vLLM、SGLang 等），提供负载均衡、熔断限流、认证鉴权等企业级功能。

## 核心特性

- **智能路由**：支持多种负载均衡策略（轮询、随机、加权等）
- **认证鉴权**：集成 smg-auth，支持多种认证方式
- **可观测性**：内置 Prometheus 指标、OpenTelemetry 链路追踪、结构化日志
- **流量管理**：支持熔断、限流、重试等流量控制策略
- **协议兼容**：完整支持 OpenAI API 协议
- **高性能**：基于 tokio 异步运行时，axum Web 框架

## 模块说明

| 模块 | 说明 |
|------|------|
| **matrix_router** | 核心路由库，基于 smg 框架构建，提供 HTTP/WebSocket 服务、路由逻辑、负载均衡等功能 |
| **matrix_cli** | Python 扩展模块，使用 PyO3 构建为 cdylib，支持从 Python 直接调用 router 功能 |

## 技术栈

- **运行时**：Tokio (异步 I/O)
- **Web 框架**：Axum
- **序列化**：Serde (JSON/YAML)
- **Python 集成**：PyO3
- **链路追踪**：OpenTelemetry
- **指标监控**：Prometheus Client
- **协议支持**：OpenAI Protocol

## 快速开始

### 前置要求

- Rust 1.85+
- Python 3.8+
- Cargo 工作区

### 构建

```bash
# 构建 Rust 项目
make build

# 构建 Python wheel 包
make wheel

# 运行测试
make test
```

### 安装

```bash
# 安装 Python 包
make install

# 卸载
make uninstall
```

### 启动服务

```bash
# 启动 router 服务
cargo run -p matrix_router

# 或通过 Python 调用
python -c "from matrix_cli_rs import start_router; start_router()"
```

## 配置说明

主要配置项（通过命令行参数或环境变量）：

- `--config`: 配置文件路径
- `--host`: 服务监听地址
- `--port`: 服务监听端口
- `--backend`: 推理后端地址

详细配置请参考各模块的文档。

## 项目结构

```
matrix/
├── Cargo.toml           # Workspace 配置
├── Makefile             # 构建脚本
├── matrix_router/       # 路由核心库
│   ├── src/lib.rs       # 库入口
│   └── src/main.rs      # 二进制入口
└── matrix_cli/          # Python 扩展
    └── src/lib.rs       # PyO3 绑定
```

## 开发指南

```bash
# 格式化代码
make format

# 生成文档
make docs

# 清理构建产物
make clean
```

## License

See [LICENSE](LICENSE) file.