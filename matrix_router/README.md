
# Matrix Router

matrix router整合了sglang的router和vllm的router，兼容sglang和vllm的启动参数，支持多种推理引擎后端，支持多种路由策略。

# 快速开始

支持sglang和vllm的参数启动，示例为自定义的简化参数

## PD分离场景

```bash
./matrix-router --port 8000 \
    --prefill-urls http://127.0.0.1:18000 \
    --decode-urls http://127.0.0.1:18001 \
    --backend vllm \
    --static-models qwen 
```

- static-models参数：指定模型名称，可省略。当不指定时，会自动通过发给推理引擎的请求获取。

## 混合部署的场景

```bash
./matrix-router --port 8000 \
    --worker-urls http://127.0.0.1:18000 http://127.0.0.1:18001 \
    --log-level info 
```

- 不需要指定推理引擎后端
