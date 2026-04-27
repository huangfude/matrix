
# Matrix CLI

Matrix Router 的命令行界面（CLI）

## Build

```shell
cargo build --release
```


## Examples

启动 Matrix Router

```shell
# 普通模式
matrix router --worker-urls http://localhost:8000 http://localhost:8001

# PD分离
matrix router --prefill-urls http://localhost:8000 --decode-urls http://localhost:8001
```

支持设置推理接口的API KEY，可以通过--data-plane-api-keys参数添加，但推荐使用环境变量的方式

```shell
# 通过DATA_PLANE_API_KEYS设置API KEYS
export DATA_PLANE_API_KEYS="sk-key1,sk-key2"
matrix router --worker-urls http://localhost:8000 http://localhost:8001
```

启动 vLLM / sglang

```shell

# 兼容vllm / sglang参数
matrix vllm /models/Qwen3-32B --tensor-parallel-size 8
matrix sglang --model /models/Qwen3-32B --tp 8

# 通过配置文件
matrix vllm --config ./sample_conf/qwen3_vllm.yaml
matrix sglang --config ./sample_conf/qwen3_sglang.yaml
```