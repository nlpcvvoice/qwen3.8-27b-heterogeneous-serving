# Qwen3.8-27B TPU Service — Ready 状态报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-08 |
| 内核 | tentenshishi/qwen38-tpu-serve (v4) |
| 状态 | READY / serving |
| 运行时长 | 180 min 上限 (17:04 起) |
| 模型 | qwen3.8-27b |
| 上下文 | 262144 |
| 基准 | 126.8 tok/s (单流) |
| 自检 | image-test: red ✓ |

## 端点

| 项 | 值 |
|---|---|
| Base URL | https://equality-dated-cal-metabolism.trycloudflare.com/v1 |
| API Key | sk-<redacted for repo> (见 tmp/kaggle-tpu-lab.json) |
| 验证 | /v1/models 200 · chat "alive" · tool_calls list_files ✓ |

## opencode 配置

| 项 | 值 |
|---|---|
| Provider | kaggle (kaggle/qwen3.8-27b 为默认模型) |
| 生效 | 需重启 opencode |

## GPU 兜底路径 (进行中)

| 步骤 | 结果 |
|---|---|
| 1. 2xT4 探针 | ✓ 2×Tesla T4 / 31.27GB |
| 2. GGUF 私有数据集 | ✓ q4-k-m-private (16.46GB) |
| 3. GPU serve 内核 | 构建中 (版本 3, 修复 libcuda.so.1 路径) |

## 备注

| 事项 | 说明 |
|---|---|
| API Key 轮换 | 每次内核重跑生成新端点/密钥, 本文档仅为本次会话快照 |
| Kaggle Token | reference/API-Token 不入库 · 仅内存 |