# Locust 压测报告 — TPU vs GPU 对照

| 项 | 值 |
|---|---|
| 日期 | 2026-09-08 |
| 工具 | Locust 2.46.5 (headless) |
| 载荷 | 5 并发用户 · ramp 2/s · 70s · 混合任务 (流式6:非流式3) |
| 语料 | 15 条真实 prompt (聊天/代码/工具调用/长文/数学) |
| 引擎 | TPU vLLM bf16 · GPU llama.cpp Q4_K_M (-np 4) |
| 失败率 | 两端均 **0%** |

## 结果

| 指标 | TPU v5e-8 | GPU 2xT4 | 差距 |
|---|---|---|---|
| 请求数 | 161 | 14 | T=11.5x |
| RPS | 2.36 | 0.21 | T=11x |
| TTFT p50 | 125 ms | 2926 ms | T=23x |
| TTFT p95 | 172 ms | 9368 ms | — |
| 总延迟 p50 | 571 ms | 16.6 s | T=29x |
| 总延迟 p95 | 1.9 s | 37.7 s | — |
| 错误率 | 0% | 0% | 持平 |

## 解读

| 观察 | 原因 |
|---|---|
| 5 用户即 50% 请求都在 600ms 内完成 (TPU) | vLLM 连续批处理 + TPU 高宽带 |
| GPU 慢 11-29x | 13.6 tok/s 解码速度理论上限 + 4 slot 排队 |
| GPU TTFT p50 2.9s >> TPU | llama.cpp 预填充 + T4 显存带宽限制 |
| 两端零失败 | 并发队列无超时, keepalive 正常 |

## 说明与下一步

| 事项 | 备注 |
|---|---|
| 负载未饱和 | 5 用户受 wait_time 限制; 下一步 ramp 到 20-50 并发找饱和点 |
| persona 语料 | 当前静态 15 条; 升级为 LangGraph 生成的 20+ persona 真实分布 |
| GPU v4 (KV q8_0, ctx 98304) | 构建中, ready 后补测并对比 v3 |
| 观测落库 | Locust CSV 已在 tmp/locust/; 后续接 Langfuse/MLflow |
| SLO 目标 | TTFT p95 < 2s (TPU 已达标 172ms) |