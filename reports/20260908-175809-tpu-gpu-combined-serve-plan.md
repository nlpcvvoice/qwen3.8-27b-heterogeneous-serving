# TPU+GPU 联合 Serve — 实施规划

| 项 | 值 |
|---|---|
| 日期 | 2026-09-08 |
| 目标 | 单一稳定入口 · TPU 优先 GPU 兜底 · ≤4 客户端共用 |
| 前置 | TPU v4 READY (126.8 tok/s) · GPU v3 READY (13.6 tok/s) ✓ 双引擎已验证 |

## 引擎现状

| 维度 | TPU v5e-8 | GPU 2xT4 |
|---|---|---|
| 模型 | Qwen3.8-27B bf16 | Qwen3.8-27B Q4_K_M (16.46GB) |
| 引擎 | vLLM (vllm-tpu) | llama.cpp llama-server `-np 4` |
| 吞吐 | 126.8 tok/s | 13.6 tok/s |
| ctx | 262144 | 16384 (每请求 4096) |
| 启动 | env数据集 ~20-35 min | 构建 25min / 冷启 3min* |
| 排队 | 0-3h | 即时 |
| 额度 | ~20h/周 | ~30h/周 |
| 在线上限 | 180 min (keepalive) | 420 min (keepalive) |
| 端点 (当前) | equality-dated-cal… /v1 | induced-wife-print… /v1 |

\* = 构建产物缓存到私有 dataset 后可到 3min (见 P3)

## 架构

```
客户端 ≤4 (opencode → http://<controller>:8080/v1 单配置)
        │
        ▼
┌───── 本地 Controller (本机/任一客户端, 不占 Kaggle 额度) ─────┐
│  [调度器] 每 60s: kaggle status + /v1/models 健康探活          │
│           → 决定推送/重启哪个内核 (TPU 首选, GPU 兜底)         │
│  [路由器] :8080/v1 → 转发存活端点 (SSE 流式透传)               │
│           密钥跟随 state 文件轮换 (kaggle-tpu/gpu-lab.json)   │
└───────────────────────────────────────────────────────────────┘
        │
   TPU 端点 ──── GPU 端点 (各自独立的免费额度会话)
```

## 路由策略

| 规则 | 行为 |
|---|---|
| 优先级 | TPU READY > GPU READY > 503(原因) |
| 切换 | TPU 心跳丢失 >60s → GPU;TPU 恢复 → 回切 |
| 瞬时抖动 | 504/502 重试 1 次再判故障 |
| 模型 id | 单 `qwen3.8-27b` 自动路由;query `?route=gpu` 可强制 |
| 协议 | /v1/models /v1/chat/completions (+ stream) |

## 配额预算 (周)

| 模式 | TPU | GPU | 结论 |
|---|---|---|---|
| A 常开双机 | 7×2.5h≈17.5h ✓ | 7×~4h≈28h ✓ | 全覆盖,耗双额度 |
| B 按需 GPU | 同上 | 仅 TPU 失联时启用 ~5h | 省额度,兜底有冷启代价 |

推荐 B 主推(冷启经 P3 优化到 ~3min)

## 分阶段交付

| 阶段 | 内容 | 验证 |
|---|---|---|
| P1 | 路由器+健康检查+故障切换 (纯本机) | 拔 TPU→自动切 GPU;回切 ✓ |
| P2 | 调度器: 定时保活/自动补推内核 | TPU 上限前自动重启续期 |
| P3 | 缓存 built llama-server → 私有 dataset | 冷启 ≤3min,省 25min 构建 |
| P4(可选) | 双引擎分流加载 · 多客户端鉴权 | 压测对比 |

## 风险与对策

| 风险 | 对策 |
|---|---|
| tunnel URL/密钥每次轮换 | state 文件追踪,路由器 60s 自动刷新 |
| 本地单点故障 | controller 可跑在任何一台客户端 |
| 会话被强停 | 调度器自动重推,队列等待转 GPU |
| stream 断流 | 透传为主,失败降级非流式重试 |

## 待确认

1. 常开(A) or 按需(B)?
2. controller 跑本机(GCP)还是某一台客户端?
3. 是否接受将 built llama-server 上传私有 dataset(绕过 25min 构建)?
4. 单 id 自动路由 or 双 id (qwen3.8-27b-tpu / -gpu)?