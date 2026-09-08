# GPU (2xT4) 冷启动耗时实测 — 2026-09-08

> MASTER 命名 `date+time+summary`。测量对象：serve kernel v4（ctx 98304, np4, KV q8_0, 源码构建 llama-server）。数据来自 ntfy 事件时间戳（真实运行，非估算）。

## 相位分解（push → available）

| 相位 | 时长 | 起止 | 说明 |
|---|---|---|---|
| push → RUNNING | ~3 min | ~18:12 → 18:15:41 | Kaggle 队列 + T4 x2 槽位 |
| weights-mounted | <10 s | 18:15:41 | 私有 dataset 直接挂载（16.46 GB 无下载） |
| engine-built | 25 min 11 s | →18:40:52 | llama.cpp 源码 cmake 构建（CUDA, -j nproc） |
| tunnel-url | ~6 s | 18:40:58 | cloudflared 保留 URL |
| compiling→serving→ready | 2 min | 18:42:59 | llama-server 加载 15.3 GiB → VRAM offload + warmup（startup_secs=120） |
| 自检 | ~23 s | 18:43:21 | benchmark 13.5 tok/s + parallel4 4.5s（后端同时可服务） |

## 结论

| 口径 | 数值 |
|---|---|
| push → READY（总） | **≈31 min** |
| RUNNING → READY | ≈27.3 min（1640 s） |
| engine-built → READY | ≈127 s（模型加载+预热） |
| 长桩 | llama.cpp 源码构建 25 min（占 82%） |

## 减负点（roadmap item 5）

llama-server 若缓存进私有 dataset：

| 项 | 效果 |
|---|---|
| 去掉构建 | push → READY ≈ **4-6 min**（queue 3min + 挂载 <10s + 加载 2min） |
| 对 controller 意义 | GPU 仅当需要时预热即可，不用常驻；TPU 到期前 ~40 min 预热窗口充足 |

## 旁证（v3 vs v4 一致性）

| 内核 | engine-built | 模型加载 startup |
|---|---|---|
| v3 | 1504 s | 95 s（ctx 16384） |
| v4 | 1509 s | 120 s（ctx 98304, KV q8_0） |

源码构建时间稳定 ≈ 25 min（1504-1509 s）。备注：本次为真实运行测量，未另起新 GPU session 复测（省 quota）。