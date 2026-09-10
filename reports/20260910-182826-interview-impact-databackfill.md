# 面试 Impact 数据回填 — 一眼看懂项目价值

> 用途:把已经验证过的数据整理成面试口径,回答「Business use case & impact」等 5 问。
> 所有数字均来自实测(reports/ 内对应文件),不是估算。

---

## 一句话定位

免费的自托管 Qwen3.8-27B 推理服务(OpenAI 兼容),**双加速器自动故障切换**,
面向 agent / tool-call / 长上下文负载,对比按量付费 API 成本为 **\$0/百万 token**。

---

## 核心 Impact 指标(面试必答表)

| 指标 | 实测值 | 数据来源 / 日期 | 亮点表述 |
|---|---|---|---|
| 推理吞吐(单流) | **126.5 tok/s** | TPU benchmark 2026-09-09 | 首 token 即出的量级,满足交互式 agent |
| 高规格上下文 | **262,144**(TPU)· **98,304**(GPU) | /models 实测 | 长文档/多轮 agent 不爆窗 |
| 并发稳定 | **4 并发,4.3s,0 错误**(GPU) | parallel4-test 2026-09-09 | 多 slot 连续批处理无冲突 |
| 负载下时延 | **TTFT p50 125ms / p95 172ms**, 0% 错 | Locust 5VU 2026-09-08 | 看报告文件 `loadtest/` |
| 故障切换 | **TPU 故障→自动切 GPU;恢复→自动回切**, 全程零人工 | controller 实测 2026-09-09 | 高可用叙事 |
| 端点即插即用 | 每次 READY **自动登记** endpoint+key 到单文件 | `current_services.json` 实测 | agent 直接消费,免手工配 |
| 冷启动 | TPU ~22min · GPU ~31min(含25min构建) | 历次启动实测 | 已定位长尾,roadmap 有缓存方案 |
| 推理成本 | **\$0** | Kaggle 免费配额 | vs 商业 API ≤\$0.3/百万 tok |
| uptime | **未测**(按配额 9h/30h 会话) | — | 面试可主动讲:受免费配额约束,生产化=SLA/autoscale |

## 五个面试问题的回填

### Q1 Business use case & impact
| 维度 | 数据/表述 |
|---|---|
| Use case | 自托管旗舰开源模型跑 agent / tool-call / 长上下文 |
| 成本对比 | \$0 推理 vs 按量 API;**每百万 token 省 \$0.3+** |
| 量级证据 | 126.5 tok/s 单流 · 4 并发无错 · 双引擎故障切换 |
| 模拟生产 | 5VU Locust 压测(RPS/TTFT/p95/错误率)已落库 `loadtest/` |

### Q2 Model architecture & trade-offs
| 选择 | 取舍 | 证据 |
|---|---|---|
| TPU v5e-8 bf16(vLLM) vs GPU 2xT4 Q4_K_M | 全精度×262k ctx×快 vs 4bit×省电×随时可用 | README 对照表 |
| vLLM 需 Ampere+ → T4 腿跑 llama.cpp | 硬件约束驱动引擎选型 | `machine_shape=NvidiaTeslaT4` 关键坑 |
| KV q8_0 预算 | ctx 98304 total = 4×24576/slot,针对帧存封顶 | 实测 runtime buffer 印证 |

### Q3 Model-level work
| 项 | 落地证据 |
|---|---|
| 量化+显存 offload | Q4_K_M 16.5GB → 2xT4, KV 帧存数学 |
| 连续批处理 | `--parallel 4` slots, 4 并发无错 |
| 从源码编译 llama.cpp | 修 `libcuda.so.1` 驱动发现 |
| 自定义评测 | 工具调用断言、4 并发实证、benchmark 门禁 |

### Q4 Production challenges
| 项 | 处理 |
|---|---|
| 版本化 | GGUF pinned + 私有 dataset 镜像(零运行下载) |
| 监控 | ntfy 事件总线 + 心跳 + 每阶段自检 |
| 故障切换 | controller: unhealthy→failover, 恢复→回切(实测) |
| 可观测 | Langfuse/MLflow 接入(规划,P2) |

### Q5 Evaluation methodology
| 项 | 证据 |
|---|---|
| 探针脚本 | `probe_tpu_kernel.py` / GPU probe |
| 吞吐+并发 | 126.5 tok/s · 4 并发 4.3s 0错 |
| LLM-as-judge | bf16 vs Q4 对照(规划 P2) |

## 缺口清单(已认领,按优先级)

| 缺口 | 阻碍 | 计划 |
|---|---|---|
| uptime%/SLA | 免费配额限制 | 面试主动转为自己设定目标并监控 |
| 20-50 VU 饱和压测 | 需引擎在线(配额) | 等下一次自然升起时跑 |
| Langfuse/MLflow 落库 | 纯本地工作,零配额 | 可直接开始(P2) |
| LangGraph personas | 纯本地离线 | 可直接开始(P2) |
| llama-server 缓存 | 需 1 次 GPU 配额做真实上传 | P3,代码可零配额先写 |

## 面试叙事一句话模板

> 「我用 \$0 在双块免费硬件上把 Qwen3.8-27B 部署成 OpenAI 兼容服务,实测单流
> 126.5 tok/s、262k 上下文、4 并发 0 错误,并实现了自动故障切换与端点自动
> 登记;同一套能力在商业环境 1:1 映射到 vLLM/Triton + gateway 路由 + 可观测栈。」

## 数据可信度标注

| 标记 | 含义 |
|---|---|
| ✅ 实测 | 有内核/脚本运行输出作为证据 |
| 🟡 部分 | 样本少(如 5VU)或非全量对比 |
| 🔴 规划 | 尚未执行 |