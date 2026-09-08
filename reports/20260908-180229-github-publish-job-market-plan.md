# GitHub 发布 + 求职竞争力规划

| 项 | 值 |
|---|---|
| 日期 | 2026-09-08 |
| 目标 | 把"TPU+GPU 异构 Qwen3.8-27B Serving"做成可面试背书的 production 级项目 |
| GitHub 账号 | nlpcvvoice (gh 已认证, scope: repo/workflow) |

## 1. 市场结论 (检索背书)

| 来源要点 | 对项目的启示 |
|---|---|
| 面试官要"判断力": 为什么选/不选什么、失败模式、指标为什么重要 | README 必须讲 trade-off, 不是技术清单 |
| "个人项目要像工程"：结构良好代码、架构决策记录、可复现实验、被使用/被评审的证据 | 提交记录整洁、报告入 repo、metrics 可复现 |
| README > 代码本身; 配小架构图; 置顶 3-5 个仓库 | 本仓库置顶 + 架构图 |
| 端到端所有权: data→train/opt→serve→observe→eval | 完整生命周期都出现在 README |

## 2. GitHub 仓库方案

| 项 | 方案 |
|---|---|
| 仓库 | qwen3.8-27b-heterogeneous-serving (public) |
| 内容 | kaggle-tpu-lab/ · app/ · reports/ 精选 · 架构图 · README |
| 排除 | tmp/ (16GB) · reference/API-Token · Venv/ · opencode.json(含实时key) · session-*.md |
| README 骨架 | 问题→架构图→trade-off 表→实测数字→Skills Used→如何复现 |

## 3. README 叙事 (对齐 5 面试问题)

| 面试问题 | README 呈现 |
|---|---|
| Business impact | 免费高规格 Serving;$0 vs API;支撑 agent/工具负载 |
| Architecture & trade-offs | 双引擎对比表 (见下) |
| Model-level work | 量化/offload/parallel4/批处理/评测断言 |
| Production | 版本化/监控/心跳/故障切换 |
| Evaluation | benchmark + parallel4 + LLM-as-judge |

### Trade-off 表 (核心材料)

| 维度 | TPU v5e-8 | GPU 2xT4 |
|---|---|---|
| 精度 | bf16 全精度 | Q4_K_M 4bit |
| 上下文 | 262144 | 16384 |
| 吞吐 | 126.8 tok/s | 13.6 tok/s |
| 引擎 | vLLM | llama.cpp |
| 排队 | 0-3h | 即时 |
| 定位 | 主引擎 | 兜底 |

## 4. 生产化模拟路线 (免费)

| 层 | 工具 | 动作 |
|---|---|---|
| 真实流量 | LangGraph personas | 离线生成 20+ persona 的 prompt 语料(prompt 长度/工具调用/推理开关分布真实化) |
| 并发压测 | Locust (llm-locust) | RPS 递增, 测 RPS/TTFT/p50/p95/错误率/饱和点 |
| 观测 | Langfuse 自托管 或 MLflow (OSS) | token/成本/轨迹/eval 落库; 仪表盘 |
| 门禁 | 阈值脚本 | latency budget / error rate 超标即红, 产出报告 |
| 背书 | Reports | 压测报告+图表入 README |

**LangGraph 用法裁决**: 不用 LangGraph 做压测本体(压测成熟方案=Locust);用它离线生成真实分布的 persona 语料, Locust 回放。answer 面试官时这句最有说服力。

## 5. 免费 LLM (OpenRouter 之外)

| 提供商 | 特点 |
|---|---|
| Groq / Cerebras | 快, 免费无需卡, 推理部署型 |
| NVIDIA NIM | 120+ 开源权重 |
| GitHub Models | Copilot 随附 |
| Gemini AI Studio | Flash 免费(RPM 低, 非EEA 数据用于训练) |
| Cloudflare Workers AI | 10k neuron/天 |
| HuggingFace / Mistral / Z.ai / SiliconFlow | 各有免费额度 |
| 自建聚合 (FreeLLMAPI / steadyroute) | 拼所有免费档 + failover |

**本项目自产自销**: 我们自己的 TPU/GPU Qwen3.8 端点免费可用 → persona 生成器直接 dogfood 本服务。

## 6. 待确认

1. 仓库名/公开或私有? (建议 public + 置顶)
2. 是否现在就建仓库推首版 (README 我先起草)?
3. 生产化模拟: 先做 Locust 压测, 还是 Langfuse 观测先?