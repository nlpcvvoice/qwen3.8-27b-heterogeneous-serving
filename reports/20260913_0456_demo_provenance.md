# 可点 Demo 归属考证 · Track1 / ZeroGPU 从何定义

时间:2026-09-13 04:56 · 状态:考证完成,与 Router/LoRA 同批复查

## 结论
**"可点 demo"是本项目 MASTER.md 自己定义的 Track1 计划(两天级上线的自研物),不是从 GH 其它项目搬来的。** ARahim3/kaggle-tpu-lab 里没有任何 RAG/HF Space/ZeroGPU demo 内容。

## 证据(本项目文档原文定位)

| 规划项 | 位置 | 内容 |
|---|---|---|
| 摩擦点诊断 | MASTER.md 就业摩擦点 A | 面试官 5 分钟看不到"可直接点开"的 demo |
| Track1 行动 | MASTER.md Track 1 | 1a 保险 RAG agent 包 FastAPI → HF Space 免费 CPU 常驻;1b README 加 curl + 5 个 RAGAS 分数;1c 27B ZeroGPU(见 Demo 计划) |
| ZeroGPU 专项 | MASTER.md「Demo 计划 (HF Spaces ZeroGPU)」 | 免费配额、部署方案(模型=Qwen3.8-27B Q4_K_M GGUF≈18GB + llama-cpp-python)、红线、公开流程 |
| 外部采纳 | reports/20260912-001000-external-aiengineer-feedback-eval-and-fastest-job-path.md | Track1 来自外部 AI Engineer 反馈,于 2026-09-12 定案 |
| 成本/配额预研 | reports/20260910-193000-free-resources-model-rankings.md | ZeroGPU/HF 免费资源横向对比 |

## 与 ARahim3/kaggle-tpu-lab 的边界

| 项 | ARahim3 | 本项目 demo 计划 |
|---|---|---|
| RAG | 无 | insurance-rag-agent(FastAPI) |
| HF Space / ZeroGPU | 无(只在 Kaggle TPU 上 serve) | Track1 + Demo 计划(公开浏览器演示) |
| 27B GGUF demo | 无(用 bf16 safetensors + TPU) | Q4_K_M GGUF + llama-cpp-python(ZeroGPU) |
| 重复对象 | —(不重叠) | 仅"27B 免费推理"主题相同,形态不同 |

## 备注(防止再混淆的一句话)
- "可点 demo"来源 = 外部反馈(2026-09-12)+ MASTER Track1/Demo 计划 → 本项目
- 之前对话中我把 ARahim3 的启动时延数字(40→22→6min)引入只是**技术参考**,与本项目的 demo 计划无关