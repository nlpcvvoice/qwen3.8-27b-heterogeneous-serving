# 「简历/README 量化」是什么 · 定义与现状

时间:2026-09-13 04:56 · 状态:定义澄清 + 现状盘点(不可执行项为零)

## 这一格到底指什么
指把项目的"过程痕迹"翻译成**面试官可直接打分的完成态度量**,分落在两个载体:GitHub README 与求职简历(bullet/展示页)。不是独立功能模块,而是贯穿全项目的一套"度量完成态"纪律。

## 出处(MASTER 原文定位)

| 定义点 | 位置 | 要求 |
|---|---|---|
| 面试题 Q1 impact | MASTER 求职叙事框架 | QPS / TTFT / uptime 等指标回填 |
| 摩擦点 C | MASTER 就业摩擦点 | bullet 从"过程态"改"度量+完成态" |
| Track 2 素材表 | MASTER Track2 | serving tok/s 表、router phase 表、RAGAS 分数、8 repo→放大 3 个量化收尾 |
| Skills Used 注册表 | MASTER | 增量写入 GitHub README「Skills Used」 |
| 生产化模拟路线 | MASTER | Locust 压测报告 → README(面试背书) |
| impact 数据回填专项 | reports/20260910-182826-interview-impact-databackfill.md | 明确"哪些指标、去哪回填"的方法论 |

## 已完成 / 未完成对照

| 量化项 | 数值 | 载体 | 状态 |
|---|---|---|---|
| TPU serving 单流 | 126.8 tok/s(262k ctx) | README/report | ✅ 有 |
| GPU llama.cpp | 13.6 tok/s(2xT4 Q4_K_M) | README/report | ✅ 有 |
| MTP 提速 | ~16 tok/s(+33%,draft-mtp/2) | report 20260912_2140 | ✅ 有(README 待补) |
| bf16 vs Q4 对照/量化取舍 | 45 页报告 | report 20260912_2245 | ✅ 有 |
| router phase 表 | 完成态表述 | — | 🔶 文案待写(内核 ERROR 待修) |
| RAGAS 分数 | correctness 0.87 / context_precision 0.90 | — | 🔶 在 demo(保险 RAG)活体上才有实证 |
| uptime / TTFT / 压测 | 无 | — | 🔶 P2(Locust+Langfuse 落库后回填) |
| README「Skills Used」真实段落 | — | README | 🔶 尚未成段 |
| 正式简历 JSON/PDF | — | — | 🔶 未产出 |

## 一句话小结
"简历/README 量化" = **把已跑出来的数字(serving/量　化/router/RAGAS)整合成 README 定量说明 + 简历 bullet 的纪律**:一半素材已就绪(MTP/量化/双引擎),另一半卡在 router 修复、RAG demo 活体与压测可观测(P2)。

## 可立即做的零资源项(等指令)
- README 加「Benchmarks」段(用现有数字)
- 「Skills Used」段落落地
- router phase 表文案起草