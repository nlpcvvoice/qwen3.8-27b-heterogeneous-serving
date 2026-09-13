# 更正:两个 "router" 项目的边界考证(nlpcvvoice 名下)

时间:2026-09-13 05:32 · 状态:此前考证部分有误,本报告更正

## 一句话结论
本项目(heterogeneous-serving)的 router = **故障切换网关**(初衷高频词,✓ completed);而 MASTER「Track2 简历话术」里的 "router phase 表(Phase1 数据工程/Phase2 LoRA/CI 全绿)" 引用的状态**实际来自 nlpcvvoice/hybrid-llm-router-cost-optimizer**,属于另一个项目——我在简历叙事里把两个"router"混用了一处。

## 两 repo 事实对照(GitHub 已核)

| 项 | heterogeneous-serving | hybrid-llm-router-cost-optimizer |
|---|---|---|
| 项目定位 | 双加速器免费自托管 serving + 自动 failover | LLM 难度路由网关 + 成本优化 |
| "router"指代 | controller+router 探活转发网关(`app/router.py`,`app/controller.py`) | Router-1.5B 微调判题模型 + 级联 fallback |
| Roadmap/Phase | Failover router = Roadmap #1 **✓ done**(README:"TPU→GPU failover router ✓ done (live controller)") | Phase1 数据工程 complete / Phase2 LoRA in progress / CI passing |
| LoRA 出现 | ❌ 无 | ✅ Router-1.5B LoRA fine-tune(P2) |
| 与本项目初衷 | 高度相关(README 首句 "automatic failover TPU → GPU") | 独立项目 |

## 被我混用的证据链

| 出处 | 原文 | 出处真身 |
|---|---|---|
| MASTER「Track2」 | router phase 表 = "Phase1 数据工程完成 / Phase2 LoRA 训练就绪 / CI 全绿" | = hybrid-llm-router README Status 原文(逐字) |
| 本账号内核 | `tentenshishi/router-lora-classifier-debug`(ERROR) | 大概率 hybrid 项目产物(分类器),非 serving 项目 |
| 我之前报告 | 20260913_0456_router_lora_provenance.md 断定 Router/LoRA 属本项目 | **部分错误**,已更正 |

## 更正后归属表
- Failover/Router(故障切换、健康检查、app/controller+router):**本项目**,维持 ✓
- "Phase1 数据工程 / Phase2 LoRA / CI 全绿" 话术:属于 **hybrid-llm-router-cost-optimizer**;简历里要改为该独立项目的真实状态(或将本项目改为真正属于 serving 的完成态指标,如 failover 梯级证据)
- `router-lora-classifier-debug` 内核:归 **hybrid 项目**,其 ERROR 待 hybrid 项目修复(不在 serving 项目职责内)

## 待办(等指令)
- 修订 MASTER「Track2」话术:serving 项目改用"failover 网关梯级已验(Roadmap#1 done)"表述;hybrid 项目单独成行
- 更正 20260913_0456_router_lora_provenance.md(Router/LoRA 归属表述)
- serving 项目 README 无需改(其 "router" 语义正确)