# 免费 GPU/TPU 资源大普查 + 模型全球排名

| 报告日期 | 2026-09-10 19:30 | 用途 | 审阅后决定资源/模型战略 |
|---|---|---|---|
| 数据来源 | live web search(2026-09) | 结论 | ⚠️ 3 个重大发现,请审阅 |

## 一、免费 GPU/TPU 资源对比表(可用的)

| 平台 | 免费额度 | 硬件 | 连续上限 | 能跑我们项目? | 备注 |
|------|----------|------|----------|----------------|------|
| **Kaggle** | 30h GPU/周 + 20h TPU/周 | T4 / P100 / TPUv3-8 | GPU 12h / TPU 9h | ✅ **正在用** | TPU 配额我们已耗尽,GPU 低 |
| **Colab 免费** | 有 GPU,但**不公布固定额度** | T4(不确定) | 需求驱动,随时断 | 🟡 不确定 | 无固定配额,p95 不保 |
| **Colab Pro**($11.99/m) | +**15h 额外 GPU/周(Kaggle 上)** | 同 Kaggle | 12h | ✅ 划算 | 相当于给 Kaggle **加 15h/周**,不扣 Colab 额度 |
| **Colab Pro+**($49.99/m) | +**30h 额外/周** | 同 Kaggle | 12h | ✅ 划算 | 同理 |
| Hugging Face Spaces ZeroGPU | 免费 5 分钟/天 | 任意 | 短 | ❌ 不够 | 只够 demo |
| Lightning AI | 15 credits/月(≈22h T4) | T4 | 短 | 🟡 | 太少 |
| NVIDIA Inception | 免费云额度(申请制) | 多 | — | 🟡 需申请 | 学生/研究员可试 |

**🔑 关键发现 1:Colab Pro 的钱 = 给 Kaggle 加配额**
> Colab Pro/Pro+(Google 官方 2026 公告)给 **Kaggle 账户额外 15h/30h GPU/周**,用 Kaggle 相同硬件(T4/P100/TPUv3-8)。**最划算的扩容方案**。

## 二、模型全球排名表(主要测评站)

### Qwen3.8-27B(我们正在 serve 的模型 = **全精度 bf16**)

| 排行站 | 分数 | 世界排名 | 说明 |
|--------|------|----------|------|
| LMArena Elo | 待查 | 🟡 未独立收录(2026-08 发布,部分榜未评) | Qwen 官方自报,第三方刚起步 |
| **Artificial Analysis** Intelligence Index | **52** | **约 #21~22 / 367 全部 | #8~10 / 开源权重** | 此分数为 **xhigh(xhigh 推理档)** |
| BenchmarkList ECI | 155.29 | **#20 / 346 全部**;开源 **#5 / 137** | WebDev Arena 1595,Context Arena **#1** |
| BenchLM | 59.4/100 | **#71 / 399** | 90% 区间 47.9–71.0 |
| AndroidWorld | 81.9% | #4 / 7(开源差) | 顶级开源 agent 能力 |
| MathVision | 94.6% | **#4 / 35**;开源 #1 | 多模态数学 |
| 同排位参照 | Qwen3.8 Flash | DeepSeek V4 Flash 0731 | Qwen3.8 Max |

> **注意**:或车榜中有两档分数:AA **52**(官方/AA 测,推理 xhigh)却排在 ~22 名,说明 27B 全精度是"**小身材、高智商**",在 27B 级开源模型里顶级,但世界总榜被 70B+ 巨型模型压过。

### Qwen3.8 Flash / Flash-Next(我们测评的"世界级"参照)

| 排行站 | 分数 | 排名 | 参数 | 价格 |
|--------|------|------|------|------|
| **Artificial Analysis** | **46** | **#5 / 112 同规模段** | 180B MoE(6B active) | $0.15 / $0.47 per M |
| AndroidWorld | 84.5% | **#2 / 7** | 125B | $0.15/$0.47 |
| MathVision | 95.7% | **#2 / 35** | 125B | — |
| DataCamp 评测 | 胜 Opus 4.6 Max | SWE-bench Pro 62.5 vs 53.4 | 125B | — |
| LMArena Agent | 59.1 | #24 / 40 | Flash-Next | — |

> Qwen3.8-Flash = 顶配"聪明快",27B = 顶配"小又快",两者是兄弟,Flash 在 180B MoE 里把 27B 打趴,但还是同一家族。

### DeepSeek V4.1 Flash(2026-09-10 刚发布!)

| 排行站 | 分数 | 排名 | 参数/架构 | 价格 | 备注 |
|--------|------|------|-----------|------|------|
| **Artificial Analysis** | **~50** | 约 #15(0731 版 52)→ V4.1 更高 | 552B MoE,预编 8B/解码 16B active | $0.30 / $1.20 | V4.1 刚出,AA 分数待最终 |
| Vals Index | +4.3 vs 0731 | 大幅提升 | 1M ctx,384k 输出 | — | 9/10 发布 |
| DeepInfra | — | CED 架构,全球 KV 缓存降 4x | 552B | $0.30/$1.20 | 便宜 |

> **关键发现 2:DeepSeek V4.1 Flash ="今天的'聪明新秀"**,1M 上下文,价格只有 Pro 的 1/3,可本地部署(MIT 许可证)。**未来若要用免费模型,优先考虑它**。

## 三、报告:Colab 能否 API 接入用它的免费模型?

| 问题 | 答案 |
|------|------|
| Colab 有"免费模型 API"吗? | **没有**。Colab 官方 API 是 **runtime 管理 API**(创建/删除运行时,beta 且 allowlist 制),**不是模型推理 API** |
| 那 Colab 免费模型指什么? | Colab 笔记本里**调用 Gemini API 免费层**($0,需 AI Studio key),不是 Colab 的 GPU 跑模型 |
| 能不能拿 Colab GPU 服务我们自己部署的模型? | **能,但要改架构**:Colab 是一个"临时虚拟机",可跑你的 vLLM/llama.cpp,再开 tunnel → 变成临时 endpoint。**但 quota 不公布,随时被掐**,只适合短实验 |
| 结论 | 🟡 Colab **不能**作为"免费模型 API"接入;但可作为"临时 GPU 虚拟机"用(不稳定)。**替代:GitHub 上已经有 Colab/Kaggle tunnel 自托管方案** |

## 四、报告:Antigravity 的 quota(3.8 flash)能给我(opencode)用吗?

### 先弄清楚 Antigravity 的 quota 是啥

| 项 | 事实 |
|----|------|
| 是什么产品 | Google 的 agent 编程应用,提供 **UI(App) + CLI(`agy`)**;quota = 产品内模型额度 |
| 有哪些模型可用 | 核心: Gemini 3.8/3.7/3.6 Flash、Gemini 3.1 Pro、Claude Sonnet/Opus 4.6、GPT-OSS-120b。**没有 Qwen3.8!!!** |
| quota 机制 | free/Plus: 每周刷新;**Pro/Ultra**: 每 5 小时刷新 + 更高周额度;共享 quota(所有模型共享一个池子) |
| 能否被 opencode 直接调用? | **不行(官方)**。Antigravity 不暴露通用 API,token 在内部系统钥匙串,版权条款禁止提取 |
| 社区 hack(⚠️违规险) | 有人写 `antigravity-cli-mcp` 把 agy CLI 包成 MCP 工具给 opencode 调用;**Google 已因"提取 token 供外部使用"封号**(连 Gmail 都丢了) |
| 安全结论 | ❌ **绝对不要提取 Antigravity token 给 opencode**。要么在主界面/CLI 用它的 agent(作 sub-agent),要么放弃 |

### "只给 AGY 用" 的准确含义

| 用法 | 可以吗 |
|------|--------|
| 用 Antigravity 本身(UI/CLI)当 coding agent | ✅ 官方支持,变成 opencode 的 sub-agent 桥接也行(不提取 token) |
| 把 Antigravity quota 模型当 opencode 的主 model | ❌ 无 API |
| 提取 token 给别的工具 | ❌ 封号风险高(官方已封) |
| Qwen3.8 flash 在 Antigravity 里? | ❌ **根本没有 Qwen 模型**,全是 Google/Claude/OCR 系 |

## 五、给决策的落地建议

| 优先 | 动作 | 成本 |
|------|------|------|
| ① 扩容 GPU | Kaggle 绑定 **Colab Pro ($11.99/m)** → Kaggle 每周 +15h GPU | 💰 |
| ② 未来模型候选 | **DeepSeek V4.1 Flash**(本地可跑,1M ctx,便宜) | 0~$ |
| ③ 免费 quota 陷阱 | Colab 不稳定、Antigravity 无 API | 了解即可 |
| ④ 当前计 | 继续用 Kaggle TPU/GPU 免费配额,不够再买 Colab Pro | 0 |

---
**红线**:token 绝不打印/上传 | Antigravity token 绝不提取,后果=封号 | 本报告为 2026-09 数据快照,排名会随时变化 | 写死每日刷新