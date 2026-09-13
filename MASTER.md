**注意事项**
文件access严格限制在当前目录。绝对不允许access当前目录以外的任何目录或者文件。
如果在安装某些依赖过程中临时需要access某些在当前目录之外的文件，需要征得user的允许。

安装依赖全部在当前目录下的Venv目录下，不要污染系统环境。
all tmp files in ./tmp/ folder.

如果需要sudo，请把命令写到md文件，我来执行。

每次写文档，文档命名方式是date+time+summary，文档内容要求高度图形化，可以用表格，但是尽量少的文字，尽可能不要有代码。每次写完文档，put your report in ./reports/

不可以有任何文件上传。



# Project Rules & Qwopus Reasoning Protocol

## 1. Identity & Tone
- Your goal is to provide high-precision code and architectural guidance.
- Maintain a professional, senior engineer-level tone. Avoid conversational filler.

## 2. Reasoning Protocol (MANDATORY)
- **THINK FIRST**: For every request, you MUST start your response with a `<thought>` block.
- **Content of Thought**:
  - Breakdown the user requirements.
  - Scan relevant files (@filenames) to check for side effects.
  - Plan the specific lines to be changed.
   - Assess potential memory risks given the local environment (15Gi RAM, no GPU).
- **Chain of Thought**: If the solution is complex, use step-by-step numbering inside the thought block.

## 3. Coding Standards
- **Surgical Edits**: Prefer partial search-and-replace over full file rewrites. Match existing indentation and style exactly.
- **Type Safety**: In Python/TypeScript, always check definitions before usage.
- **Clean Output**: Do not include introductory text like "Sure, I can help." Go straight from `</thought>` to the code block or action.
- **Validation**: After editing, provide a brief suggestion on how to test the change (e.g., specific CLI command).

## 4. Local Environment Constraints
- **Hardware**: GCP Workbench Notebook - Intel Xeon @ 2.20GHz (4 vCPUs), 15Gi RAM, No GPU, Debian 12.
- **Optimization**: Be concise in explanations to keep the context usage efficient.

---


**The Four Principles**
1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

State assumptions explicitly — If uncertain, ask rather than guess
Present multiple interpretations — Don't pick silently when ambiguity exists
Push back when warranted — If a simpler approach exists, say so
Stop when confused — Name what's unclear and ask for clarification
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

Combat the tendency toward overengineering:

No features beyond what was asked
No abstractions for single-use code
No "flexibility" or "configurability" that wasn't requested
No error handling for impossible scenarios
If 200 lines could be 50, rewrite it
The test: Would a senior engineer say this is overcomplicated? If yes, simplify.


3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting
Don't refactor things that aren't broken
Match existing style, even if you'd do it differently
If you notice unrelated dead code, mention it — don't delete it
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused
Don't remove pre-existing dead code unless asked
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform imperative tasks into verifiable goals:

Instead of...	Transform to...
"Add validation"	"Write tests for invalid inputs, then make them pass"
"Fix the bug"	"Write a test that reproduces it, then make it pass"
"Refactor X"	"Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

---

** 严格控制不可以将目录reference/API-Token的内容打印在会话界面上，只能在内存读取 **

---

## Kaggle 登录复用说明(本目录专用)

| 项 | 约定 |
|---|---|
| 模块 | kaggle_login.py(已实现,必须用 ./Venv/bin/python 运行) |
| 依赖 | ./Venv 内已装 kaggle 2.2.4 + requests(本机python无kaggle包) |
| Token源 | reference/API-Token(只读,内存读取,不打印) |
| 运行环境 | ./Venv/bin/python |
| 环境隔离 | KAGGLE_CONFIG_DIR=./tmp/kaggle_cfg,不写~/.kaggle |
| 认证方式 | Bearer(KGAT_)/KGRT_自动刷新 |

**使用步骤(已验证)**
| 步骤 | 代码 | 说明 |
|---|---|---|
| 1 初始化 | `kl.login()` | 内存读Token→export KAGGLE_API_TOKEN/os.environ |
| 2a 官方API | `api=kl.get_kaggle_api()` | 已认证KaggleApi实例,可调 kernels_list 等 |
| 2b HTTP会话 | `s=kl.auth_session()` | requests Bearer会话,自定义REST调用 |
| 2c 查余额 | `kl.quota()` | 返回 GPU/TPU 每周已用/剩余/总额小时 + 刷新时间(官方 SDK quota_view) |
| 3 校验 | `print(kl.verify())` | 只显示掩码/长度/ok状态,不显Token原文 |

```python
import kaggle_login as kl
kl.login()              # 内存读取+export KAGGLE_API_TOKEN
s = kl.auth_session()   # Bearer会话,用于requests调用
api = kl.get_kaggle_api()  # 官方kaggle.api已认证实例
print(kl.verify())      # 脱敏校验,只看ok/status
```

| 禁止 | 要求 |
|---|---|
| 打印/日志Token原文 | 只展示掩码与长度 |
| 写~/.kaggle | 保持当前目录内 |
| 上传Token/密钥文件 | 本地内存使用 |

---

## 求职叙事框架 (AI Engineer 面试对齐 · 每步进展都要回填)

| 面试问题 | 本项目定位 | 状态 |
|---|---|---|
| 1 Business use case & impact | 免费高规格开源 LLM 自托管 Serving;支撑 agent/tool 工作负载;对比按量计价 API 的成本优势 | 双引擎在线;impact 指标待补 (QPS/TTFT/uptime) |
| 2 Model architecture & trade-offs | 异构加速器部署: TPU v5e-8 bf16(vLLM) vs 2xT4 Q4_K_M(llama.cpp);取舍: 全精度 vs 4bit、262k vs 16k ctx、引擎-硬件约束(vLLM 需 Ampere+);付费 H200 层(BF16 262k)补 long-context 空档 | 成型 |
| 3 Model-level work | 推理优化: 量化选择/显存 offload/parallel slots/连续批处理;自定义评测: tool-call 断言、4 并发实证 | 成型 |
| 4 Production challenges | 模型版本化(private dataset + GGUF pinned);监控(ntfy 事件/心跳/自检);可观测规划(Langfuse/MLflow);故障切换(TPU→GPU);付费层运营(H200 SleepMode 快照/scale-to-zero/$0 闲置) | 部分完成 |
| 5 Evaluation methodology | 探针脚本、单流 126.8 tok/s benchmark、parallel4 并发测试、LLM-as-judge、bf16 vs Q4 对照 | 部分完成 |

## 就业摩擦点 & 三轨最快就业 (2026-09 深搜核实 · 采纳外部反馈)

### 共识诊断(认可)
| 摩擦点 | 影响 | 对策 |
|---|---|---|
| A 无"可直接点开"demo | 面试官 5 分钟看不到东西 → 信任打折 | Track1 |
| C 简历"度量+完成态"密度 | bullet 多为过程态 | Track2 |

### Track 1 · 可点 demo(第 1-3 天 · 最快通道)
| 步骤 | 内容 | 工时 | 成本 |
|---|---|---|---|
| 1a | insurance-rag-agent **包 FastAPI** → HF Space **免费 CPU 常驻** | 1-2天 | $0 |
| 1b | README 加 curl 例子 + 5 个 RAGAS 分数 → 简历首行 `live at …` | 0.5天 | $0 |
| 1c | 27B ZeroGPU 演示(见 Demo 计划) | P1 stretch | $0 |

> 为什么 RAG 先于 27B serving:CPU 可托管=当天上线,不依赖 GPU 配额/内核时段;且 RAG/agents 是最大岗类(2026: 3038 岗)。

### Track 2 · 简历完成态(第 1-5 天并行)
| 素材 | 简历话术 |
|---|---|
| serving tok/s 表 | "TPU vLLM 126.8 tok/s / GPU llama.cpp 13.6 tok/s,自动故障切换" |
| router phase 表 | "Phase1 数据工程完成 / Phase2 LoRA 训练就绪 / CI 全绿" |
| RAGAS 分数 | "0.87 correctness / 0.90 context_precision / 4 层失败面覆盖" |
| 8 repo | 只放大 3 个(serving / RAG / router),全部量化收尾 |

### Track 3 · 契约/PT 最快变现(第 1 天起 · 首单 1-4 周)
| 渠道 | 定位 | 进场速度 |
|---|---|---|
| Upwork | 自托管 LLM + 微调 + 评测 niche($60-150/hr 起) | 24h-1周首单(平台费5-20%) |
| Contra | 0% 平台费 + Contra Payments | 适合稳定长单 |
| Toptal | 筛选率<3%,过则 $90-150+/hr | 2-4周 |
| FT 投递池 | fastaijobs(163入门岗) · landedjobs · aidevboard · AgenticCareers | 与 T1/T2 并行 |

### 目标岗与预期
| 岗型 | 匹配 repo | 薪资带 | 预期时间 |
|---|---|---|---|
| LLM/Inference Engineer | serving + p3 toolchain | $197-284k | FT 3-6月 |
| AI Engineer (RAG/Agent) | insurance-rag + eval_suite + personas | $120-200k | FT 3-6月 |
| 契约:LLM 自托管/微调 | kaggle-dual-t4-qlora + serving | $60-450/hr | **1-4周首单** |

## Skills Used 注册表(增量写入 GitHub README "Skills Used")

| Skill | 落地物 | 状态 |
|---|---|---|
| LLM Serving (vLLM / llama.cpp / OpenAI 兼容) | TPU+GPU 双内核 | ✓ |
| 模型优化 (GGUF Q4_K_M / offload / 批处理) | GPU 引擎 --parallel 4 | ✓ |
| 异构调度 (TPU/GPU via Kaggle API, machine_shape) | push/probe 脚本 | ✓ |
| 工具调用解析 (qwen3_coder parser) | TPU 评测链路 | ✓ |
| 故障切换 / 健康检查 | router (规划) | P1 |
| 可观测 (Langfuse/MLflow trace + eval) | 指标落库 | P2 |
| Load test (Locust + persona corpus) | 压测报告 入 README | P2 |
| LLM-as-judge 评测 | 评测集 + 对照实验 | P2 |
| 付费 GPU 层 (Modal H200, BF16 262k, serverless) | 三加速器基准表 + 256k 实测 + SleepMode 快照 | ✓ |

## 生产化模拟路线 (免费)

```
personas (LangGraph 离线生成真实 prompt 分布)
   → Locust 并发压测 (RPS/TTFT/p50/p95/错误率)
   → Langfuse/MLflow 观测 (token/成本/轨迹/eval)
   → 阈值门禁 → 压测报告入 README (interview 背书)
```

## Demo 计划 (Hugging Face Spaces ZeroGPU · 免费公开演示)

### 目标
公开一个 ZeroGPU Space:浏览器直接对话 Qwen3.8-27B(Q4_K_M llama.cpp),直观展示"自托管开源 LLM Serving"成果 → README 放链接(面试背书)。
> 优先级:先上 RAG CPU demo(Track1,当天可上线),27B ZeroGPU 为 stretch P1(受 5min/天 quota 制约)。

### 资源与配额(2026-09 官方 docs 核实)

| 项 | 值 |
|---|---|
| GPU | NVIDIA **RTX Pro 6000 Blackwell**:large 48GB(quota×1)/ xlarge 96GB(quota×2) |
| 免费 host | 每人 ≤**2 个** ZeroGPU Space(邮箱验证 + 账户≥30 天) |
| 免费 quota | **5 GPU-min/天**(另有非官方"≈3 次运行/天"限次) |
| PRO($9/m) | 40 min/天 + 队列最高优先;超出 $1/10min credits |
| Space 基线 | 16GB RAM / 2 vCPU / 50GB 盘,GPU 按请求动态分配 |

### 部署方案

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 模型 | Qwen3.8-27B **Q4_K_M GGUF ≈18GB** → HF model repo(私有→公开) | P1 |
| 2 Space | `demo/` Gradio chat + llama-cpp-python(CUDA)+ `@spaces.GPU` | P1 |
| 3 硬件 | `hardware=zero-a10g large`(48GB,quota 1×,冒 96GB xlarge 会吃 2×) | P1 |
| 4 验证 | build→RUNNING→真问答;quota 5min 提前预热再演示 | P1 |
| 5 公开 | Space 链接 + 截图入 README | P2 |

### 决策与约束

| 项 | 决定 | 原因 |
|---|---|---|
| 演示模型 | 27B Q4_K_M(18GB)| bf16 需 54GB+ 超 xlarge;TPU 内核空间不可迁移 |
| 引擎 | llama.cpp / llama-cpp-python | 与 GPU 内核同源,OpenAI 兼容接口 |
| 负载 | 仅 demo 单用户 | ZeroGPU 是共享按需,不是常驻服务 |
| 压测 | 仍走 Kaggle/Colab | demo 不承担压测任务 |

### 红线

| 禁止 | 要求 |
|---|---|
| 密钥落库 | api_key/token 只走 `.env`,Space 内绝不写死/打印 |
| 大文件污染 repo | 模型走 HF Hub,不入 Git;`git lfs` 不用 |
| demo 代码 | 进 `app/demo/`(脱敏),可随主 repo push |

## GitHub 卫生规则

| 推 | 不推 |
|---|---|
| kaggle-tpu-lab/ (内核+launch+patches) | tmp/ (16GB 权重) |
| app/ (push/watch 骨架, 脱敏) | reference/API-Token |
| kaggle_login.py (只读路径, 无明文) | Venv/ · session-*.md · Untitled.ipynb |
| reports/ 规划+报告 | opencode.json (含实时 API Key) |
