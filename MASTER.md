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
| 2 Model architecture & trade-offs | 异构加速器部署: TPU v5e-8 bf16(vLLM) vs 2xT4 Q4_K_M(llama.cpp);取舍: 全精度 vs 4bit、262k vs 16k ctx、引擎-硬件约束(vLLM 需 Ampere+) | 成型 |
| 3 Model-level work | 推理优化: 量化选择/显存 offload/parallel slots/连续批处理;自定义评测: tool-call 断言、4 并发实证 | 成型 |
| 4 Production challenges | 模型版本化(private dataset + GGUF pinned);监控(ntfy 事件/心跳/自检);可观测规划(Langfuse/MLflow);故障切换(TPU→GPU) | 部分完成 |
| 5 Evaluation methodology | 探针脚本、单流 126.8 tok/s benchmark、parallel4 并发测试、LLM-as-judge、bf16 vs Q4 对照 | 部分完成 |

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

## 生产化模拟路线 (免费)

```
personas (LangGraph 离线生成真实 prompt 分布)
   → Locust 并发压测 (RPS/TTFT/p50/p95/错误率)
   → Langfuse/MLflow 观测 (token/成本/轨迹/eval)
   → 阈值门禁 → 压测报告入 README (interview 背书)
```

## GitHub 卫生规则

| 推 | 不推 |
|---|---|
| kaggle-tpu-lab/ (内核+launch+patches) | tmp/ (16GB 权重) |
| app/ (push/watch 骨架, 脱敏) | reference/API-Token |
| kaggle_login.py (只读路径, 无明文) | Venv/ · session-*.md · Untitled.ipynb |
| reports/ 规划+报告 | opencode.json (含实时 API Key) |
