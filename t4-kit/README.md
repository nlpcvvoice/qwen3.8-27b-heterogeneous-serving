# t4-kit — Kaggle T4(S) 独立 Job 包

> 用途:让**另一个 agent**在**其它项目目录**里,只读本文档,即可登录 Kaggle、向 2xT4 推送
> Qwen3.8-27B 推理任务、实时控制与取回结果。**只涉及 GPU T4,不含任何 TPU 逻辑。**
>
> 引擎能力与主仓库 GPU 内核完全一致:llama.cpp CUDA 源码编译 / 引擎缓存(跳过 25 分钟编译)/
> OpenAI 兼容服务 / 事件监控 / 配额管控 / 输出下载 / 自测基准。

| 项 | 值 |
|---|---|
| 硬件 | Kaggle GPU T4 x2(`NvidiaTeslaT4`,各 16GB),quantized Q4_K_M |
| 模型 | Qwen3.8-27B(GGUF `UD-Q4_K_M`) |
| 引擎 | llama.cpp `llama-server`(OpenAI 兼容 `/v1`) |
| 上下文 | ctx 98304 · q8_0 KV · parallel 4 slots · gpu-layers 全部 offload |
| 每周配额 | GPU 30h(超额自动等待;TPU 与本包无关) |
| 控制通道 | ntfy topic(推流事件 + CONTROL-stop 远程停机) |

---

## 1. 目录布局

| 文件 | 角色 | 说明 |
|---|---|---|
| `README.md` | 文档 | 本文件,唯一需要读的 |
| `start.sh` | 主入口 | 一键 CLI(test/push/watch/stop/download/cache-*) |
| `kit_config.py` | 常量 | 账号/数据集/路径,一次性配置 |
| `kaggle_login.py` | 认证 | 内存读 token,静默横幅,含配额查询(仅 GPU) |
| `push_job.py` | 推送 | 注入 CFG + 挂数据集 → 推 T4 内核 |
| `kernel/serve_t4_gpu.py` | 内核脚本 | 在 T4 内执行的负载(6 步流水线) |
| `watch_job.py` | 监控 | 轮询事件 + 内核状态;可选编译完自动停机 |
| `stop_job.py` | 停机 | 向内核发 CONTROL-stop,立即释放配额 |
| `download_output.py` | 下载 | 内核输出 → `<项目>/out/<版本>/` |
| `engine_cache.py` | 缓存 | `engine.tar.gz` 的 validate / upload / fetch |
| `check.py` | 诊断 | token/配额/内核/数据集状态 |
| `self_test.py` | 自测 | 离线验证,不耗配额 |

运行时状态(不提交):`<项目>/tmp/t4-kit/`(job 状态、kaggle_cfg、日志、staging)。

---

## 2. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | ≥3.10 | 建议用项目自带的 Venv |
| kaggle SDK | ≥2.1 | `pip install kaggle`(装在 Venv 内,勿污染系统) |
| Token | `reference/API-Token` | 读取顺序:env `KAGGLE_TOKEN_FILE` → 本目录 `reference/API-Token` → 项目根 `reference/API-Token` |
| 数据集 | 私有 `tentenshishi/qwen3-8-27b-q4-k-m-private` | 16.46GB GGUF 镜像;缺失时内核自动走 HF 下载兜底 |
| 出网 | 可访问 `api.kaggle.com` 与 `ntfy.sh` | 前者推送/查询,后者事件与控制 |
| 配额 | GPU >0.5h | `./start.sh check` 查看 |

---

## 3. 部署到新项目(只需一次)

| 步骤 | 操作 | 产出 |
|---|---|---|
| 1 | 拷贝本目录到目标项目根:`cp -r t4-kit <proj>/` | 携带全部依赖 |
| 2 | 放置 token:`<proj>/reference/API-Token`(或 `<proj>/t4-kit/reference/API-Token`) | 内存读取 |
| 3 | 项目内装 kaggle:`<proj>/Venv/bin/pip install kaggle`(或无 Venv 用 `python3`) | SDK 就绪 |
| 4 | `./start.sh test` | 离线全绿 = 可部署 |
| 5 | `./start.sh check` | 确认 token/配额/数据集正常 |

> 其它项目可用:输出在 `<proj>/out/`,状态在 `<proj>/tmp/t4-kit/`,互不干扰。

---

## 4. CLI 参考(其它 agent 只需这 8 条)

| 命令 | 动作 | 典型参数 |
|---|---|---|
| `./start.sh test` | 离线自测(0 配额) | — |
| `./start.sh check` | token/配额(kernel/dataset)诊断 | `--quiet` |
| `./start.sh push` | 推 T4 内核 | `--keepalive-min 120` · `--no-cache` · `--dry` |
| `./start.sh watch` | 事件监控 | `--stop-on built`(编译完自动停机) / `serve`(默认,随 keepalive) · `--timeout-min` |
| `./start.sh stop` | 立即停内核(释放 GPU) | — |
| `./start.sh download` | 下载输出 | `--out DIR` |
| `./start.sh cache-upload <tar>` | 上传引擎缓存(私有数据集) | `--sha` · `--msg` |
| `./start.sh cache-fetch` | 拉取+校验引擎缓存 | `--check` |

> `./start.sh push --help` / `watch --help` 有完整参数。

---

## 5. Job 生命周期

```
push ──▶ 排队 ──▶ 1 权重 ──▶ 2 引擎 ──▶ 3 llama-server ──▶ 4 隧道 ──▶ 5 READY ──▶ 6 自测 ──▶ keepalive ──▶ 停机
        │                 │   │                                   │                    ├─ 单流 tok/s 基准
        │                 │   └─ cache-hit(秒)                    │                    └─ 4 并发实测
        │                 └─ source build(~1452s)+打包 engine.tar.gz
```

**内核内部自动决策(无需干预):**

| 引擎来源 | 触发 | 耗时 |
|---|---|---|
| A 缓存命中 | 当前运行挂载了 `llama-server-qwen38-cache`(含 `engine.tar.gz`) | ~秒 |
| B 源码编译 | cmake + CUDA build(sm_75),编译完打包 `engine.tar.gz` 到 `/kaggle/working` | ~25 min |
| C cu124 wheel | A/B 不可用时 pip 安装 llama-cpp-python | ~2-5 min |

---

## 6. 事件流(ntfy topic,`watch` 实时可读)

| phase | 含义 | 关键字段 |
|---|---|---|
| `weights-mounted` / `weights-downloaded` | GGUF 就绪 | `path` |
| `engine-cache-hit` | 命中缓存,跳过编译 | `secs`,`size_mb` |
| `engine-built` | 源码编译成功 | `secs` |
| `engine-cached` | engine.tar.gz 已落盘(可下载时机) | `path`,`size_mb` |
| `engine-build-failed` | 编译失败(自动改走 wheel) | `tail` |
| `tunnel-url` | Cloudflare 公网 URL | `endpoint` |
| `serving` / `ready` | 健康通过 / OpenAI 接口可用 | `startup_secs`,`endpoint` |
| `benchmark` / `parallel4-test` | 自测指标 | `decode_tok_s`,`wall_secs` |
| `heartbeat` | 每 10 min 保活 | `up_min` |
| `auto-shutdown` / `stopped` | 结束原因 | `reason` / `served_min` |

---

## 7. 引擎缓存:一次编译、永久复用(核心省钱点)

```
第一轮(无缓存): push ──▶ watch --stop-on built ──▶ engine-cached ──▶ 自动 CONTROL-stop
                                   │                      │
                                   ▼                      ▼
                            download(取 engine.tar.gz)  cache-upload → 私有数据集
第二轮起:                 push(探测到缓存数据集 auto 附加) ──▶ cache-hit,~秒级启动
```

| 规则 | 值 |
|---|---|
| 缓存内容 | `bin/llama-server` + `bin/libllama-server-impl.so` + `lib/*.so`(薄壳 17KB 不可单独用) |
| 上传前校验 | `cache-validate`:必须是 gzip tar 且含 launcher + runtime libs |
| 拉取校验 | `cache-fetch`:sha256 对照 `manifest.json` + 内容结构复查 |
| 数据集可见性 | 私有 `tentenshishi/llama-server-qwen38-cache`,推送免费(不耗 GPU) |
| 修复脏缓存 | `push --no-cache`(强制源码编译)→ 重新 upload 覆盖 |

---

## 8. 配额与成本控制

| 项 | 建议 | 原因 |
|---|---|---|
| `--keepalive-min` | 120(默认) | 服务时长上限;到点内核自停机 |
| 缓存命中后 | `watch --stop-on built` 自动停 | 编译完即停,零空烧 |
| 超额 | `kaggle` 自动排队,不扣钱,只等额度 | 30h/周刷新可见 `check` |
| 捐赠币 | 免费;若选有币机器会消耗币 | 本包固定 `NvidiaTeslaT4`(免费档) |

---

## 9. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `No token found` | token 未放对路径 | 看"前置条件"表;或设 `KAGGLE_TOKEN_FILE` |
| `push` 说 cache dataset not found | 尚未上传过引擎缓存 | 属正常:第一轮走源码编译 |
| `watch` 报 `ntfy poll error` | 本机到 ntfy.sh 断连 | 重试即可;重启 `watch`;`stop` 需求窗口期内重发 |
| 内核一直 RUNNING、不 READY | 引擎编译中(约 25 min) | 看 `watch` 的 `engine-built` |
| `engine-build-failed` | 环境/网络异常 | push 会兜底 cu124 wheel;或 `--no-cache` 重试 |
| 必须立刻停机 | 超出预期 | `./start.sh stop`(内核收 CONTROL 后 os._exit,配额立即释放) |
| 内核已 COMPLETE 但日报没手动停 | keepalive 到点自停 | `watch --timeout-min` 建议 ≥ keepalive |

---

## 10. 安全铁律(与主仓库 MASTER 一致)

| 禁止 | 要求 |
|---|---|
| 打印/上传 token | 只显示掩码与长度;token 仅内存 |
| 写 `~/.kaggle` | 全部写入 `<项目>/tmp/` |
| trap: 提交运行时状态 | `tmp/` 均 gitignore |
| 混入 TPU 概念 | 本包内核/脚本/文档均无 TPU |

---

## 11. 其它 agent 的最小工作流(速查)

| 场景 | 命令序列 |
|---|---|
| 首次全流程 | `test` → `check` → `push` → `watch --stop-on built` → `download` → `cache-upload <out>/<v>/engine.tar.gz` → 完成 |
| 复用缓存再跑任务 | `push` → `watch --timeout-min 300` → 用 `ready` 事件里的 endpoint 调模型 → keepalive 自停或手动 `stop` |
| 只取结果 | `download`(`out/<version>/` 即内核产物) |