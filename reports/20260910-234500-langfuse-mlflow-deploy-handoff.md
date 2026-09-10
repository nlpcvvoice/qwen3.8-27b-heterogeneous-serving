# 记账本部署交接方案 — Langfuse v4 vs MLflow 3(交给另一个 agent 执行)

| 交接方 | 本机(opencode) | 执行方 | 另一个 agent(用户个人电脑) |
|---|---|---|---|
| 目标机器 | ❌ 本机不装 | ✅ 用户个人电脑(有 Docker) | 日期 2026-09-10 |
| 交付物 | 一份可执行的部署+接入方案 | 产出 | 可验收的后端 + 埋点脚本 |

> 用途:在**个人电脑**上自托管可观测后端(Langfuse 或 MLflow),远端 Kaggle TPU/GPU 内核仍在云上,追账客户端脚本在本机跑,本机不部署。

## 一、总决策

| 项 | Langfuse v4 🖍️ | MLflow 3 📚 | 建议 |
|---|---|---|---|
| 形态 | Web+API+4 数据库(重) | 单个 server(轻) | 有 Docker → 两个都可 |
| 装法 | `git clone langfuse` + docker compose | `pip install mlflow` 或官方 docker 镜像 | MLflow 更快上手 |
| 追账重点 | 对话流大屏、Prompt 版本、判分面板 | OpenAI 全自动 trace、模型版本登记、evaluation 分数 | Langfuse:可观测展示 / MLflow:实验&评审 |
| 共同点 | 数据 100% 留个人电脑,不向云发 | 同左 | 满足隐私红线 |
| ⭐ 推荐 | **Langfuse v4**(对话 trace 可视化) | MLflow 作为第 2 套(模型注册+评估) | 时间紧 → 先 Langfuse |

## 二、目标架构(个人电脑侧)

```
┌──────────── 用户个人电脑 ────────────┐
│  ① Langfuse v4 (docker compose)     │   http://localhost:3000
│     └─ web:3000 + worker + postgres17│
│        + clickhouse25.12 + redis7 +  │
│        minio(S3)                     │
│  ② (可选) MLflow 3 pip server       │   http://localhost:5000
│  ③ 埋点脚本:replay_eval.py /        │   读远端 cloudflare /v1 端点
│     probe 客户端 (OpenAI SDK 包装)    │   发 trace 到 ①②
└──────────────────────────────────────┘
        ▲ trace 输出
        │  HTTPS
   Kaggle TPU vLLM(bf16)  ·  GPU llama.cpp(Q4)   ← 不动,仍在云上
```

| 角色 | 跑在哪 | 端口 | 说明 |
|---|---|---|---|
| Langfuse | 个人电脑 Docker | 3000(web/API) | 其余服务仅内网 |
| MLflow | 个人电脑 venv | 5000 | 也可用官方镜像 |
| 追账客户端 | 个人电脑 venv | — | 调远端端点 + 本地写 trace |
| 内核/服务 | Kaggle 云 | tunnel /v1 | 零改动、零配额消耗 |

## 三、部署方案 A:Langfuse v4(推荐先做)

| 前置 | 命令 | 说明 |
|---|---|---|
| Docker+Compose | `docker --version` | Docker Desktop 即可 |
| 拉代码 | `git clone https://github.com/langfuse/langfuse && cd langfuse` | 官方仓库自带 compose |
| 改密 | 编辑 `docker-compose.yml` 中所有 `CHANGEME` | 改 DATABASE_URL、CLICKHOUSE、REDIS 密码 |
| 单机模式 | 设 `CLICKHOUSE_CLUSTER_ENABLED=false` | 单容器必需 |
| 启动 | `docker compose up -d` | 等 2~3 分钟 web 显示 Ready |
| 验证 | 浏览器 `http://localhost:3000` | 建本地账号(数据不出机) |
| 健康检查 | `curl http://localhost:3000/api/public/health` | 返回 ok |

| 版本钉子(踩坑点) | 要求 |
|---|---|
| Langfuse | 镜像 `docker.langfuse.com/langfuse/langfuse-web:4`(主分支 v4) |
| ClickHouse | **≥ 25.12**(v4 的文字索引/JSON 类型硬性最低版) |
| Postgres | 17 |
| Redis | 7 |
| 时区 | **全部容器强制 UTC**,否则 trace 时间乱 |

| 备份/升级 | 命令 |
|---|---|
| PG 备份 | `docker compose exec -T postgres pg_dump -U postgres postgres > backup.sql` |
| ClickHouse 备份 | `docker compose exec -T clickhouse clickhouse-client --query "BACKUP DATABASE default TO Disk('default','backup.zip')"` |
| 升级 | `docker compose pull && docker compose up -d` |
| 停(勿删数据) | `docker compose stop` / 彻底删 `docker compose down -v` |

## 四、部署方案 B:MLflow 3(可选,第二套)

| 步骤 | 命令 | 说明 |
|---|---|---|
| 建 venv | `python3 -m venv ~/mlflow-venv && source ~/mlflow-venv/bin/activate` | 不污染系统 |
| 安装 | `pip install "mlflow>=3" mlflow-tracing openai requests` | mlflow-tracing 体积小 95% |
| 启动 server | `mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///$HOME/mlflow/mlflow.db --default-artifact-root $HOME/mlflow/artifacts` | 仅本机监听 |
| 验证 | 浏览器 `http://localhost:5000` | 建 experiment `qwen38-serving` |

| MLflow 能力(供验收) | 用法 |
|---|---|
| OpenAI 一键 trace | `mlflow.openai.autolog()` + `mlflow.set_experiment("qwen38-serving")` |
| 远端端点也能追 | OpenAI SDK 指 `base_url=<远端>/v1` → 全自动捕获 prompt/延迟/token/工具调用 |
| 手工 span | `@mlflow.trace` 装饰器包装 router/probe 函数 |
| streaming | 自动拼合输出,chunk 存 Event 页 |
| 模型登记 | regiter bf16(TPU) vs Q4_K_M(GPU) 两个 model version |

## 五、埋点/接入设计(两端对接点)

| 数据源(本机脚本) | 动作 | 打到哪里 |
|---|---|---|
| `eval_suite.py --live` | OpenAI SDK 包装 + `base_url` → 远端内核 | Langfuse 或 MLflow |
| `locustfile.py`(压测) | 每个请求 client 包装 | 同上 |
| probe/controller 心跳 | 手工 span(仅记录 start/ready/duration) | 同上 |
| 考试卷断言结果 | 每条 case 打分 → 单独 experiment | MLflow metrics |

| Langfuse 接入(推荐) | 示例参数 |
|---|---|
| Python SDK | `pip install langfuse openai` |
| 初始化 | `Langfuse(host="http://localhost:3000", public_key=…, secret_key=…)`(从 `.env` 读,不写死) |
| 包装对象 | `langfuse.openai.OpenAI()` 替代裸 `OpenAI()`,其余代码不变 |
| 打钩 | trace 完成后 `trace.score(name="qa_accuracy", value=1)` |

| MLflow 接入(做法同) | 示例参数 |
|---|---|
| 初始化 | `mlflow.set_tracking_uri("http://localhost:5000")` |
| 自动 | `mlflow.openai.autolog()` 后照常 `client.chat.completions.create(...)` |
| 记录分 | `mlflow.log_metric("p95_latency", x)` |

> 埋点脚本对接用 env var 开关:`OBSERVABILITY=langfuse|mlflow|off`,默认 off,不炸现场。

## 六、验收清单(执行 agent 自检)

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | Langfuse UI 登录 | `localhost:3000` 建号成功 |
| 2 | Langfuse 健康 | `/api/public/health` → ok |
| 3 | 一条真实 trace | 用 `eval_suite.py --live --only tpu`(需内核在线)后,Trace 页出现 1 条,含 token/延迟 |
| 4 | 工具调用可见 | tool-calc 用例的 function call 被高亮 |
| 5 | MLflow trace(若做) | OpenAI autolog 后 experiment 里 1 条 span,流式响应拼合 |
| 6 | 模型注册(若做) | bf16/Q4 两个版本可查 |
| 7 | 端口封闭 | `netstat` 仅 127.0.0.1:3000/5000,无对外 |
| 8 | 无明文密钥 | git/grep 无 token 原文 |

## 七、红线(两个方案都适用)

| 项 | 要求 |
|---|---|
| 密钥 | 全部走 `.env`,绝不硬编码/入库/打印 |
| 数据 | 只留个人电脑本机,不启用任何云同步 |
| 端口 | 只绑 `127.0.0.1`,别开防火墙公网 |
| 本机 | 完成前先在本机验证一轮,再接远端 Kaggle 端点 |
| Kaggle | 内核零改动;追账只在客户端,不占配额 |

---
**下一步(给执行 agent)**:按方案 A 部署 Langfuse → 跑验收 1-2-3-7 → 回填结果;MLflow 方案 B 可选。
**doc**:本次仅方案交付,不产生代码落盘;后续埋点脚本由下游按"五、埋点接入设计"实现,代码文件另起 task。