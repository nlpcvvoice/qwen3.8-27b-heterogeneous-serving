# Controller + Router (TPU/GPU scheduling) — 2026-09-08

> 文件因 MASTER 规则命名 `date+time+summary`。关注点：本地调度控制器 P1 + 混合路由 + 总开关 + 红线保护。

## 决策（用户拍板）

| 项 | 决定 |
|---|---|
| model id 方案 | **混合**：`qwen3.8-27b`(auto) + `qwen3.8-27b-tpu` + `qwen3.8-27b-gpu` |
| 冷启动顺序 | T4 先起（快）→ TPU 就绪验证后切主 → 后台关 T4 省 quota |
| TPU 到期 | 到期前 ~40min 预热 T4；T4 服务期间若“新 9h 立即可用”则立即起 TPU，验证后切回 |
| controller 位置 | 本机(GCP)测试；实际可放任意客户端 |
| 引擎缓存 | llama-server 二进制 → 私有 dataset（绕 25min 构建，下轮 v5 实现） |
| **红线** | **dev 期间绝不关 TPU**（有外部 client 在用）；TPU 中断类测试一律等用户确认 |

## 交付物

| 文件 | 说明 |
|---|---|
| `app/router.py` | OpenAI 兼容转发；`auto/tpu/gpu` 三 id；SSE 直通；auto 连接级失败单次换引擎重试；每 30s 健康探测；读取 `tmp/controller_state.json` 的 `primary` 偏好 |
| `app/controller.py` | 调度状态机(eco)：冷启动→gpu-first→TPU-ready 切主+停 GPU→到期预热→FAILOVER 回读；ntfy 事件轮询(双引擎 topic)；总开关 power；dry/live 模式；**TPU 保护锁**；**sim 仿真开关**`/admin/sim` |
| `app/power.py` | 一键 CLI：`power on/off`、`mode live/dry`、`start/stop gpu|tpu`、`sim tpu on/off` |
| kernel 双内核 | `start_control_listener()`：ntfy 收到 `CONTROL/stop` → 优雅退出（总开关实现路径，非停 TPU） |
| `launch.py serve` | 新增 `--no-watch`（controller 只 push 不阻塞） |

## 验证结果

| 测试 | 结果 |
|---|---|
| router 健康/模型列表 | tpu/gpu 均 true，3 个 id 正常 |
| tpu-pinned chat | `TPU PIN OK`（重写上游 model id 正确） |
| auto 优先级 | primary=控制器偏好；failover 测试用**死端点 config**验证（`router_test_fail.json`） |
| controller 冷启动 | `cold: gpu-first`，primary=gpu |
| 总开关 off | primary=None，引擎→stopping（dry 无真实动作） |
| sim tpu down(workaround) | `tpu-sim-down (failover workaround)`，primary=gpu，**真实 TPU 分毫未动** |
| GPU v4 经 router | `V4 OK`（端点 `quit-confidential-worlds…`，ctx 98304 np4 q8_0 KV） |

## 状态机（confirmed）

| 条件 | 动作 |
|---|---|
| power=off | 停双引擎（TPU 有保护锁拦截，除非显式放开） |
| 双 down | 先起 gpu，primary=gpu |
| tpu starting | primary 保持 gpu（有则） |
| tpu ready+healthy | primary=tpu；停 gpu（省 quota） |
| tpu 服务中临近 TPU_MAX−40min | 预热 gpu |
| primary 引擎不健康 | failover 到另一引擎（controller+router 双保险） |
| tpu down + gpu 服务 | 超冷却期(10min)自动重试起 tpu（live 下推内核） |

## 待办 / 下一步（均不碰 TPU）

| 项 | 状态 |
|---|---|
| v5 GPU 内核：vram 用 nvidia-smi 采集 + RAW_LOG flush + built 二进制拷贝 /kaggle/working | 待 push（会新开 GPU session，需用户 OK） |
| 引擎缓存上传私有 dataset（绕 25min 构建） | v5 后做 |
| controller 绑定真实双引擎跑 dry→live | 待用户择时 |
| README 更新（controller 结构 + 红线规则） | 待上一条确认后一并 |