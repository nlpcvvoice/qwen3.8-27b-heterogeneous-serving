# Qwen3.8-27B Kaggle TPU v5e-8 私有数据集 Serve 全流程验证报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-08 (UTC) |
| 作者目录 | /home/jupyter/opencode/kaggleTPUserveQwen3p8full27bSaveModelFile |
| 上游项目 | github.com/ARahim3/kaggle-tpu-lab (MIT) |
| 模型 | Qwen3.8-27B (bf16, ~55.6 GB safetensors) |
| 硬件 | Kaggle 免费 TPU v5e-8 (8 芯片 × 16GB) |
| 状态 | ✅ 完整成功并验证 |

## 产出物

| 名称 | 标识 | 说明 |
|---|---|---|
| 私有权重数据集 | tentenshishi/qwen3-8-27b-bf16-private | 28 文件, 55,586,027,785 B, isPrivate ✓, ready |
| serve 内核 | tentenshishi/qwen38-tpu-serve | machineShape=TpuV5E8, v3 |
| 探针内核 | tentenshishi/qwen38-tpu-probe | 验证 TPU 分配(最终通过) |
| OpenAI 端点 | https://autumn-gathering-neck-particular.trycloudflare.com/v1 | cloudflared 隧道(临时) |

## 全流程时间线

| 时刻(UTC) | 事件 |
|---|---|
| 04:27 / 04:45 | serve v1/v2 失败: TPU 未分配,静默 CPU 兜底 (found 1 device, expected 8) |
| 06:24→07:31 | 探针 v2 排队 ~2.5h → 真实 TPU: TPU_ACCELERATOR_TYPE=v5litepod-8, /dev/vfio ✓, jax.device_count=8 |
| 07:34→09:02 | serve v3 排队 ~1.5h → RUNNING |
| 09:11 | `Weights found mounted (no download needed)` ✓ |
| 09:32 | READY banner + 基准 112.1 tok/s; image-test → "red" ✓ |
| 10:33 | 持续 serving (60 min up, keepalive 180) |
| ~11:33 | 会话被终止 (serving 120 min 处, 早于 keepalive 180min); 端点随后失效 |

## 测试结果

| 测试 | 结果 | 详情 |
|---|---|---|
| GET /v1/models | 200 (0.3s) | id=`qwen3.8-27b`; root=/kaggle/input/datasets/tentenshishi/qwen3-8-27b-bf16-private |
| POST /chat/completions | 200 (2.9s) | 中文答案正确; 64 tok; 含 29 思考 token |
| 流式 stream | 200 (0.5s, 12 chunks) | "2+2=?" → "4" |
| 内核自检 | ✓ | 基准 112.1 tok/s; 图像测试 answer=red |

## 零网络下载验证 (核心目标)

| 证据 | 位置 |
|---|---|
| `PHASE weights-mounted {"/kaggle/input/datasets/tentenshishi/qwen3-8-27b-bf16-private"}` | 内核日志 |
| `Weights found mounted (no download needed)` | serve.log |
| `/models` 返回 root=私有数据集路径 | 本次实测 |
| 数据集挂载清单 | metadata.datasetDataSources=[私有,+rahim3/qwen38-tpu-env-v5e8] |

→ 下次启动: 权重直接挂载, 仅需 ~20 min TPU 编译(env 缓存), 无网络下载。

## 排障经验 (社区共识佐证)

| 现象 | 结论 |
|---|---|
| 一直 QUEUED | TPU 高峰正常, 数小时级别(Kaggle 官方: demand/supply) |
| 秒开但 CPU-only | 静默降级 CPU 的已知平台问题; 用探针确认 jax.device_count=8 后再 serve |
| 手机验证 | 已通过; 无其他门槛 (enable_tpu=true + TpuV5E8 即正确用法) |

## 常用操作

| 操作 | 命令 |
|---|---|
| 重新拉起 serve | ./Venv/bin/python app/run_launch.py serve --weights-dataset tentenshishi/qwen3-8-27b-bf16-private --slug qwen38-tpu-serve --keepalive-min 180 |
| 查看状态/端点 | ./Venv/bin/python launch.py status |
| 停止 | ./Venv/bin/python launch.py stop |
| 模型名 | qwen3.8-27b (注意是点, 非横杠) |

## 局限与提醒

| 项 | 说明 |
|---|---|
| TPU 额度 | ~20 h/周, 单会话 ≤9 h, keepalive=180 min 自动停止; 本次于 serving 120 min 处被外部终止(疑额度触发), 精确余额仅浏览器可见 |
| 隧道 URL | cloudflared 随机域名, 内核结束即失效 |
| 数据集大小 | 55.6 GB → 私有, 挂载零复制(内核 /kaggle/working 仅 ~21 GB 的替代方案已验证) |
| Token 保密 | reference/API-Token 全程仅内存读取, 未打印未上传 ✓ |