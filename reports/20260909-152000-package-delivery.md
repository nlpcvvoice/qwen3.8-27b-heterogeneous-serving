# Package 交付报告 (可携带启动包)

| 项 | 值 |
|---|---|
| 位置 | `package/` (项目根下) |
| 目标 | 整体拷到另一台机器, agent 凭 README + 代码即可启动 TPU / GPU |
| 依赖 | `requirements.txt` (kaggle 2.2.4 / requests / fastapi / uvicorn), 不携带 Venv |
| Token | `reference/API-Token` 已原样放入, 内存读取不打印 |

## 目录结构

```
package/
├── README.md                     # 主文档: 安装/启动/参数/错误/红线
├── SERVICES_CURRENT.md           # 部署后登记真实 endpoint + key 状态
├── requirements.txt
├── kaggle_login.py               # 认证 (token 内存读取, 掩码校验)
├── reference/API-Token           # Kaggle token (只读)
├── kaggle-tpu-lab/               # TPU: launch.py + kernel/serve_qwen38.py + MTP patch
└── app/                          # run_launch.py(wrap) + push_gpu_serve.py + watch_gpu_serve.py
```

## 已测试项

| 测试 | 结果 | 说明 |
|---|---|---|
| 全部脚本 py_compile | ✓ 8/8 | package 内独立编译通过 |
| 干净 venv 装 requirements | ✓ | pip install 无报错 |
| testenv 内认证链 | ✓ | `kl.verify()` → ok=True, token 掩码 |
| launch.py --help | ✓ | serve/status/stop/build-env 子命令可见 |
| GPU CFG 注入逻辑 | ✓ | `__LAUNCHER_CONFIG__` 替换 n=1, 值正确 |
| TPU CFG 注入逻辑 | ✓ | max_model_len=262144 注入正确 |
| MTP patch 内嵌 | ✓ | 内核自含 base64 diff, 运行时不需要外部文件 |
| run_launch.py import 安全 | ✓ | 加了 `if __name__ == "__main__"` 防误触发 |
| 状态文件路径解析 | ✓ | launch/push/watch 均指向 `package/tmp/` |
| KAGGLE_CONFIG_DIR | ✓ | `package/tmp/kaggle_cfg` 隔离, 不污染 ~/.kaggle |

## 修正（相对仓库原版）

| 文件 | 修改 |
|---|---|
| `app/run_launch.py` | 模块级执行 → `main()` + guard, 防 import 即 push |
| `app/push_gpu_serve.py` | 同上 |
| 目录布局 | `kernel/serve_qwen38.py` 归位 `kernel/`; patch/tools 归位 |

## 未做(避免副作用)

| 项 | 原因 |
|---|---|
| 不真 push 内核 | 测试会占用 Kaggle 槽位 (GPU 2 上限), 真实部署时目标机器执行 |
| 不 exec 内核源码 | 曾 exec 触发 16GB 下载+编译, 已清理 (教训记录) |

## 新增: 服务端点自动登记 (v2)

`tmp/current_services.json` 在引擎 READY 时由代码**自动写入** `{endpoint, api_key, model, kernel, topic, ready_at}`, client 机直接读即可连接。

| 入口 | 触发 | 引擎 |
|---|---|---|
| launch.py (TPU) | ntfy `ready` | tpu |
| watch_gpu_serve.py | ntfy `serving`/`benchmark` | gpu |
| controller (repo) | harvest READY (ready/serving/benchmark) | tpu/gpu |
| register_service.py take | 手动 / --no-watch 场景 | 指定 |

已验证(真实数据 2026-09-09 15:35):

| 验证项 | 结果 |
|---|---|
| `register_service.py take gpu` (v7 真实 topic) | ✓ 写入 repo tmp, 权限 600 |
| controller 重启后事件回放自动登记 | ✓ `[services] gpu registered` |
| package 8 脚本编译 | ✓ |
| register_service.py repo/package 同步 | ✓ 相同 |

> 已登记 GPU v7: `https://bizarre-commonwealth-gen-scoring.trycloudflare.com/v1` (key 掩码 sk-d40...214e)

## 部署步骤 (目标机器)

1. `python3 -m venv Venv && ./Venv/bin/pip install -r requirements.txt`
2. `./Venv/bin/python -c "import kaggle_login as kl; kl.login(); print(kl.verify())"` → ok=True
3. TPU: `./Venv/bin/python app/run_launch.py serve` (或 `--no-watch`)
4. GPU: `./Venv/bin/python app/push_gpu_serve.py` + `watch_gpu_serve.py`
5. READY 后更新 `SERVICES_CURRENT.md`

## 当前 TPU 部署状态 (本机)

| 项 | 值 |
|---|---|
| 内核 | qwen38-tpu-serve (keepalive 540 = 9h) |
| 状态 | QUEUED (等待 Kaggle 槽位 ~65min+) |
| 无事件 | 尚未分配硬件 |

> TPU READY 后: 端点写入 SERVICES_CURRENT.md; 全程不动 TPU (外部 client 使用)。