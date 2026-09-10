# P3 engine-cache

## 流程图

```
┌─────────────────────┐   上传二进制    ┌─────────────────────────────────┐   push 内核时自动挂载   ┌──────────────────────────────┐
│  有 GPU 的机器        │ ──────────────→ │  Private Dataset                │ ←────────────────────── │  GPU 内核 (serve_qwen38_gpu) │
│  构建 llama-server    │  upload_cache   │  llama-server-qwen38-cache      │   dataset_sources        │  cache-hit → 跳过 25min 编译  │
└─────────────────────┘                  └─────────────────────────────────┘                          └──────────────────────────────┘
         │                                         ↑                                               │
         │ kaggle kernels output                  │ download                                     等同
         └────→ 从 kernel output 取得二进制        │                                                ↓
                                                   │                                    /kaggle/input/.../llama-server
                                                   │                                     (skip 25-min build)
                                          ┌─────────────────────┐
                                          │  任意机器 / 任意 agent │
                                          │  start.sh fetch      │
                                          └─────────────────────┘
```

## 一键命令 (start.sh)

| 命令                    | 说明                                                    | 配额消耗 |
|-------------------------|---------------------------------------------------------|----------|
| `./start.sh test`      | 完整本地验证 (内核决策模拟 + 静态断言)                    | 0        |
| `./start.sh static`    | 仅静态断言 (内核/脚本 hook 完整性)                        | 0        |
| `./start.sh status`    | 查看缓存 dataset 状态 (私有/文件数/大小)                  | 0 (只读) |
| `./start.sh upload BIN` | 上传 llama-server 二进制 → 私有 cache dataset            | dataset quota |
| `./start.sh fetch`     | 下载缓存的 llama-server 并校验 SHA                       | 0 (只读) |
| `./start.sh live-test` | 真实 Kaggle 端到端验证 (需要 API Token)                   | dataset quota |

## 本地验证输出 (test)

```
[P3 validate] engine-cache full test
  [local] cache dataset mounted      -> engine=llama-server (cache-hit)  OK
  [local] no cache dataset           -> source-build                    OK
  [local] tiny/garbage binary        -> rejected -> source-build        OK
  [local] force no-cache flag        -> source-build                   OK
  [local] datasets/*/<slug> layout   -> cache-hit                      OK
  [static] kernel  : skip-compile hook + else-build + cache-hit publish   OK
  [static] push    : existence probe + dataset_sources injection        OK
  [static] kernel  : py_compile clean                                     OK
[P3 validate] ALL PASS
```

## 生产流程

```
步骤           命令/动作                                      产出
────────────────────────────────────────────────────────────────────────────────
1 构建         GPU 内核正常运行 (无 cache 命中时)              llama-server 二进制
2 拉取         kaggle kernels output tentenshishi/qwen38-gpu-serve -p tmp/
3 上传         ./start.sh upload tmp/llama-server               私有 cache dataset
4 后续任意会话  push_gpu_serve.py 自动探测 cache dataset          内核跳过编译,启动时间 <2min
```

## 目录结构

```
p3/
├── start.sh                一键入口
├── requirements.txt        依赖
├── validate_cache.py       完整验证 (本地 + 可选 live)
├── upload_cache.py         上传二进制
├── fetch_cache.py          下载并校验
├── status.py               查看 dataset 状态
├── common.py               共享常量 + 工具函数
├── kaggle_login.py         认证 (in-memory, 无明文)
├── kernel/                 内核副本 (含 cache 命中逻辑)
│   └── serve_qwen38_gpu.py
└── reference/
    └── API-Token           认证 token (不提交 git)
```

## copy 到新机器

1. 整个 p3/ 目录 + reference/API-Token → 新机器
2. `pip install -r requirements.txt` (或直接用系统 Python)
3. `./start.sh test` → 验证可用
4. `./start.sh fetch` → 拉取缓存的二进制

## 红线

- reference/API-Token **只读**，内存使用，不打印原文
- 内核 cache 命中仅复制二进制，**不写 /kaggle/working**
- 任何 push 到 Kaggle **必须 is_private=true**
- upload_cache 只更新**私有** dataset，不做其他操作
