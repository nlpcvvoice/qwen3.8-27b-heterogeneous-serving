# q4km-mtp-deploy 第三方安装包 · 完成

时间:2026-09-13 03:39 · 状态:全部构建+校验通过,未上线实测(等用户在你的账号跑一次冷启动)

## 目标
任意 Kaggle 账号一键部署 Qwen3.8-27B Q4_K_M + MTP(96K/q8_0/draft-mtp/2),零依赖参考账号私有资源。

## 交付物(tmp/q4km-mtp-deploy/,独立包)

| 文件 | 作用 | 相对参考包(q4km-mtp)改造 |
|---|---|---|
| README.md | 第三方用户手册 | 全新 |
| setup.sh | 建 Venv+kaggle | 文案中性化 |
| reference/README.md | 只读 token 说明 | 任意账号,去 touch 资源说明 |
| kaggle_login.py | 认证(内存) | 去 tentenshishi 兜底,新增 `username()` |
| kaggle-tpu-lab/kernel/serve_qwen38_gpu_mtp.py | 主内核 | ctx 98304+q8_0(已验证配置);`weights_dataset="_hf_direct"` 强制 HF;HF import 失败自动 pip |
| p3/kernel/serve_qwen38_gpu_mtp.py | 镜像副本 | 与主内核 md5 相同 |
| p3/kernel/ctrl_stop.py | 免费停机 | 不变 |
| app/push_gpu_serve_mtp.py | 拉起 | USER=`kl.username()`;权重走 HF;自检自己 `llama-server-qwen38-cache` 是否存在 |
| app/start_q4km_mtp.py | 一键启动+写 opencode.json | `SERVE=f"{kl.username()}/..."` |
| app/push_stopper_mtp.py | 停机脚本 | id=`kl.username()/qwen38-gpu-ctrl` |
| app/bootstrap_cache.py | 首次自编译后缓存引擎到自己私有 dataset | 新增 |

## 首次冷启动时序

| 阶段 | 事 | 耗时 |
|---|---|---|
| 1 | push 私有内核(权重 attach=无,引擎 cache=无) | — |
| 2 | 内核:HF 下载 16.5GB 官方 GGUF + git clone 编译 llama.cpp(CUDA/T4) | ~25-40min |
| 3 | serve 起来,自检 /models → `config written` | — |
| 4 | --stop 后跑 bootstrap_cache.py → 自己账号存 engine.tar.gz | ~5min |
| 5 | 再 start → 自动 attach 自己的 cache,cache-hot | ~5-8min |

> 权重每次走 HF(官方公开)或用户后续自行镜像;引擎通过 bootstrap 共用一处缓存,实现「编译一次,以后秒开」。

## 校验结果

| 检查 | 结果 |
|---|---|
| 全包 py_compile(./Venv/bin/python) | OK |
| 内核两副本 md5 | 一致(e6756619...) |
| 残留 tentenshishi | 仅 README 两处对比文字(资源零引用) |
| `__LAUNCHER_CONFIG__` 注入点 | 存在 |
| 三脚本动态 USER | push/start/stopper 均 `kl.username()` |
| token 红线 | 只读内存,无落库/打印;pyc 已重编译为最新 |

## 下一步(等用户)
在**你的一个第三方 Kaggle 账号**跑一遍 `bash setup.sh` → 放 token → `start`(冷启动实测,捕获 PHASE 日志) → 验证 → `stop` → `bootstrap` → 再 `start`(cache-hot)。通过后可选:weight 镜像自己账号,finalize 文档。