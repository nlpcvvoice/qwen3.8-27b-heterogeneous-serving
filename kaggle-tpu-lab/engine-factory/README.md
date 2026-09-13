# engine-factory — 离线预编译 llama-server(容器化,不需要 GPU)

把 Kaggle 内核里最耗时的一步(在考场内现攒 llama-server,~25 min)搬到**工厂**:
用 Docker + CUDA devel 镜像,在任意有 CPU 的机器上编译出与内核源码构建**完全相同**的
`engine.tar.gz`(`bin/` + `lib/` 布局),再交给现有的 `bootstrap_cache.py` 喂进内核。

- 编译只用 CPU(nvcc 生成 sm_75 目标码,不需要显卡);
- 同一镜像 + 零参数改动 → 每台"车"可复现(记录 `engine.sha`);
- 内核启动即 `engine-cache-hit`,不再发生冷构建。

## 用法

```bash
# 默认:master(与内核源码构建同源),输出到 ./out
./build_engine.sh

# 指定输出目录 / 固定版本
./build_engine.sh /tmp/engout --commit <sha-or-master>

# 产物(./out)
#   engine.tar.gz   内核要求的 tar 布局(bin/ + lib/),由 package.sh 用 readelf 校验动态库
#   engine.sha      本次编译所锁定的 llama.cpp commit
```

> 首次构建镜像会拉取 `nvidia/cuda:12.4.0-devel-ubuntu22.04`(~5 GB)+ 卷代码 + 编译,约 20-40 min;
> 再次构建走 Docker 缓存,秒级出包。

## 与内核的对应关系

| 项 | 内核原版(serve_qwen38_gpu_mtp.py §2) | 本工厂 |
|---|---|---|
| 源码 | git clone ggml-org/llama.cpp master | 同一仓库(master,可锁定 commit) |
| 标志位 | GGML_CUDA=ON / FORCE_DMMV / CCACHE=OFF / Release / ARCH sm_75 / NATIVE=OFF | **一字不差** |
| 产物 | WORK/engine.tar.gz(bin+lib) / 单文件 llama-server | /out/engine.tar.gz + engine.sha |
| 消耗的资源 | Kaggle GPU 会话(~25 min,GPU 空转) | 本地 CPU + Docker(零 Kaggle 资源) |

## 怎么喂给内核(复用现有管线)

```
工厂 → out/engine.tar.gz → 上传到私有 dataset(如 llama-server-qwen38-cache)
     → push_gpu_serve_mtp.py 自检该 dataset → 自动 attach
     → 内核 find_input → unpack_engine() → 引擎即开即用
```

## 校验清单

| 检查 | 方法 |
|---|---|
| tar 布局正确 | `tar tzf out/engine.tar.gz` 含 `bin/llama-server` 与 `lib/` |
| 目标架构 sm_75 | `cuobjdump`/`nvdisasm` 或构建期日志确认 `arch=compute_75` |
| 动态库依赖正常 | package.sh 内 readelf 检查 NEEDED(libcuda/libcudart/libcublas/libllama) |
| 版本可复现 | `out/engine.sha` 记录 commit;换版本再跑一次 |
| 真实启动 | 待 GPU 放行后跑一次内核,观察 `engine-cache-hit`(运行时核验项) |

## 注意

- 运行时 CUDA 驱动由 Kaggle 主机提供,本机不装/不下载即可;
  工厂镜像与 Kaggle 运行时同为 Ubuntu 22.04 + CUDA 12.x,ABI 对齐。
- 产物只放 `out/`,不入 git;本目录仅提交构建脚本与文档。