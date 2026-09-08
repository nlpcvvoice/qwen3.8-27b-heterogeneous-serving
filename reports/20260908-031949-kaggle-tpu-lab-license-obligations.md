# License 责任与义务分析报告 — ARahim3/kaggle-tpu-lab

> 生成:2026-09-08 03:19 UTC | 依据:GitHub LICENSE + README + Kaggle数据集元数据(API) | 状态:仅报告,未执行任何操作

## 1. 结论速览

| 组件 | 来源 | 许可证 | 商用 | 强制开源修改 | 关键义务 |
|---|---|---|---|---|---|
| 仓库代码(launch.py/kernel/notebook/patches/tools) | ARahim3/kaggle-tpu-lab | MIT | 允许 | 否 | 保留版权+许可声明 |
| 模型权重 Qwen3.8-27B | Qwen/Qwen3.8-27B(HF) | Apache-2.0 | 允许 | 否 | 附许可副本/修改声明/专利条款 |
| 数据集 rahim3/qwen3-8-27b-bf16 | 权重镜像(55.6GB safetensors) | Kaggle标注:Other(description为空) | ⚠️ | — | 标注与描述不符,需确认镜像内LICENSE |
| 数据集 rahim3/qwen38-tpu-env-v5e8 | 环境包(xla_cache+cloudflared+manifest) | Kaggle标注:Unknown | ⚠️ | — | License缺失,再分发风险 |
| 依赖 vLLM / tpu-inference | vllm-project | Apache-2.0 | 允许 | 否 | 随包附带许可(委托上游) |
| cloudflared 二进制 | Cloudflare | Apache-2.0 | 允许 | 否 | 随包附带许可 |
| 托管于 Kaggle notebook | — | Kaggle平台ToS | — | — | 遵守Kaggle条款,与MIT/Apache无关 |

## 2. MIT — 仓库代码的责任清单

| # | 责任 | 触发场景 | 履约方式 |
|---|---|---|---|
| M1 | 在所有副本/实质性部分保留版权与许可声明 | 拷贝、分发、嵌入launch.py/serve_qwen38.py/ipynb/字dependencies | 保留 LICENSE 文件原文,含 "Copyright (c) 2026 Abdur Rahim" |
| M2 | 保留 "AS IS" 无担保/免责声明 | 向第三方分发 | 不得删除 MIT 免责条款 |
| M3 | 无强制开源 | 修改后可闭源/商用/售卖 | 无需回馈源码 |
| M4 | 商标不授权 | 宣传/命名 | 不得暗示作者背书 |
| M5 | 无专利授权 | 涉及专利维权 | MIT无专利条款,需自行评估 |

## 3. Apache-2.0 — 模型权重/上游依赖的责任清单

| # | 责任 | 触发场景 | 履约方式 |
|---|---|---|---|
| A1 | 再分发时附许可全文副本 | 权重再上传/分发/封装 | 附带 Apache-2.0 文本 |
| A2 | 保留 NOTICE/版权/专利/attribution 声明 | 打包发布 | 不删除 Qwen NOTICE 等声明 |
| A3 | 修改须声明变更 | 修改权重/派生代码 | "Modified"标注+变更摘要 |
| A4 | 专利授权与终止条款 | 专利侵权诉讼 | 每个贡献者授予专利;起诉即终止 |
| A5 | 商标/名称限制 | 宣传/命名 | 不得用 Qwen/Abab 名义误导 |
| A6 | 无强制开源/无 Copyleft | 商业API封闭 | 无需开源调用方代码 |

## 4. 数据集专项(风险最高)

| 数据集 | 标注 | 实际内容 | 风险 | 建议 |
|---|---|---|---|---|
| rahim3/qwen3-8-27b-bf16 | Other(specified in description),但description为空 | Qwen3.8-27B 权重镜像(Apache-2.0 上游) | 标注缺失→法律指向不明确 | 下载镜像内 LICENSE/config 核对;若为原样镜像,按 Apache-2.0 履约 |
| rahim3/qwen38-tpu-env-v5e8 | Unknown | XLA编译缓存+cloudflared二进制+manifest | 无License→再分发无法律依据 | 使用前向作者索取授权或自建等价缓存 |

## 5. 交叉组件注意点

| 场景 | 注意 |
|---|---|
| patches/mtp-rollback-v0280.diff | README声称repo全MIT;但diff移植自上游 tpu-inference PR#3178(Apache-2.0),理论上需同时满足两许可 |
| env数据集内含cloudflared二进制 | 独立Apache-2.0作品,许可随二进制附带 |
| 生成的API endpoint + key | 安全性自担,不属License范畴 |
| Kaggle免费TPU配额 | 一次一会话/9h上限/~20h每州,遵守Kaggle平台条款 |

## 6. 结论

| 结论 | 内容 |
|---|---|
| 总体许可 | 宽松双许可(MIT + Apache-2.0),可商用、可闭源、可修改 |
| 核心履约 | ①保留MIT版权声明;②保留Apache-2.0许可副本+NOTICE;③修改标注 |
| 最大风险 | 两个Kaggle数据集的License标注不清晰(Other-空描述 / Unknown) |
| 低风险使用方式 | 仅克隆仓库跑代码(本地)+ 引用HF官方 Qwen 权重,不直接依赖/再分发两个数据集 |