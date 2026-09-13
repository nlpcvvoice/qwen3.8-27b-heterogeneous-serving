# Router/LoRA 归属考证 · 是否串了 GH 其它项目

时间:2026-09-13 04:56 · ⚠️ 本报告结论部分已被 20260913_0532_router_two_projects_disambiguation.md 更正:
"Router(=failover 网关)"属本项目;"LoRA + Phase1/2/CI 话术"与 `router-lora-classifier-debug` 内核实属
nlpcvvoice/hybrid-llm-router-cost-optimizer(另一项目)。以下表格保留供追溯。

## 结论
**Router/LoRA 是本项目自有的规划与落地物,不是从 `ARahim3/kaggle-tpu-lab` 或其它 GH 项目搬来的。** 两家都叫 "kaggle-tpu-lab" 且都做 Kaggle 免费 TPU 跑 Qwen3.8-27B,但内容不重叠;Router/LoRA 仅出现在本项目。

## 证据(出自我们自己的文件,非链接 GH)

| 证据 | 位置 | 类型 |
|---|---|---|
| Router 实体代码 | `app/router.py`(OpenAI 兼容转发、auto/tpu/gpu 三 id、SSE 直通、30s 健康探测、互操作重试) | 本仓库文件 |
| Controller 实体代码 | `app/controller.py`(primary/备选偏好,写 `tmp/controller_state.json`) | 本仓库文件 |
| 专项规划报告 | `reports/20260908-190510-controller-router-p1.md`(Auto-failover 方案 + failover 用死端点验证 + GPU 经 router 实测 OK) | 本仓库报告 |
| MASTER 任务定义 | MASTER.md「Track 2 · 简历完成态」:`router phase 表` = "Phase1 数据工程完成 / Phase2 LoRA 训练就绪 / CI 全绿";Skills 注册表:故障切换/健康检查 = router(P1) | 本仓库文档 |
| 本项目账号实体内核 | `tentenshishi/router-lora-classifier-debug`(你贴的 job 列表,u状态 ERROR) | Kaggle 账号 |
| 相关核心 | `app/power.py`、`reports/20260910-191700-prod-mapping-1to1-kids.md`(生产映射) | 本仓库 |

## 对照 ARahim3/kaggle-tpu-lab(导致疑似混淆的对象)

| 项 | ARahim3/kaggle-tpu-lab | 本项目 |
|---|---|---|
| 仓库结构 | `kernel/ notebook/ patches/ tools/ launch.py`(仅 serve/stop/status) | 多 module:`app/(router/controller/power/…)` + `p3/ kaggle-tpu-lab/ loadtest/ test/` |
| Router/LoRA/RAG demo | **无**任何 router/controller/lora 文件或描述 | 有实体+报告+内核 |
| 混淆点 | 两边目录都叫 `kaggle-tpu-lab`、同模型同免费 TPU 主题 | 仅此 |
| 我方借用面 | TPU 加速思想参考(uv+XLA cache 方案,当轮已写入报告 B) | 与 Router/LoRA 无关 |

## 结论表
- Router/LoRA:属于本项目(有代码/报告/账号内核/MASTER 定义四层证据)
- 排查对象(ARahim3):无该功能 → 无串项目
- 唯一风险项:`router-lora-classifier-debug` 内核当前 `ERROR`,属于**本项目的未完成工作**,非他人产物

## 建议下一步(待你指令)
- 修复 `router-lora-classifier-debug` ERROR;或按 P1 收尾 router 灰度方案。当前已具备写作"归属无忧"证据链,可直接写入 README「项目血缘」小节。