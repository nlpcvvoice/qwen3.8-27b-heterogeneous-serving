# 故障切换 TPU→GPU · 是否需要启动 TPU(只分析,不碰 TPU)

时间:2026-09-13 04:56 · 原则:本报告仅文字推演,未启动/未触碰任何 TPU 会话

## 结论(一句话)
**"自动 failover"机制已验证到 Tier0/1,不需要 TPU;但"真实异构切换演练(Tier2)"从定义上就要求启动一次 TPU——由你出指令时才做。**

## Failover 能力现状(本项目已有,不是规划)

| 组件 | 已落地 |
|---|---|
| router 转发 | `app/router.py`:auto/tpu/gpu 三 id,SSE 直通,connect 级失败单次换引擎重试 |
| 健康探测 | 每 30s HTTP 探活,prefer 由 controller 写 `tmp/controller_state.json` |
| 验证记录 | reports/20260908-190510:死端点 config(`router_test_fail.json`)模拟 TPU 挂 → GPU 顶替 ✓;GPU 经 router 实测 `V4 OK` |

## 需要 TPU 的分级

| Tier | 内容 | 需 TPU? | 时长/成本 | 能证明啥 |
|---|---|---|---|---|
| 0(已做) | 死端点模拟 TPU 故障 → 切到 GPU | 不 | 0 | 切换逻辑、健康探测、重试行为 |
| 1 | GPU 为主 + 假 TPU 端点(dark)持续探活 | 不 | 0 | 常驻探活/状态机的真实运行形态 |
| 2 | 真 TPU v5e-8 服务在线 → 使其中断 → GPU 顶替 → 恢复 TB 回切 | **要** | 1 次 TPU 会话 ~1-2h(20h/周配额内) | 跨加速器真失真的端到端体验(TTF/丢流/TTFT) |

**为什么 Tier2 逃不开 TPU**:failover 的"故障源"必须是那个真实在场的主引擎,才能产生真实的连接中断→切换→回切;没有主引擎在场,演练就只是"影子测试"。这与 token/逻辑无关,是属性决定的。

## 建议路径(不碰 TPU 的当前最优)
- 现在:把 Failover 写成"梯级已验"文案 → README(bench 附 Tier0/1 证据)
- 面试叙事:主打 **"auto-failover(router 30s 探活 + 双保险)机制在真 GPU + 模拟 TPU 故障下通过"**
- Tier2 留给一次**你授权的** TPU 会话(约 1-2h),并提前备好 keepalive/自停

## 红线
- 未获明确指令:不 push/probe/启动任何 TPU 内核,不消耗 TPU 配额