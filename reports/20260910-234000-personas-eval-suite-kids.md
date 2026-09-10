# 台词本📖 + 考试卷📝 — 交付报告(kids 图解)

| 报告日期 | 2026-09-10 23:40 | 状态 | ✅ 已落地(离线部分) / 🟡 在线部分待下次营业 |
|---|---|---|---|
| 关联想法 | 三件新鲜事 #2 台词本 / #3 考试卷 | 配额消耗 | 🟢 0(纯离线生成 + dry 自测) |

## 一、台词本 📖(personas)

### 我们做了一个"假顾客台词本"

```
app/personas.py  (零依赖,纯离线)
   │
   ├── 10 类"顾客"人设
   │     ├─ 化学老师 🧪 开发者 💻 小学生 🧒 老奶奶 👵
   │     ├─ 销售经理 📊 数据分析师 📈 医学助理 🧬
   │     ├─ 创始人 🚀 译者 🌍 ML工程师 ⚙️
   │     └─ 每人设 2~3 种"问法"(解释/调试/总结/翻译…)
   │
   └── 生成 200 条真实感提问
         → tmp/personas/corpus.jsonl
```

| 数据 | 数值 |
|---|---|
| 人设数 | 10 |
| 生成条数 | 200(每个 20 条,均匀分布) |
| 难度档 | d0(96 tok) / d1(192 tok) / d2(384 tok) |
| 运行方式 | 离线,不占任何 GPU/TPU 配额 |

### 台词本怎么用?

```
  台词本 corpus.jsonl
       │
       ├──▶ Locust 压测  (app/locustfile.py)
       │       启动时设 PERSONAS=tmp/personas/corpus.jsonl
       │       → 假顾客自动变得"像真人"
       │
       └──▶ 以后考试卷也能抽样提问
```

| 用法 | 命令 |
|---|---|
| 看人设清单 | `./Venv/bin/python app/personas.py --list` |
| 生成 200 条 | `./Venv/bin/python app/personas.py` |
| 自定义 | `--count 500 --seed 7 --out tmp/personas/x.jsonl` |
| Locust 用台词本 | `PERSONAS=tmp/personas/corpus.jsonl` |

> 之前:压测 15 条固定问句 → 现在:200 条真实人设提问,分布可复现(seed)。

## 二、考试卷 📝(eval_suite)

### 我们做了一个"自动判卷机器人"

```
app/eval_suite.py  (默认 --dry 离线自测,不碰 API)
   │
   ├── 7 道考题,10 项断言(assert)
   │     ① 算术 123×7  → 必须答「861」
   │     ② JSON 返回    → 必须真是 JSON,count 字段 = 3
   │     ③ JSON 只有 ok → 必须含 ok 键
   │     ④ 工具 calc     → 必须调用 calc,参数有 expr ✅(工具调用解析)
   │     ⑤ 工具 list    → 必须调用 list_files,参数有 path ✅
   │     ⑥ 监控指标     → 必须命中关键词(延迟/RPS/错误率…)
   │     ⑦ KV 显存估算  → 必须答出 9 千~1 万附近(正则)
   │
   └── 自动给 TPU/GPU 两个店员一起判卷
         → tmp/eval/result-<时间戳>.json  (成绩单存档)
```

### 工具调用解析(qwen3_coder parser 检查)怎么判?

```
  店员答:「我要用计算器:expr=2+3」
     ↓(在线被测服务自带 qwen3_coder 工具解析器)
  response.tool_calls[0].function.name = "calc"    → 判 ✅
  response.tool_calls[0].arguments.expr 存在        → 判 ✅
```

| 断言类型 | 个数 | 说明 |
|---|---|---|
| contains_any | 2 | 答案必须含某词 |
| regex | 1 | 符合数字模式 |
| json_valid / json_field | 1+1 | 真 JSON + 字段值比对 |
| tool_call | 2 | 工具名正确 |
| tool_arg | 2 | 参数键齐全 |

### 怎么跑?

| 模式 | 命令 | 说明 |
|---|---|---|
| 离线自测(默认) | `app/eval_suite.py --dry` | 5 场景全过,验证卷面本身没毛病 |
| 在线判卷 | `--live --only tpu` / `--only gpu` / `--only all` | 需要店员在营业 |
| 局部抽考 | `--live --limit 4` | 只考前 4 道 |

## 三、验证结果(本次实跑)

| 项 | 结果 |
|---|---|
| personas 生成 | ✅ 200/200,10 人设各 20 |
| locust 加载台词本 | ✅ 200 条池子生效 |
| eval dry 自测 | ✅ 5/5 断言场景符合预期 |
| eval live 在线 | 🟡 两个小店都打烊了(tunnel DNS 失效),下次营业再判 |
| 代码编译 | ✅ 3 个脚本 py_compile 全过 |

## 四、验收标准(面试叙事对齐)

| 面试问题 | 本次落地 | 下一缺口 |
|---|---|---|
| Evaluation methodology | 探针 + 7 题断言卷 + 工具调用解析 + 台词本驱动压测 | 在线判卷成绩单、LLM-as-judge 评委 |
| Load test 真实分布 | 10 人设/200 prompt 真实提问 | 20~50 VU 饱和度测试(等下次配额) |

### 一句话总结 🗣️

> **台词本和考试卷都在家里准备好了,卷子已自检无误;等店员下次开工(下一次 TPU/GPU 配额),当场就能开考。**

---
**红线**:token 不打印/不上传 | 本报告为 2026-09-10 快照 | tmp/ 内容不入库