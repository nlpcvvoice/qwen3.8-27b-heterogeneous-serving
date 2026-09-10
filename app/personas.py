#!/usr/bin/env python3
"""台词本 — offline persona prompt corpus generator (zero dependency, no quota).

Generates a realistic distribution of user prompts across personas + intents,
meant to feed Locust load tests (PERSONAS_CORPUS) or the eval suite.

Output: tmp/personas/corpus.jsonl
  rows: {"persona","role","intent","prompt","domain","difficulty","max_tokens_hint"}

Usage:
  Vev/bin/python app/personas.py                     # default: 200 prompts
  Vev/bin/python app/personas.py --count 500 --seed 7 --out tmp/personas/corpus.jsonl
  ./Venv/bin/python app/personas.py --list           # show personas only
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tmp" / "personas" / "corpus.jsonl"

PERSONAS = [
    {
        "persona": "chemistry_teacher",
        "role": "高中化学老师正在备课",
        "domain": "chemistry",
        "difficulty": 1,
        "intents": {
            "explain": [
                "帮我解释一下什么是化学平衡,举例说明",
                "用最直白的话讲清楚氧化还原反应的原理",
                "怎么给学生讲'电子转移'这个概念才不晕?",
                "设计一个3分钟的课堂演示来解释催化剂的作用",
            ],
            "prepare": [
                "帮我出一套高一化学期中模拟题,10道选择题",
                "写一个酸碱中和反应的教学板书提纲",
                "整理本周要讲的知识点成一份备课清单",
            ],
        },
    },
    {
        "persona": "developer",
        "role": "后端开发在排查问题",
        "domain": "software",
        "difficulty": 2,
        "intents": {
            "debug": [
                "这段 Python 代码哪里错了?\n{code}",
                "为什么我的 Redis 连接池会耗尽?给排查思路",
                "写一个快速定位内存泄漏的 Python 脚本",
            ],
            "design": [
                "设计一个高并发下的限流方案,对比令牌桶和漏桶",
                "给一个 1000 QPS 的 API 网关画架构选型建议",
                "Postgres 索引失效的常见原因有哪些?如何预防",
            ],
        },
    },
    {
        "persona": "student",
        "role": "六年级小学生问自然课问题",
        "domain": "general-knowledge",
        "difficulty": 0,
        "intents": {
            "why": [
                "鲸鱼为什么不是鱼?",
                "为什么天空是蓝色的?讲简单一点",
                "为什么鸡蛋煮熟会变硬?",
                "为什么会有白天和黑夜?",
            ],
            "fun": [
                "讲一个关于蜜蜂的冷知识,要有趣",
                "给我出一个3以内的数学口算题",
            ],
        },
    },
    {
        "persona": "elderly",
        "role": "初学者老奶奶学习使用手机",
        "domain": "life-help",
        "difficulty": 0,
        "intents": {
            "howto": [
                "怎么在手机上用微信交电费?一步一步讲",
                "手机声音太小了,怎么调大?",
                "怎么把照片发给女儿?",
            ],
            "health": [
                "血压偏高平时要注意什么?说简单点",
                "晚上睡不好,有什么生活小习惯能改善?",
            ],
        },
    },
    {
        "persona": "sales_manager",
        "role": "销售经理准备季度汇报",
        "domain": "business",
        "difficulty": 1,
        "intents": {
            "summary": [
                "把下面这段拜访记录总结成3条要点:\n{content}",
                "本季度销量下滑20%,帮我分析可能的原因",
                "给老板写一段一页纸的季度summary",
            ],
            "email": [
                "帮我写一封给客户的道歉信,语气诚恳不卑不亢",
                "写一封跟供应商砍价的邮件,要礼貌又有理有据",
            ],
            "analysis": [
                "给一个销售漏斗报表设计3个最该看的指标",
            ],
        },
    },
    {
        "persona": "data_analyst",
        "role": "数据分析师处理一张业务表",
        "domain": "data",
        "difficulty": 2,
        "intents": {
            "sql": [
                "写一个 SQL:统计每天订单量 top3 的城市",
                "这段 SQL 慢,帮我分析慢在哪:\n{sql}",
            ],
            "stats": [
                "A/B 实验 p=0.045,能下结论说新版更好吗?为什么",
                "解释一下回归分析和相关分析的区别,给业务例子",
            ],
        },
    },
    {
        "persona": "medical_researcher",
        "role": "医学科研助理检索文献",
        "domain": "medicine",
        "difficulty": 2,
        "intents": {
            "search": [
                "帮我检索关于ai辅助诊断肺癌的综述,给出检索式",
                "列5篇meta分析看看二甲双胍对心血管的作用",
            ],
            "explain": [
                "解释一下ROC曲线,AUC接近1意味着什么?",
                "混杂因素在观察性研究里会导致什么问题?",
            ],
        },
    },
    {
        "persona": "founder",
        "role": "初创公司创始人打磨商业计划",
        "domain": "startup",
        "difficulty": 1,
        "intents": {
            "pitch": [
                "给我的AI客服SaaS写10秒电梯演讲",
                "第一轮融资路演,最该强调哪3个数字?",
            ],
            "biz": [
                "免费转付费的定价策略,给3种方案比较",
                "对比自建模型 vs 用第三方LLM API 的成本账",
            ],
        },
    },
    {
        "persona": "translator",
        "role": "本地化译者翻译产品文案",
        "domain": "translation",
        "difficulty": 1,
        "intents": {
            "translate": [
                "把这句话翻成地道的英文:'我们的产品简单可靠,开箱即用。'",
                "中译英:请用口语化的美式英语,不要太正式",
            ],
            "polish": [
                "帮我润色这段产品介绍,更吸引人:\n{content}",
            ],
        },
    },
    {
        "persona": "ml_engineer",
        "role": "机器学习工程师搭建推理服务",
        "domain": "mlops",
        "difficulty": 2,
        "intents": {
            "serving": [
                "对比 vLLM 和 llama.cpp 在 2xT4 上跑 27B 模型的取舍",
                "什么是 continuous batching?对吞吐的影响多大?",
                "TTFT 和 p95 延迟怎么取舍?给调参建议",
            ],
            "quant": [
                "Q4_K_M 和 bf16 对 27B 模型的效果差异怎么评估?",
                "KV cache 显存怎么估算?给公式",
            ],
        },
    },
]

INTENT_LABEL = {
    "explain": "概念解释",
    "prepare": "备课/整理",
    "debug": "代码调试",
    "design": "方案设计",
    "why": "原理问答",
    "fun": "趣味问答",
    "howto": "操作指引",
    "health": "健康生活",
    "summary": "总结提炼",
    "email": "邮件撰写",
    "analysis": "指标分析",
    "sql": "SQL 编写",
    "stats": "统计推断",
    "search": "文献检索",
    "pitch": "路演表达",
    "biz": "商业决策",
    "translate": "翻译",
    "polish": "润色",
    "serving": "推理服务",
    "quant": "量化评估",
}

FILLERS = {
    "code": [
        "def add(a, b):\n    return a+b\n\nprint(add(1))  # 预期 3?",
        "import os\nfiles = os.listdir('.')\nprint(file)\n# NameError: name 'file' is not defined",
        "try:\n    x = 1 / 0\nexcept:\n    pass\n# 异常被吞了,怎么改?",
    ],
    "sql": [
        "SELECT * FROM orders JOIN users ON orders.uid=users.id\nWHERE orders.dt > '2026-01-01' ORDER BY created_at;",
        "SELECT city, COUNT(*) FROM orders GROUP BY city ORDER BY COUNT(*) DESC;",
        "SELECT date_trunc('day', created_at) d, COUNT(*) FROM events WHERE type='click' GROUP BY d;",
    ],
    "content": [
        "周三拜访了华南区大客户,对方对价格有顾虑,对我们的交付周期比较满意;约定下周再报一轮方案。",
        "上海渠道商反馈竞品降价促销,我们合同还有2个月到期,续约率承压。",
        "西南区新签3家代理,首单金额都不大,但试单意愿强烈。",
    ],
}

MAX_TOKENS_HINT = {"difficulty0": 96, "difficulty1": 192, "difficulty2": 384}


def _fill(template: str, rng: random.Random) -> str:
    for key, pool in FILLERS.items():
        marker = "{" + key + "}"
        if marker in template:
            template = template.replace(marker, rng.choice(pool))
    return template


def build_corpus(count: int, seed: int):
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        for p in PERSONAS:
            if len(rows) >= count:
                break
            intent = rng.choice(sorted(p["intents"]))
            prompt = _fill(rng.choice(p["intents"][intent]), rng)
            rows.append({
                "persona": p["persona"],
                "role": p["role"],
                "intent": intent,
                "intent_label": INTENT_LABEL.get(intent, intent),
                "prompt": prompt,
                "domain": p["domain"],
                "difficulty": p["difficulty"],
                "max_tokens_hint": MAX_TOKENS_HINT.get(f"difficulty{p['difficulty']}", 192),
            })
    return rows


def _write(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="persona prompt corpus generator (offline)")
    ap.add_argument("--count", type=int, default=200, help="number of prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="print personas and exit")
    args = ap.parse_args()

    if args.list:
        for p in PERSONAS:
            intents = ", ".join(INTENT_LABEL.get(i, i) for i in sorted(p["intents"]))
            print(f"{p['persona']:22s} | d{p['difficulty']} | {intents}")
        return 0

    rows = build_corpus(args.count, args.seed)
    _write(rows, args.out)
    persona_counts = {}
    for r in rows:
        persona_counts[r["persona"]] = persona_counts.get(r["persona"], 0) + 1
    print(json.dumps({
        "rows": len(rows),
        "personas": len(PERSONAS),
        "out": str(args.out),
        "distribution": persona_counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())