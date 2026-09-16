#!/usr/bin/env python3
"""Persona prompt corpus generator (offline, zero dependency, no quota).

Generates a realistic distribution of user prompts across personas + intents,
meant to feed Locust load tests (PERSONAS_CORPUS) or the eval suite.

Output: tmp/personas/corpus.jsonl
  rows: {"persona","role","intent","prompt","domain","difficulty","max_tokens_hint"}

Usage:
  Venv/bin/python app/personas.py                     # default: 200 prompts
  Venv/bin/python app/personas.py --count 500 --seed 7 --out tmp/personas/corpus.jsonl
  ./Venv/bin/python app/personas.py --list            # show personas only
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
        "role": "High-school chemistry teacher prepping a lesson",
        "domain": "chemistry",
        "difficulty": 1,
        "intents": {
            "explain": [
                "Explain chemical equilibrium to me, with examples.",
                "Explain oxidation-reduction reactions in the plainest words possible.",
                "How do I teach the concept of 'electron transfer' without confusing students?",
                "Design a 3-minute classroom demo that shows what a catalyst does.",
            ],
            "prepare": [
                "Write a 10-multiple-choice midterm mock exam for 10th-grade chemistry.",
                "Outline a blackboard plan for a lesson on acid-base neutralization.",
                "Turn this week's topics into a lesson-prep checklist.",
            ],
        },
    },
    {
        "persona": "developer",
        "role": "Backend engineer debugging an issue",
        "domain": "software",
        "difficulty": 2,
        "intents": {
            "debug": [
                "What's wrong with this Python code?\n{code}",
                "Why is my Redis connection pool getting exhausted? Give me a debugging plan.",
                "Write a quick Python script to locate a memory leak.",
            ],
            "design": [
                "Design a rate-limiting scheme for high concurrency; compare token bucket vs leaky bucket.",
                "Give architecture recommendations for an API gateway handling 1000 QPS.",
                "What are the common causes of Postgres index inefficiency, and how to prevent them?",
            ],
        },
    },
    {
        "persona": "student",
        "role": "Sixth-grader asking science-class questions",
        "domain": "general-knowledge",
        "difficulty": 0,
        "intents": {
            "why": [
                "Why is a whale not a fish?",
                "Why is the sky blue? Keep it simple.",
                "Why does a boiled egg get hard?",
                "Why do we have day and night?",
            ],
            "fun": [
                "Tell me a fun piece of trivia about bees.",
                "Give me a quick mental-math problem with numbers under 3.",
            ],
        },
    },
    {
        "persona": "elderly",
        "role": "Elderly first-time smartphone user",
        "domain": "life-help",
        "difficulty": 0,
        "intents": {
            "howto": [
                "How do I pay an electric bill with WeChat on my phone? Step by step, please.",
                "My phone's sound is too quiet, how do I turn it up?",
                "How do I send a photo to my daughter?",
            ],
            "health": [
                "My blood pressure is a bit high, what should I watch out for? Keep it simple.",
                "I'm sleeping badly at night; are there small habits that can help?",
            ],
        },
    },
    {
        "persona": "sales_manager",
        "role": "Sales manager prepping a quarterly review",
        "domain": "business",
        "difficulty": 1,
        "intents": {
            "summary": [
                "Summarize this visit report into 3 key points:\n{content}",
                "Sales dropped 20% this quarter, help me analyze the possible causes.",
                "Write a one-page quarterly summary for my boss.",
            ],
            "email": [
                "Write a sincere but dignified apology letter to a client.",
                "Write a polite but firm price-negotiation email to a supplier.",
            ],
            "analysis": [
                "Pick the 3 metrics that matter most in a sales-funnel report.",
            ],
        },
    },
    {
        "persona": "data_analyst",
        "role": "Data analyst working through a business table",
        "domain": "data",
        "difficulty": 2,
        "intents": {
            "sql": [
                "Write a SQL query: top 3 cities by daily order count.",
                "This SQL is slow; help me analyze where the slowness is:\n{sql}",
            ],
            "stats": [
                "A/B test with p=0.045 — can I conclude the new version is better? Why or why not?",
                "Explain the difference between regression analysis and correlation analysis, with business examples.",
            ],
        },
    },
    {
        "persona": "medical_researcher",
        "role": "Medical research assistant searching the literature",
        "domain": "medicine",
        "difficulty": 2,
        "intents": {
            "search": [
                "Help me search for reviews on AI-assisted lung-cancer diagnosis; give me the search query.",
                "List 5 meta-analyses on metformin's cardiovascular effects so I can check them.",
            ],
            "explain": [
                "Explain the ROC curve: what does an AUC close to 1 mean?",
                "What problems do confounding factors cause in observational studies?",
            ],
        },
    },
    {
        "persona": "founder",
        "role": "Startup founder polishing a business plan",
        "domain": "startup",
        "difficulty": 1,
        "intents": {
            "pitch": [
                "Write a 10-second elevator pitch for my AI customer-support SaaS.",
                "For our first funding round, which 3 numbers should I emphasize in the pitch?",
            ],
            "biz": [
                "Compare 3 pricing options for a free-to-paid conversion.",
                "Break down the cost comparison of building our own model vs using a third-party LLM API.",
            ],
        },
    },
    {
        "persona": "translator",
        "role": "Localization translator working on product copy",
        "domain": "translation",
        "difficulty": 1,
        "intents": {
            "translate": [
                "Translate this into idiomatic English: 'Our product is simple, reliable, and works out of the box.'",
                "Chinese to English: use conversational American English, not too formal.",
            ],
            "polish": [
                "Polish this product intro to make it more compelling:\n{content}",
            ],
        },
    },
    {
        "persona": "ml_engineer",
        "role": "ML engineer standing up an inference service",
        "domain": "mlops",
        "difficulty": 2,
        "intents": {
            "serving": [
                "Compare the trade-offs of vLLM vs llama.cpp for serving a 27B model on 2xT4.",
                "What is continuous batching? How much does it affect throughput?",
                "How should I trade off TTFT vs p95 latency? Give tuning advice.",
            ],
            "quant": [
                "How do I evaluate the quality difference between Q4_K_M and bf16 for a 27B model?",
                "How do I estimate KV-cache memory? Give me the formula.",
            ],
        },
    },
]

INTENT_LABEL = {
    "explain": "Concept explanation",
    "prepare": "Lesson prep / organization",
    "debug": "Code debugging",
    "design": "Solution design",
    "why": "How/why questions",
    "fun": "Fun questions",
    "howto": "Step-by-step help",
    "health": "Healthy living",
    "summary": "Summarization",
    "email": "Email writing",
    "analysis": "Metric analysis",
    "sql": "SQL writing",
    "stats": "Statistical inference",
    "search": "Literature search",
    "pitch": "Pitching",
    "biz": "Business decisions",
    "translate": "Translation",
    "polish": "Polishing",
    "serving": "Inference serving",
    "quant": "Quantization eval",
}

FILLERS = {
    "code": [
        "def add(a, b):\n    return a+b\n\nprint(add(1))  # expected 3?",
        "import os\nfiles = os.listdir('.')\nprint(file)\n# NameError: name 'file' is not defined",
        "try:\n    x = 1 / 0\nexcept:\n    pass\n# the exception is swallowed, how to fix?",
    ],
    "sql": [
        "SELECT * FROM orders JOIN users ON orders.uid=users.id\nWHERE orders.dt > '2026-01-01' ORDER BY created_at;",
        "SELECT city, COUNT(*) FROM orders GROUP BY city ORDER BY COUNT(*) DESC;",
        "SELECT date_trunc('day', created_at) d, COUNT(*) FROM events WHERE type='click' GROUP BY d;",
    ],
    "content": [
        "Visited a key account in South China on Wednesday; the client is concerned about pricing but satisfied with our delivery timeline; agreed to send another proposal next week.",
        "A Shanghai channel partner reported a competitor's promotional price cut; our contract expires in 2 months and the renewal rate is under pressure.",
        "Signed 3 new agents in the Southwest region; first orders are small in value but their willingness to trial is strong.",
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
