#!/usr/bin/env python3
"""Locust load test for OpenAI-compatible LLM endpoints (TPU vLLM / GPU llama.cpp).

Metrics captured out of the box by Locust: RPS, response time (p50/p95),
failure rate. This file additionally records:
  - TTFT  (time to first token, streaming)  -> CSV  TTFT
  - total stream duration + chunk count     -> same CSV

Env vars:
  ENDPOINT  base URL incl /v1   (default: http://127.0.0.1:8000/v1)
  API_KEY   bearer key          (default: sk-test)
  MODEL     model id            (default: qwen3.8-27b)
  MODE      tpu | gpu           (tpu adds chat_template_kwargs for vLLM)
  TTFT_CSV  out csv path        (default: tmp/ttft.csv)
  PERSONAS  jsonl corpus path   (default: unset -> static PROMPTS)
"""
import json
import os
import csv
import random
import time
from pathlib import Path

from locust import HttpUser, between, task

BASE = os.environ.get("ENDPOINT", "http://127.0.0.1:8000/v1")
KEY = os.environ.get("API_KEY", "sk-test")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
MODE = os.environ.get("MODE", "tpu")
TTFT_CSV = os.environ.get("TTFT_CSV", "tmp/ttft.csv")
PATH = "/v1/chat/completions"

HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

PROMPTS = [
    "Explain the difference between quantization and distillation in one paragraph.",
    "Write a python function that returns the n-th Fibonacci number.",
    "Summarize: attention is all you need (one short paragraph).",
    "List three production monitoring metrics for an LLM API and why each matters.",
    "Translate 'The quick brown fox' into French.",
    "Draft a polite two-line email declining an unplanned meeting.",
    "What is the VRAM budget formula for a KV cache? Be concise.",
    "Which is better for a 27B model on 2xT4: vLLM or llama.cpp? Justify briefly.",
    "List files in '.' using the list_files tool, then say what you see.",
    "Explain continuous batching in serving LLMs.",
    "Give the trade-offs of Q4_K_M vs bf16 for a 27B model.",
    "Write a bash one-liner to find the 5 largest files under /tmp.",
    "Concept: what is time-to-first-token and how does batching affect it?",
    "Draft a README section titled 'Skills Used' for a serving repo.",
    "Math: if KV is 0.13 MiB/token and VRAM is 12 GiB, max ctx?",
]

with_kt = MODE == "tpu"


def _load_personas():
    path = os.environ.get("PERSONAS")
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"PERSONAS corpus not found: {p}")
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"PERSONAS corpus empty: {p}")
    return rows


PERSONAS = _load_personas()


def _prompt_pool():
    if PERSONAS:
        return [r["prompt"] for r in PERSONAS]
    return PROMPTS


def _body(prompt, stream):
    d = {"model": MODEL,
         "messages": [{"role": "system",
                       "content": "You are a concise helpful engineer."},
                      {"role": "user", "content": prompt}],
         "max_tokens": random.choice([32, 48, 64, 128, 256]),
         "temperature": 0.7,
         "stream": stream}
    if with_kt:
        d["chat_template_kwargs"] = {"enable_thinking": False}
    return d


class QwenUser(HttpUser):
    host = "/".join(BASE.split("/")[:3])
    wait_time = between(0.4, 2.4)

    def _stream(self, prompt):
        t0 = time.time()
        with self.client.post(PATH, json=_body(prompt, True),
                              headers=HEADERS, catch_response=True,
                              stream=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
                return
            ttft = tot = None
            chunks = 0
            for line in r.iter_lines():
                if not line:
                    continue
                chunks += 1
                if ttft is None:
                    ttft = time.time() - t0
            tot = time.time() - t0
            if ttft is None:
                r.failure("no stream chunks")
                return
            try:
                with open(TTFT_CSV, "a") as f:
                    f.write(f"{time.time():.3f},{ttft:.3f},{tot:.3f},{chunks}\n")
            except Exception:
                pass
            r.success()

    def _sync(self, prompt):
        with self.client.post(PATH, json=_body(prompt, False),
                              headers=HEADERS, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                r.success()

    @task(6)
    def chat_stream(self):
        self._stream(random.choice(_prompt_pool()))

    @task(3)
    def chat_sync(self):
        self._sync(random.choice(_prompt_pool()))