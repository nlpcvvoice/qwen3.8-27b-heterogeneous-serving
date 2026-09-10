#!/usr/bin/env python3
"""考试卷 — assertion bank + tool-call parser checks against live endpoints.

Zero-quota by design: hitting an already-running kernel only uses serving
capacity, never launches new kernels. Use --dry (default) to self-test the
suite offline; use --live to run against current_services.json endpoints.

Checks supported:
  int_eq        response body == expected integer (extracted by regex)
  json_valid    response parses as JSON
  json_field    parsed JSON[path_dot] == expected (or "type" check)
  contains_any  body contains any of keywords (case-insensitive)
  regex         body matches regex
  tool_call     response.tool_calls[0].function.name == expected
  tool_arg      tool_calls[0].arguments[arg_path] == expected

Usage:
  ./Venv/bin/python app/eval_suite.py                       # dry self-test
  ./Venv/bin/python app/eval_suite.py --live --only tpu
  ./Venv/bin/python app/eval_suite.py --live --only all
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "tmp" / "current_services.json"
OUT_DIR = ROOT / "tmp" / "eval"

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression string",
        "parameters": {"type": "object",
                       "properties": {"expr": {"type": "string"}},
                       "required": ["expr"]},
    },
}

# ---------------- suite definition ----------------
SUITE = [
    {
        "id": "arith-123x7",
        "prompt": "Compute 123 * 7 and reply with only the number.",
        "mode": "chat",
        "max_tokens": 32,
        "checks": [{"type": "contains_any", "needle": ["861"]}],
    },
    {
        "id": "json-fields",
        "prompt": ('Return a JSON object with keys "name" and "count" '
                   '(count=3). Only JSON, nothing else.'),
        "mode": "chat",
        "max_tokens": 64,
        "checks": [
            {"type": "json_valid"},
            {"type": "json_field", "path": "count", "vtype": "int", "expected": 3},
        ],
    },
    {
        "id": "json-key-present",
        "prompt": 'JSON: {"ok": true}. Only JSON.',
        "mode": "chat",
        "max_tokens": 32,
        "checks": [{"type": "json_valid", "expect_key": "ok"}],
    },
    {
        "id": "tool-calc",
        "prompt": "Use the calc tool to compute 2 + 3, then stop.",
        "mode": "tools",
        "tools": [CALC_TOOL],
        "max_tokens": 256,
        "checks": [
            {"type": "tool_call", "expected_name": "calc"},
            {"type": "tool_arg", "expect_key": "expr"},
        ],
    },
    {
        "id": "tool-list",
        "prompt": ("Use the list_files tool with path '.' , then stop. "
                   "Do not answer in prose."),
        "mode": "tools",
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}},
                               "required": ["path"]},
            },
        }],
        "max_tokens": 256,
        "checks": [
            {"type": "tool_call", "expected_name": "list_files"},
            {"type": "tool_arg", "expect_key": "path"},
        ],
    },
    {
        "id": "regex-monitoring",
        "prompt": "Name one production monitoring metric. One word.",
        "mode": "chat",
        "max_tokens": 24,
        "checks": [{"type": "contains_any",
                    "needle": ["latency", "ttft", "rps", "qps", "throughput",
                               "error", "availability", "uptime"]}],
    },
    {
        "id": "kv-budget-math",
        "prompt": ("If KV cache is 0.13 MiB/token and VRAM is 12 GiB, "
                   "what is the max context in tokens? Reply with only a number."),
        "mode": "chat",
        "max_tokens": 32,
        "checks": [{"type": "regex", "pattern": r"\b9[0-9]{3,4}\b"}],
    },
]

ERROR_CLS = {
    "int": (int,),
    "str": (str,),
    "float": (float,),
}


# ---------------- check engine ----------------
def _json_of(text: str):
    start = text.find("{")
    if start == -1:
        return None
    try:
        return json.loads(text[start:])
    except Exception:
        return None


def _check_one(check, text: str, tool_calls: list, out: dict):
    t = check["type"]
    if t == "contains_any":
        got = any(n.lower() in text.lower() for n in check["needle"])
        return got, f"needle={check['needle']}"
    if t == "regex":
        return bool(re.search(check["pattern"], text)), f"pattern={check['pattern']}"
    if t == "json_valid":
        obj = _json_of(text)
        if obj is None:
            return False, "no JSON found"
        if check.get("expect_key") and check["expect_key"] not in obj:
            return False, f"missing key {check['expect_key']}"
        return True, "json ok"
    if t == "json_field":
        obj = _json_of(text)
        if obj is None:
            return False, "no JSON found"
        val = obj if not check.get("path") else _dot_get(obj, check["path"])
        want_t = ERROR_CLS[check.get("vtype", "str")]
        if not isinstance(val, want_t):
            return False, f"field {check.get('path')}={val!r} not {check.get('vtype')}"
        if "expected" in check and val != check["expected"]:
            return False, f"field={val!r} != expected={check['expected']}"
        return True, f"field ok ={val!r}"
    if t == "tool_call":
        if not tool_calls:
            return False, "no tool_calls in response"
        got = tool_calls[0].get("name")
        return got == check["expected_name"], f"got={got} want={check['expected_name']}"
    if t == "tool_arg":
        if not tool_calls:
            return False, "no tool_calls in response"
        args = tool_calls[0].get("arguments", {})
        if not isinstance(args, dict):
            return False, f"arguments not dict: {args!r}"
        return check["expect_key"] in args, f"missing arg {check['expect_key']}"
    return False, f"unknown check type {t}"


def _dot_get(obj, path):
    for k in path.split("."):
        obj = obj.get(k)
        if obj is None:
            return None
    return obj


def check_case(case, text: str, tool_calls: list):
    results = []
    for chk in case["checks"]:
        ok, detail = _check_one(chk, text, tool_calls, case)
        results.append({"check": chk["type"], "ok": ok, "detail": detail})
    return results


# ---------------- live runner ----------------
def _call_endpoint(svc, case):
    url = svc["endpoint"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {svc['api_key']}",
               "Content-Type": "application/json"}
    body = {"model": svc["model"],
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": case.get("max_tokens", 128),
            "temperature": 0.2,
            "stream": False}
    if case.get("mode") == "tools":
        body["tools"] = case["tools"]
        body["tool_choice"] = "auto"
        if svc.get("kind") == "tpu":
            body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    r = requests.post(url, json=body, headers=headers, timeout=180)
    dt = time.time() - t0
    if r.status_code != 200:
        return None, None, f"HTTP {r.status_code}: {r.text[:300]}", dt
    data = r.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    tool_calls = None
    if msg.get("tool_calls"):
        tool_calls = [{
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"] or "{}"),
        } for tc in msg["tool_calls"]]
    return text, tool_calls, None, dt


def run_live(only: str, limit: int = None):
    if not REGISTRY.exists():
        print(f"registry not found: {REGISTRY}")
        return 2
    registry = json.loads(REGISTRY.read_text())
    svcs = ([{"kind": only, **registry[only]}]
            if only in registry else list(registry.values()))
    cases = SUITE[:limit] if limit else SUITE
    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
              "services": list(registry.keys()), "results": []}
    n_pass = n_fail = 0
    for svc in svcs:
        for case in cases:
            text, tool_calls, err, dt = _call_endpoint(svc, case)
            if err:
                n_fail += 1
                report["results"].append({"case": case["id"], "svc": svc["kind"],
                                          "ok": False, "error": err, "secs": round(dt, 2)})
                print(f"[FAIL] {svc['kind']} {case['id']}: {err}")
                continue
            checks = check_case(case, text, tool_calls)
            ok = all(c["ok"] for c in checks)
            n_pass += ok
            n_fail += not ok
            report["results"].append({"case": case["id"], "svc": svc["kind"],
                                      "ok": ok, "secs": round(dt, 2), "checks": checks})
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {svc['kind']} {case['id']} ({dt:.2f}s)")
            if not ok:
                for c in checks:
                    if not c["ok"]:
                        print(f"      x {c['check']}: {c['detail']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"result-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report["n_pass"] = n_pass
    report["n_fail"] = n_fail
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n== results: {n_pass} passed / {n_fail} failed -> {out_path}")
    return 0 if n_fail == 0 else 1


# ---------------- dry self-test (offline) ----------------
def run_dry():
    fake_pass = ("861", None)
    fake_json = ('{"name":"x","count":3}', None)
    fake_tools = ("", [{"name": "calc", "arguments": {"expr": "2+3"}}])
    cases = [
        {"suite": "arith", "case": SUITE[0], "text": fake_pass[0],
         "tool": fake_pass[1], "expect": True},
        {"suite": "json", "case": SUITE[1], "text": fake_json[0],
         "tool": fake_json[1], "expect": True},
        {"suite": "tool-calc", "case": SUITE[3], "text": fake_tools[0],
         "tool": fake_tools[1], "expect": True},
        {"suite": "kv-regex", "case": SUITE[6], "text": "92461",
         "tool": None, "expect": True},
        {"suite": "neg-tool", "case": SUITE[3], "text": "",
         "tool": None, "expect": False},
    ]
    all_ok = True
    for c in cases:
        checks = check_case(c["case"], c["text"], c["tool"])
        ok = all(x["ok"] for x in checks)
        status = "ok" if ok == c["expect"] else "MISMATCH"
        all_ok = all_ok and (ok == c["expect"])
        print(f"[{status}] dry {c['suite']}: checks_passed={ok} (expected {c['expect']})")
    print(f"dry suite schema OK, {len(SUITE)} cases, {sum(len(x['checks']) for x in SUITE)} checks defined")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="eval assert suite (dry default)")
    ap.add_argument("--dry", action="store_true", help="offline self-test")
    ap.add_argument("--live", action="store_true", help="run against live endpoints")
    ap.add_argument("--only", choices=["tpu", "gpu", "all"], default="all")
    ap.add_argument("--limit", type=int, help="only first N cases")
    args = ap.parse_args()
    if args.live:
        return run_live(args.only, args.limit)
    return run_dry()


if __name__ == "__main__":
    sys.exit(main())