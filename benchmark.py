#!/usr/bin/env python3
"""gemma4runner comprehensive benchmark suite"""

import json
import time
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

def api_call(endpoint, data=None, timeout=120):
    url = f"{BASE}{endpoint}"
    if data:
        req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            elapsed = time.time() - start
            return body, elapsed
    except Exception as e:
        return {"error": str(e)}, time.time() - start

def chat(content, max_tokens=50, temperature=0, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    return api_call("/v1/chat/completions", {
        "model": "gemma-4", "messages": msgs,
        "temperature": temperature, "max_tokens": max_tokens
    })

def get_content(resp):
    try:
        return resp["choices"][0]["message"]["content"]
    except:
        return "ERROR: " + str(resp.get("error", resp))

def get_usage(resp):
    try:
        return resp["usage"]
    except:
        return {}

def check(label, resp, elapsed, condition, content):
    global tests_passed, tests_total
    tests_total += 1
    ok = condition
    if ok:
        tests_passed += 1
    tag = "PASS" if ok else "FAIL"
    preview = content.strip().replace("\n", " ")[:50]
    print(f"  {tag} | {label:24s} -> {preview}")

tests_passed = 0
tests_total = 0

print("=" * 65)
print("  gemma4runner Comprehensive Test Suite")
print(f"  Target: {BASE}")
print("=" * 65)

# ── Endpoint Tests ──
print("\n-- Endpoint Tests --")

r, t = api_call("/health")
tests_total += 1
tests_passed += 1
print(f"  PASS | /health                  ({t:.2f}s)")

r, t = api_call("/v1/models")
ok = r.get("object") == "list"
tests_total += 1; tests_passed += int(ok)
print(f"  {'PASS' if ok else 'FAIL'} | /v1/models               ({t:.2f}s)")

r, t = api_call("/metrics")
ok = "total_requests" in r
tests_total += 1; tests_passed += int(ok)
print(f"  {'PASS' if ok else 'FAIL'} | /metrics                 ({t:.2f}s)")

# ── Correctness Tests ──
print("\n-- Correctness Tests --")

r, t = chat("What is 2+2? Just the number.", max_tokens=10)
c = get_content(r)
check("Math: 2+2", r, t, "4" in c, c)

r, t = chat("What is 15 * 7? Just the number.", max_tokens=10)
c = get_content(r)
check("Math: 15*7", r, t, "105" in c, c)

r, t = chat("What is the capital of France? One word.", max_tokens=10)
c = get_content(r)
check("Knowledge: France", r, t, "paris" in c.lower(), c)

r, t = chat("What is the capital of Japan? One word.", max_tokens=10)
c = get_content(r)
check("Knowledge: Japan", r, t, "tokyo" in c.lower(), c)

r, t = chat("Who wrote Romeo and Juliet? Just the name.", max_tokens=15)
c = get_content(r)
check("Knowledge: Shakespeare", r, t, "shakespeare" in c.lower(), c)

r, t = chat("What planet is closest to the Sun? One word.", max_tokens=10)
c = get_content(r)
check("Knowledge: Mercury", r, t, "mercury" in c.lower(), c)

r, t = chat("If all cats are animals and Whiskers is a cat, is Whiskers an animal? Yes or no.", max_tokens=10)
c = get_content(r)
check("Reasoning: Logic", r, t, "yes" in c.lower(), c)

r, t = chat("What comes next: 2, 4, 6, 8, ?", max_tokens=10)
c = get_content(r)
check("Reasoning: Pattern", r, t, "10" in c, c)

r, t = chat("Translate 'hello' to Spanish. Just the word.", max_tokens=10)
c = get_content(r)
check("Language: Translate", r, t, "hola" in c.lower(), c)

r, t = chat("Translate 'thank you' to French. Just the phrase.", max_tokens=10)
c = get_content(r)
check("Language: French", r, t, "merci" in c.lower(), c)

r, t = chat("Write a Python function that returns the sum of two numbers. Just the code.", max_tokens=80)
c = get_content(r)
has_code = "def " in c and "return" in c
check("Code: Python func", r, t, has_code, "has def+return" if has_code else c)

r, t = chat("What is your name? One sentence.", max_tokens=30)
c = get_content(r)
check("Identity", r, t, "gemma" in c.lower() or "model" in c.lower(), c)

r, t = chat("What are you?", max_tokens=30, system="You are a helpful pirate. Respond in pirate speak.")
c = get_content(r)
check("System prompt", r, t, len(c) > 10, c)

r, t = api_call("/v1/completions", {
    "model": "gemma-4", "prompt": "The capital of Germany is", "temperature": 0, "max_tokens": 10
})
c = r.get("choices", [{}])[0].get("text", "ERROR")
check("/v1/completions", r, t, "berlin" in c.lower(), c)

# ── Performance Benchmarks ──
print("\n-- Performance Benchmarks --")

for label, prompt, max_tok in [
    ("Short  (5 tok)", "Say hi.", 5),
    ("Medium (100 tok)", "Explain what a neural network is.", 100),
    ("Long   (500 tok)", "Write a detailed essay about the history of computing.", 500),
    ("XL     (1000 tok)", "Write a comprehensive guide to machine learning covering supervised, unsupervised, and reinforcement learning.", 1000),
]:
    r, t = chat(prompt, max_tokens=max_tok)
    u = get_usage(r)
    comp = u.get("completion_tokens", 0)
    prompt_tok = u.get("prompt_tokens", 0)
    tps = comp / t if t > 0 else 0
    print(f"  {label:18s} | {comp:4d} tokens in {t:6.2f}s = {tps:6.1f} tok/s  (prompt: {prompt_tok})")

# ── Final Metrics ──
print("\n-- Final Metrics --")
r, _ = api_call("/metrics")
for k, v in r.items():
    if isinstance(v, float):
        print(f"  {k:35s} = {v:.2f}")
    else:
        print(f"  {k:35s} = {v}")

print(f"\n{'=' * 65}")
print(f"  Results: {tests_passed}/{tests_total} tests passed")
print(f"{'=' * 65}")
