#!/usr/bin/env python3
"""Multi-step demos: model writes code, sandbox runs it, real data flows through."""

import json
import re
import time
import urllib.request

BASE = "http://localhost:8080"

def chat(content, max_tokens=800, temperature=0):
    data = json.dumps({
        "model": "gemma-4", "temperature": temperature, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}]
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def execute(code, language="python"):
    data = json.dumps({"language": language, "code": code}).encode()
    req = urllib.request.Request(f"{BASE}/v1/sandbox/execute", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())

def run_cmd(command):
    data = json.dumps({"command": command}).encode()
    req = urllib.request.Request(f"{BASE}/v1/sandbox/execute", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())

def extract_code(text, lang="python"):
    pattern = rf"```{lang}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def demo(name, fn):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n  FAILED: {e}")

# ─────────────────────────────────────────────────

def demo_weather():
    print("  Asking model to fetch and analyze weather data...")
    content = chat(
        "Write a Python script using requests that fetches weather data from "
        "https://wttr.in/London?format=j1 and prints: "
        "1) Current temperature in C, humidity, and weather description "
        "2) 3-day forecast with high/low temps "
        "3) Sunrise and sunset times. "
        "Just the code."
    )
    code = extract_code(content)
    print(f"  Code: {len(code)} chars, executing...")
    r = execute(code)
    if r["exit_code"] == 0:
        print(f"  Output:\n")
        for line in r["stdout"].strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"  Error: {r['stderr'][:300]}")

def demo_api_chain():
    print("  Asking model to chain multiple API calls...")
    content = chat(
        "Write a Python script that: "
        "1) Fetches a random user from https://randomuser.me/api/ "
        "2) Extracts their name, email, country, and age "
        "3) Then fetches the country info from https://restcountries.com/v3.1/name/{country}?fullText=true "
        "4) Extracts the country's capital, population, and currency "
        "5) Prints a formatted profile combining both. "
        "Use requests library. Just the code."
    )
    code = extract_code(content)
    print(f"  Code: {len(code)} chars, executing...")
    r = execute(code)
    if r["exit_code"] == 0:
        print(f"  Output:\n")
        for line in r["stdout"].strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"  Error: {r['stderr'][:300]}")

def demo_data_pipeline():
    print("  Step 1: Ask model to create a data generation + analysis pipeline...")
    content = chat(
        "Write a Python script that: "
        "1) Generates a CSV file called 'employees.csv' with 200 rows: "
        "   name, department (Engineering/Sales/Marketing/HR/Finance), "
        "   salary (40000-150000), years_experience (0-30), performance_rating (1-5) "
        "2) Reads the CSV back with pandas "
        "3) Calculates and prints: "
        "   - Average salary per department "
        "   - Correlation between years_experience and salary "
        "   - Top 5 highest paid employees "
        "   - Department with best average performance rating "
        "   - Salary percentiles (25th, 50th, 75th, 90th) "
        "Just the code."
    )
    code = extract_code(content)
    print(f"  Code: {len(code)} chars, executing...")
    r = execute(code)
    if r["exit_code"] == 0:
        print(f"  Output:\n")
        for line in r["stdout"].strip().split("\n")[:25]:
            print(f"    {line}")
        total = len(r["stdout"].strip().split("\n"))
        if total > 25:
            print(f"    ... ({total - 25} more lines)")
    else:
        print(f"  Error: {r['stderr'][:300]}")

def demo_system_monitor():
    print("  Asking model to write a system monitoring script...")
    content = chat(
        "Write a Python script that monitors this Linux system and prints: "
        "1) CPU info (model, cores) from /proc/cpuinfo "
        "2) Memory usage (total, used, available) from /proc/meminfo "
        "3) Disk usage for / "
        "4) Top 5 processes by memory usage "
        "5) GPU info from nvidia-smi if available "
        "6) Network interfaces and IPs "
        "Use only standard library (os, subprocess). Just the code."
    )
    code = extract_code(content)
    print(f"  Code: {len(code)} chars, executing...")
    r = execute(code)
    if r["exit_code"] == 0:
        print(f"  Output:\n")
        for line in r["stdout"].strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"  Error: {r['stderr'][:300]}")

def demo_math_proof():
    print("  Asking model to verify a mathematical conjecture with code...")
    content = chat(
        "Write a Python script that verifies Goldbach's conjecture "
        "(every even integer > 2 is the sum of two primes) "
        "for all even numbers from 4 to 10000. "
        "Print how many numbers were checked, and for each of the first 10 "
        "even numbers, show the prime pair decomposition. "
        "Also time how long the verification takes. Just the code."
    )
    code = extract_code(content)
    print(f"  Code: {len(code)} chars, executing...")
    r = execute(code)
    if r["exit_code"] == 0:
        print(f"  Output:\n")
        for line in r["stdout"].strip().split("\n"):
            print(f"    {line}")
    else:
        print(f"  Error: {r['stderr'][:300]}")

def demo_rust_algorithm():
    print("  Asking model to write a Rust algorithm...")
    content = chat(
        "Write a Rust program that implements a binary search tree with insert and "
        "in-order traversal. Insert the values [50, 30, 70, 20, 40, 60, 80, 10], "
        "then print them in sorted order via in-order traversal. Just the code."
    )
    code = extract_code(content, "rust")
    print(f"  Code: {len(code)} chars, compiling and running...")
    r = execute(code, "rust")
    if r["exit_code"] == 0:
        print(f"  Output: {r['stdout'].strip()}")
    else:
        print(f"  Error: {r['stderr'][:300]}")

# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  gemma4runner Multi-Step Demo Suite")
    print("  26B MoE model + CUDA + Sandbox")
    print("=" * 65)

    demo("1. Live Weather Data (wttr.in API)", demo_weather)
    demo("2. API Chain (randomuser + restcountries)", demo_api_chain)
    demo("3. Data Pipeline (generate CSV + pandas analysis)", demo_data_pipeline)
    demo("4. System Monitor (CPU, RAM, GPU, Network)", demo_system_monitor)
    demo("5. Math Verification (Goldbach's Conjecture)", demo_math_proof)
    demo("6. Rust Binary Search Tree", demo_rust_algorithm)

    print(f"\n{'='*65}")
    print("  All demos complete!")
    print(f"{'='*65}")
