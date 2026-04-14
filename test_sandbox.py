#!/usr/bin/env python3
"""Test gemma4runner sandbox: model writes code, sandbox executes it."""

import json
import re
import time
import urllib.request

BASE = "http://localhost:8080"

def chat(content, max_tokens=600, temperature=0):
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

def test(name, fn):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    try:
        fn()
        print(f"  >> PASS")
    except Exception as e:
        print(f"  >> FAIL: {e}")

# ──────────────────────────────────────────────────────────

def test_data_analysis():
    print("  Step 1: Ask model to write data analysis code...")
    content = chat(
        "Write a Python script that creates a pandas DataFrame of 50 random "
        "students with columns: name, age (18-25), grade (A/B/C/D/F). "
        "Then print: 1) grade distribution, 2) average age per grade, "
        "3) the youngest student. Just the code, no explanation."
    )
    code = extract_code(content)
    print(f"  Step 2: Execute ({len(code)} chars)...")
    result = execute(code)
    print(f"  Exit: {result['exit_code']}")
    if result["stdout"]:
        for line in result["stdout"].strip().split("\n")[:10]:
            print(f"    {line}")
    if result["stderr"]:
        print(f"  stderr: {result['stderr'][:200]}")
    assert result["exit_code"] == 0, f"Code failed: {result['stderr'][:100]}"

def test_fetch_data():
    print("  Step 1: Ask model to write code that fetches data from an API...")
    content = chat(
        "Write a Python script using the requests library that: "
        "1) Fetches the current Bitcoin price from https://api.coindesk.com/v1/bpi/currentprice.json "
        "2) Parses the JSON response "
        "3) Prints the USD price formatted nicely. "
        "Just the code."
    )
    code = extract_code(content)
    print(f"  Step 2: Execute ({len(code)} chars)...")
    result = execute(code)
    print(f"  Exit: {result['exit_code']}")
    print(f"  Output: {result['stdout'][:200]}")
    if result["stderr"]:
        print(f"  stderr: {result['stderr'][:200]}")

def test_multi_step_reasoning():
    print("  Step 1: Ask model to solve a problem step by step...")
    content = chat(
        "I have a list of numbers: [23, 45, 12, 67, 34, 89, 56, 78, 90, 11]. "
        "Write Python code that: "
        "1) Sorts them, 2) Finds the median, 3) Calculates standard deviation, "
        "4) Removes outliers (values > 2 std from mean), "
        "5) Prints each step. Just the code."
    )
    code = extract_code(content)
    print(f"  Step 2: Execute ({len(code)} chars)...")
    result = execute(code)
    print(f"  Exit: {result['exit_code']}")
    for line in result["stdout"].strip().split("\n")[:10]:
        print(f"    {line}")
    assert result["exit_code"] == 0

def test_c_algorithm():
    print("  Step 1: Ask model to write a C sorting algorithm...")
    content = chat(
        "Write a C program that implements quicksort on an array of 10 integers "
        "[64, 34, 25, 12, 22, 11, 90, 1, 55, 33] and prints the sorted array. "
        "Just the code."
    )
    code = extract_code(content, "c")
    print(f"  Step 2: Compile and run ({len(code)} chars)...")
    result = execute(code, "c")
    print(f"  Exit: {result['exit_code']}")
    print(f"  Output: {result['stdout'][:200]}")
    if result["stderr"]:
        print(f"  stderr: {result['stderr'][:200]}")
    assert result["exit_code"] == 0

def test_iterative_debugging():
    print("  Step 1: Give model buggy code to fix...")
    buggy = """
def fibonacci(n):
    if n <= 0:
        return []
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-3])  # BUG: should be i-2
    return fib

print(fibonacci(10))
print("Expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]")
"""
    result1 = execute(buggy)
    print(f"  Buggy output: {result1['stdout'].strip()}")

    print("  Step 2: Ask model to find and fix the bug...")
    content = chat(
        f"This Python code has a bug. The output is wrong:\n\n```python\n{buggy}\n```\n"
        f"Output: {result1['stdout'].strip()}\n\n"
        "Find the bug, fix it, and give me the corrected code. Just the code."
    )
    fixed_code = extract_code(content)
    print(f"  Step 3: Execute fixed code...")
    result2 = execute(fixed_code)
    print(f"  Fixed output: {result2['stdout'].strip()}")
    assert "0, 1, 1, 2, 3, 5, 8, 13, 21, 34" in result2["stdout"]

def test_web_scraping():
    print("  Step 1: Ask model to fetch and parse web data...")
    content = chat(
        "Write Python code using urllib (not requests) that fetches "
        "https://httpbin.org/json and prints the parsed JSON data nicely. "
        "Just the code."
    )
    code = extract_code(content)
    print(f"  Step 2: Execute ({len(code)} chars)...")
    result = execute(code)
    print(f"  Exit: {result['exit_code']}")
    for line in result["stdout"].strip().split("\n")[:5]:
        print(f"    {line}")

def test_file_workflow():
    print("  Step 1: Ask model to write code that creates a CSV...")
    content = chat(
        "Write Python code that creates a CSV file called 'sales.csv' with columns: "
        "month, revenue, expenses. Generate 12 months of fake data. "
        "Then read the CSV back and print a summary (total revenue, total expenses, "
        "profit margin). Just the code."
    )
    code = extract_code(content)
    print(f"  Step 2: Execute ({len(code)} chars)...")
    result = execute(code)
    print(f"  Exit: {result['exit_code']}")
    for line in result["stdout"].strip().split("\n")[:8]:
        print(f"    {line}")

    # Check that the file was created
    from urllib.request import Request, urlopen
    req = Request(f"{BASE}/v1/sandbox/files")
    with urlopen(req) as resp:
        files = json.loads(resp.read())["files"]
    print(f"  Files in workspace: {files}")
    assert "sales.csv" in files

def test_system_inspection():
    print("  Step 1: Run system commands via sandbox...")
    result = run_cmd("echo '=== System ===' && uname -a && echo '' && echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader && echo '' && echo '=== Python ===' && python3 --version && echo '=== Disk ===' && df -h / | tail -1")
    print(f"  Exit: {result['exit_code']}")
    for line in result["stdout"].strip().split("\n"):
        print(f"    {line}")

# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  gemma4runner Sandbox Integration Tests")
    print("  Model writes code -> Sandbox executes -> Verify results")
    print("=" * 65)

    test("1. Data Analysis (Pandas)", test_data_analysis)
    test("2. Multi-step Reasoning", test_multi_step_reasoning)
    test("3. C Algorithm (Quicksort)", test_c_algorithm)
    test("4. Iterative Debugging", test_iterative_debugging)
    test("5. File Workflow (CSV)", test_file_workflow)
    test("6. Web Data Fetch", test_web_scraping)
    test("7. External API (Bitcoin price)", test_fetch_data)
    test("8. System Inspection", test_system_inspection)

    print(f"\n{'=' * 65}")
    print("  All tests complete!")
    print(f"{'=' * 65}")
