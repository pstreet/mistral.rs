#!/usr/bin/env python3
"""Concurrent throughput benchmark for an OpenAI-compatible server.

Fires N concurrent streaming chat completions and reports aggregate tokens/s,
time-to-first-token, and inter-token latency stats.

Usage:
  python3 conc_bench.py --port 18323 --n 16 --prompt-words 90 --gen-tokens 96 --rounds 1
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def stream_one(port, model, prompt, gen_tokens, lock, results):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rec = {}
    t0 = time.perf_counter()
    ttft = None
    last = t0
    n_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                if usage.get("prompt_tokens"):
                    rec["prompt_tokens"] = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            rec.setdefault("itl", []).append(now - last)
                        last = now
                        n_tokens += 1
    except Exception as e:
        rec["err"] = str(e)
    rec["n_tokens"] = n_tokens
    rec["ttft"] = ttft
    with lock:
        results.append(rec)


def bench(port, model, n, prompt, gen_tokens):
    import threading

    lock = threading.Lock()
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [
            ex.submit(stream_one, port, model, prompt, gen_tokens, lock, results)
            for _ in range(n)
        ]
        for f in futs:
            f.result()
    wall = time.perf_counter() - t0
    total_tokens = sum(r.get("n_tokens", 0) for r in results)
    errs = [r for r in results if "err" in r]
    tps = total_tokens / wall if wall > 0 else 0.0
    ttfts = sorted(r["ttft"] for r in results if r.get("ttft") is not None)
    itls = sorted(x for r in results for x in r.get("itl", []))
    ptok = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
    ptok_s = f" prompt_tok~{sum(ptok) // len(ptok)}" if ptok else ""
    err_s = f" ERRORS={len(errs)}" if errs else ""
    print(
        f"N={n:<3} wall={wall:7.2f}s tok={total_tokens:<5} agg_tps={tps:7.1f}"
        f" | ttft_ms avg={1000 * (sum(ttfts) / len(ttfts) if ttfts else 0):6.0f} "
        f"p50={1000 * pct(ttfts, 0.5):6.0f} p95={1000 * pct(ttfts, 0.95):6.0f}"
        f" | itl_ms avg={1000 * (sum(itls) / len(itls) if itls else 0):6.0f} "
        f"p50={1000 * pct(itls, 0.5):6.0f}{ptok_s}{err_s}",
        flush=True,
    )
    return tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument(
        "--n",
        default="4",
        help="concurrency level (comma list allowed, e.g. 4,8,16)",
    )
    ap.add_argument("--prompt-words", type=int, default=90)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    nlevels = [int(x) for x in str(args.n).split(",")]
    word = "the quick brown fox jumps over the lazy dog. "
    prompt = word * args.prompt_words
    for n in nlevels:
        for r in range(args.rounds):
            tag = "warm" if r == 0 else "run "
            print(f"--- N={n} {tag}round {r} ---", flush=True)
            bench(args.port, args.model, n, prompt, args.gen_tokens)


if __name__ == "__main__":
    main()
