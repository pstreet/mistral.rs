#!/usr/bin/env python3
"""Measure cold-prefill throughput (prompt tokens / s) on an OpenAI-compatible server.

For each unique prompt it sends a single request with max_tokens=1 and times the
time-to-first-token, reporting prefill tok/s = prompt_tokens / TTFT.

Usage:
  python3 prefill_bench.py --port 1235 --prompt-words 90 --rounds 4
"""

import argparse
import json
import time
import urllib.request


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def measure(port, model, prompt, gen_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.load(resp)
    ttft = time.perf_counter() - t0
    usage = d.get("usage") or {}
    return ttft, usage.get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--prompt-words", type=int, default=90)
    ap.add_argument("--gen-tokens", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    word = "the quick brown fox jumps over the lazy dog. "
    sentences = [
        "The quantum turbine spins under a violet noon sky. ",
        "Every porcelain statue hides a wooden exoskeleton in shade. ",
        "Mycelial banks relay pressure through porous limestone halls. ",
        "A tarnished sextant reads the arc across the amber fjord. ",
        "Bristlecone roots anchor driftwood while gulls patrol piers. ",
        "The alabaster ledger tallies silver against the riverbed. ",
        "Silhouettes of cranes stitch the horizon into routed cloth. "
        "Copper filaments coil beneath the trampled marigold field. ",
        "A whetstone hums on the lathe of the forgotten lighthouse. ",
        "Rainwater channels split the basalt plaza into brass seams. ",
    ]
    results = []
    for r in range(args.rounds):
        prompt = sentences[r % len(sentences)].strip() * args.prompt_words
        ttft, p = measure(args.port, args.model, prompt, args.gen_tokens)
        tps = p / ttft if ttft > 0 else 0.0
        results.append((p, ttft, tps))
        print(
            f"{args.tag}round {r}: prompt_tok={p:<6} ttft={ttft:8.3f}s "
            f"prefill_tps={tps:7.1f}",
            flush=True,
        )

    tps_vals = sorted(r[2] for r in results)
    ttft_vals = sorted(r[1] for r in results)
    ps = [r[0] for r in results]
    print(
        f"{args.tag}SUMMARY: prefill_tps avg={sum(tps_vals) / len(tps_vals):7.1f} "
        f"p50={pct(tps_vals, 0.5):7.1f} | prompt_tok avg={sum(ps) // len(ps)} "
        f"| ttft avg={1000 * sum(ttft_vals) / len(ttft_vals):7.0f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
