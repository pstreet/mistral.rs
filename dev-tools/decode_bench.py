#!/usr/bin/env python3
"""Single-stream decode benchmark for an OpenAI-compatible server.

Streams one chat completion and reports TTFT, mean/p50/p95 inter-token
latency, wall decode tok/s, and server-reported tok/s.

Usage:
  python3 decode_bench.py --port 1235 --model qwen3.8-27b --gen-tokens 600 \
      --rounds 3 --prompt-type essay
"""

import argparse
import json
import time
import urllib.request


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


PROMPTS = {
    "essay": "Write a well-structured essay on the history of the printed book, from Gutenberg's press through paperback mass markets to modern print-on-demand. Cover technology, economics, and social impact.",
    "arithmetic": "What is 17 * 23? Explain the multiplication step by step, then verify it with a second method.",
    "chat": "Tell me about your favorite corner of the world and why it matters to you.",
}


def bench_round(port, model, prompt, gen_tokens, temperature):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rec = {"err": None}
    t0 = time.perf_counter()
    ttft = None
    last = t0
    itl = []
    n = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
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
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itl.append(now - last)
                        last = now
                        n += 1
    except Exception as e:
        rec["err"] = str(e)
        return rec
    wall = time.perf_counter() - t0
    decode_wall = wall - (ttft or 0)
    rec.update(
        n_tokens=n,
        ttft=ttft,
        wall=wall,
        decode_tps=n / decode_wall if decode_wall > 0 and n > 1 else 0.0,
        itl_mean=sum(itl) / len(itl) if itl else 0.0,
        itl_p50=pct(sorted(itl), 0.5),
        itl_p95=pct(sorted(itl), 0.95),
        usage=usage,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--gen-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prompt-type", default="essay", choices=sorted(PROMPTS))
    ap.add_argument("--prompt-file", default="", help="read prompt text from a file")
    args = ap.parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file).read()
    else:
        prompt = PROMPTS[args.prompt_type]

    for r in range(args.rounds):
        rec = bench_round(
            args.port, args.model, prompt, args.gen_tokens, args.temperature
        )
        if rec["err"]:
            print(f"round {r}: ERROR {rec['err']}", flush=True)
            continue
        u = rec["usage"]
        server_tps = (
            (u.get("completion_tokens") or 0) / u["completion_time"]
            if u.get("completion_time")
            else 0.0
        )
        extra = ""
        if u.get("completion_time"):
            extra = f" | server_tok={u.get('completion_tokens')} server_tps={server_tps:5.2f}"
        print(
            f"round {r}: tok={rec['n_tokens']:<4} ttft={rec['ttft'] or 0:6.2f}s "
            f"decode_tps={rec['decode_tps']:6.2f} itl_ms avg={1000 * rec['itl_mean']:6.1f} "
            f"p50={1000 * rec['itl_p50']:6.1f} p95={1000 * rec['itl_p95']:6.1f}{extra}",
            flush=True,
        )
        time.sleep(2)


if __name__ == "__main__":
    main()
