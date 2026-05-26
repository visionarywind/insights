#!/usr/bin/env python3
"""Offline inference script to verify profiler on real Qwen3-8B model."""
import argparse
import dataclasses
import json
import logging
import os
import sys
import time

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"[PROFILE] Loading model: {args.model_path}", file=sys.stderr, flush=True)
    t0 = time.time()

    server_args = ServerArgs(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        skip_tokenizer_init=False,
        log_level="error",
        disable_cuda_graph=True,
    )
    engine = Engine(**dataclasses.asdict(server_args))

    print(f"[PROFILE] Model loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    prompts = [
        "Hello, please introduce yourself briefly.",
        "What are the main differences between Python and C++?",
        "Explain the concept of deep learning in simple terms.",
    ][:args.num_prompts]

    sampling_params = [
        {"temperature": args.temperature, "max_new_tokens": args.max_tokens}
        for _ in prompts
    ]

    print(f"[PROFILE] Generating {len(prompts)} responses...", file=sys.stderr, flush=True)
    t1 = time.time()

    gen_out = engine.generate(prompt=prompts, sampling_params=sampling_params)

    elapsed = time.time() - t1
    print(f"[PROFILE] Generation completed in {elapsed:.2f}s", file=sys.stderr, flush=True)

    for i, out in enumerate(gen_out):
        text = out.get("text", "")
        meta = out.get("meta_info", {})
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: {prompts[i][:80]}...")
        print(f"Response: {text}")
        print(f"Meta: {json.dumps(meta, indent=2)}")

    total_tokens = sum(o.get("meta_info", {}).get("completion_tokens", 0) for o in gen_out)
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"[PROFILE] Total generated tokens: {total_tokens}", file=sys.stderr, flush=True)
    print(f"[PROFILE] Throughput: {total_tokens/elapsed:.1f} tok/s", file=sys.stderr, flush=True)

    print(f"[PROFILE] Shutting down engine...", file=sys.stderr, flush=True)
    engine.shutdown()
    print(f"[PROFILE] Done. Profiler report above.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
