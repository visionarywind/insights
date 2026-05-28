#!/usr/bin/env python3
"""
Offline Qwen3-8B inference script with MUSA kernel profiler enabled.
Run inside mochi-sglang container:
    docker exec mochi-sglang bash -c 'cd /workspace/linux-ddk && python3 musa/doc/profiler/verify_qwen.py'
"""

import os
import sys
import time

# Enable MUSA kernel profiler before any MUSA API call
os.environ["MUSA_INJECTION64_PATH"] = (
    "/workspace/linux-ddk/musa/doc/profiler/libmusaKernelProfiler.so"
)

import torch
import torch_musa  # noqa: F401 - registers MUSA device

torch.musa.set_device(1)  # use GPU 1 (GPU 0 is busy)

def verify_musa():
    assert torch.musa.is_available(), "MUSA not available"
    device = torch.musa.current_device()
    name = torch.musa.get_device_name(device)
    mem_gb = torch.musa.get_device_properties(device).total_memory / 1024**3
    print(f"MUSA OK: device={device} name={name} memory={mem_gb:.1f}GB")

def load_model(model_path, device="musa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} (dtype=torch.bfloat16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},
        trust_remote_code=True,
    )
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    return model, tokenizer

def inference(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt[:80]}...")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"Generating {max_new_tokens} tokens...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    elapsed = time.time() - t0
    gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated {gen_tokens} tokens in {elapsed:.2f}s ({gen_tokens/elapsed:.1f} tok/s)")
    print(f"Response: {response[len(prompt):][:200]}")
    return gen_tokens, elapsed

def main():
    model_path = "/data/shanfeng/models/Qwen/Qwen3-8B"
    assert os.path.isdir(model_path), f"Model not found: {model_path}"

    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # unbuffered

    verify_musa()
    model, tokenizer = load_model(model_path)

    # Warmup: single short inference
    print("\n=== Warmup ===")
    inference(model, tokenizer, "Hello, what is 1+1?", max_new_tokens=10)

    # Real inference
    print("\n=== Inference ===")
    inference(model, tokenizer, "Explain quantum computing in one sentence.", max_new_tokens=50)

    print("\nDone. Check [PROFILER] lines above for GPU kernel timing.")

if __name__ == "__main__":
    main()
