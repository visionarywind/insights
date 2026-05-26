#!/bin/bash
# MUSA API Profiler — Test Commands
# Run inside container with MUSA driver.

PROFILER_SO=./liblatency_profiler.so

echo "=== Test 1: Driver API ==="
echo "MUSA_INJECTION64_PATH=$PROFILER_SO tests/test_driver_api"
echo ""

echo "=== Test 2: Runtime API ==="
echo "MUSA_INJECTION64_PATH=$PROFILER_SO tests/test_runtime_api"
echo ""

echo "=== Test 3: Overlay ==="
echo "MUSA_INJECTION64_PATH=$PROFILER_SO tests/test_overlay"
echo ""

echo "=== Test 4: sglang ==="
cat << 'CMD'
source /root/.virtualenvs/sglang-0.5.6/bin/activate
export MUSA_VISIBLE_DEVICES=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profile
export MUSA_INJECTION64_PATH=./liblatency_profiler.so
export PROFILER_TOP_N=25
export PROFILER_OUTPUT=/tmp/profiler_report.txt

python -m sglang.bench_one_batch \
    --model-path /data/shanfeng/models/Qwen/Qwen3-8B \
    --batch 1 --input-len 256 --output-len 64 \
    --profile --dtype float16 --trust-remote-code \
    --device musa --disable-cuda-graph \
    1>/tmp/sglang_stdout.log 2>/tmp/sglang_stderr.log
CMD
