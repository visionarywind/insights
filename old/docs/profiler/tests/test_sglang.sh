#!/bin/bash
# Test 4: sglang Integration
# ===========================
# Demonstrates zero-code-change profiler injection into sglang.
# The profiler intercepts ALL MUSA API calls during sglang inference,
# producing a report with both Driver (mu*) and Runtime (musa*) layers.
#
# Usage:
#   cd docs/profiler && bash tests/test_sglang.sh
#
# Requirements:
#   - Container with MUSA driver + sglang installed
#   - Model at /data/shanfeng/models/Qwen/Qwen3-8B
#   - liblatency_profiler.so compiled

set -e

PROFILER_SO="${PROFILER_SO:-./liblatency_profiler.so}"
MODEL_PATH="${MODEL_PATH:-/data/shanfeng/models/Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sglang_profile_results}"

echo "============================================"
echo " Test 4: sglang + MUSA API Profiler"
echo "============================================"
echo ""
echo "Profiler: $PROFILER_SO"
echo "Model:    $MODEL_PATH"
echo ""

# ── Build if needed ─────────────────────────────────────────────────────────
if [ ! -f "$PROFILER_SO" ]; then
    echo "[BUILD] Compiling profiler..."
    make profiler
fi

# ── Run sglang bench_one_batch ──────────────────────────────────────────────
echo "[RUN] Starting sglang bench_one_batch with profiler..."
echo ""

source /root/.virtualenvs/sglang-0.5.6/bin/activate 2>/dev/null || true

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-1}"
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export MUSA_INJECTION64_PATH="$(realpath "$PROFILER_SO")"
export PROFILER_TOP_N=30
export PROFILER_OUTPUT="${OUTPUT_DIR}/profiler_report.txt"

mkdir -p "$OUTPUT_DIR"

python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --batch 1 \
    --input-len 256 \
    --output-len 64 \
    --profile \
    --dtype float16 \
    --trust-remote-code \
    --device musa \
    --disable-cuda-graph \
    1>"${OUTPUT_DIR}/sglang_stdout.log" \
    2>"${OUTPUT_DIR}/sglang_stderr.log"

echo ""
echo "============================================"
echo " Results"
echo "============================================"
echo ""

# ── Show sglang metrics ─────────────────────────────────────────────────────
echo "[sglang Latency Metrics]"
grep -E "Prefill\. latency|Decode\.\s+median|Total\. latency" "${OUTPUT_DIR}/sglang_stdout.log" | head -10

echo ""
echo "[Profiler Report — Top 15 Driver/Runtime APIs]"
head -20 "${OUTPUT_DIR}/profiler_report.txt"

echo ""
echo "Full report:     ${OUTPUT_DIR}/profiler_report.txt"
echo "sglang stdout:   ${OUTPUT_DIR}/sglang_stdout.log"
echo "sglang stderr:   ${OUTPUT_DIR}/sglang_stderr.log"
echo "Chrome trace:    /tmp/sglang_profile/"
