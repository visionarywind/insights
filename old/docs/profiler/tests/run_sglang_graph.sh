#!/bin/bash
# ============================================================================
# sglang + MUSA Graph + API Profiler — Docker Container Script
# ============================================================================
# Runs sglang bench_one_batch with MUSA graph enabled + API profiler injected.
# Captures all output to /tmp/sglang_graph_results/
#
# Usage (inside container):
#   bash run_sglang_graph.sh [GPU_ID] [MODEL_PATH]
#
# Defaults:
#   GPU_ID     = 2
#   MODEL_PATH = /data/shanfeng/models/Qwen/Qwen3-8B
# ============================================================================

set -e

GPU_ID="${1:-2}"
MODEL_PATH="${2:-/data/shanfeng/models/Qwen/Qwen3-8B}"
BATCH="${3:-1}"
INPUT_LEN="${4:-256}"
OUTPUT_LEN="${5:-64}"
OUTPUT_DIR="/tmp/sglang_graph_results"

# ── Container environment ────────────────────────────────────────────────────
export MUSA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

# ── sglang virtualenv ────────────────────────────────────────────────────────
source /root/.virtualenvs/sglang-0.5.6/bin/activate
export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH

# ── sglang torch.profiler settings ───────────────────────────────────────────
export SGLANG_TORCH_PROFILER_DIR="${OUTPUT_DIR}/chrome_trace"

# ── API Profiler settings ────────────────────────────────────────────────────
PROFILER_SO="${PROFILER_SO:-/tmp/profiler/musa_profiler.so}"
export MUPTI_API_PROFILE_TOP_N=30
export MUPTI_API_PROFILE_OUTPUT="${OUTPUT_DIR}/api_profiler_report.txt"
export MUPTI_API_PROFILE_USE_PUBLIC_CALLBACK=true

# ── Setup output directory ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/chrome_trace"

echo "============================================================================"
echo " sglang + MUSA Graph + API Profiler"
echo "============================================================================"
echo " GPU:        $GPU_ID"
echo " Model:      $MODEL_PATH"
echo " Batch:      $BATCH, Input: $INPUT_LEN, Output: $OUTPUT_LEN"
echo " Profiler:   $PROFILER_SO"
echo " Output:     $OUTPUT_DIR"
echo "============================================================================"
echo ""

# ── Verify prerequisites ─────────────────────────────────────────────────────
echo "[CHECK] Verifying environment..."
python3 -c "import torch; import torch_musa; print(f'  torch={torch.__version__}, devices={torch.musa.device_count()}')" || {
    echo "ERROR: torch + torch_musa not available"
    exit 1
}

python3 -c "import sglang; print(f'  sglang={sglang.__version__}')" 2>/dev/null || {
    echo "WARN: sglang import via pip failed, using PYTHONPATH fallback"
}

FREE=$(python3 -c "import torch, torch_musa; f,_=torch.musa.mem_get_info(0); print(f'{f/1024**3:.1f}')")
echo "  GPU $GPU_ID free memory: ${FREE} GiB"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
echo "[RUN] Starting sglang bench_one_batch..."
echo "  Command: LD_PRELOAD=$PROFILER_SO python -m sglang.bench_one_batch ..."
echo ""

START_TS=$(date +%s)

LD_PRELOAD="$PROFILER_SO" \
python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --batch "$BATCH" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --profile \
    --dtype float16 \
    --trust-remote-code \
    --device musa \
    1>"${OUTPUT_DIR}/sglang_stdout.log" \
    2>"${OUTPUT_DIR}/sglang_stderr.log"

EXIT_CODE=$?
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

# ── Results ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================================"
echo " Results (elapsed: ${ELAPSED}s, exit: $EXIT_CODE)"
echo "============================================================================"
echo ""

echo "── 1. sglang Latency Metrics ──"
grep -E "Prefill\. latency|Decode\.\s+median|Total\. latency" "${OUTPUT_DIR}/sglang_stdout.log" | head -6
echo ""

echo "── 2. API Profiler Report ──"
if [ -s "${OUTPUT_DIR}/api_profiler_report.txt" ]; then
    head -30 "${OUTPUT_DIR}/api_profiler_report.txt"
else
    echo "  (API profiler report empty — may conflict with torch.profiler MUPTI subscription)"
    echo "  Check ${OUTPUT_DIR}/sglang_stderr.log for profiler init messages"
fi
echo ""

echo "── 3. sglang Operator Dispatch (CPU) ──"
grep -A20 "Name.*Self CPU.*CPU total" "${OUTPUT_DIR}/sglang_stdout.log" | tail -20 | head -12
echo ""

echo "── 4. Chrome Trace Files ──"
ls -lh "${OUTPUT_DIR}/chrome_trace/"*.trace.json.gz 2>/dev/null || echo "  (none)"
echo ""

echo "── 5. All Output Files ──"
find "$OUTPUT_DIR" -type f -exec ls -lh {} \; 2>/dev/null
echo ""
echo "============================================================================"
echo " Done. All results in: $OUTPUT_DIR"
echo "============================================================================"

exit $EXIT_CODE
