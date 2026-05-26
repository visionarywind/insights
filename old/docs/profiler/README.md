# MUSA API Profiler

Zero-code-change profiler for MUSA GPU driver and runtime APIs.
Injects via `MUSA_INJECTION64_PATH` into ANY process using MUSA — no recompilation needed.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Application (sglang / torch / custom C)    │
├─────────────────────────────────────────────┤
│  MUSA Runtime API (musaXXX)   ← ToolsCallback RUNTIME_API domain │
├─────────────────────────────────────────────┤
│  MUSA Driver API (muXXX)      ← ToolsCallback DRIVER_API domain  │
├─────────────────────────────────────────────┤
│  KMD (kernel driver)                        │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build
make profiler

# Test (inside container with MUSA driver)
make test1    # Driver API (mu*) profiling
make test2    # Runtime API (musa*) profiling
make test3    # Overlay (cross-layer) statistics
make test4    # sglang integration
```

## Usage

```bash
# Inject into any MUSA process
export MUSA_INJECTION64_PATH=/path/to/liblatency_profiler.so

# Optional: configure output
export PROFILER_OUTPUT=/tmp/report.txt    # write report to file (default: stderr)
export PROFILER_TOP_N=30                 # limit report rows (default: 60)
export PROFILER_DEBUG=1                  # enable debug logging
export PROFILER_DRIVER=1                 # enable driver API capture (default: 1)
export PROFILER_RUNTIME=1                # enable runtime API capture (default: 1)
export PROFILER_USE_MUPTI=0             # disable public MUPTI path (default: 0)

# Run your application
python -m sglang.bench_one_batch --model-path ... --batch 1 ...
```

## Report Format

```
driver:muXXX            ← Driver API calls (mu* prefix)
runtime:musaXXX         ← Runtime API calls (musa* prefix)
```

Report columns: Name, Count, Total(us), Self(us), Min(us), Max(us).

Wrapper APIs (Self < 10% of Total) are automatically marked — their time is mostly in child calls.

## File Layout

```
profiler/
├── api_latency_profiler.cpp    # Main profiler source (v6)
├── cbid_names_driver.inc        # Driver CBID → name table (811 entries)
├── cbid_names_runtime.inc       # Runtime CBID → name table (519 entries)
├── gen_cbid_names.py            # CBID name table generator
├── Makefile                     # Build & test targets
├── README.md
└── tests/
    ├── test_driver_api.c        # Test 1: mu* profiling
    ├── test_runtime_api.c       # Test 2: musa* profiling
    ├── test_overlay.c           # Test 3: cross-layer overlay
    └── test_sglang.sh           # Test 4: sglang integration
```
