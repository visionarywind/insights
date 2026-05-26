# MUSA API Profiler — Build & Test
# ==================================
#
# Requirements:
#   - MUSA driver installed (/dev/mtgpu*)
#   - libmusa.so.1 available
#   - MUSA SDK headers at ../musa_shared_include/
#
# Build profiler:
#   make profiler
#
# Run tests (in container with MUSA driver):
#   make test1    # Driver API profiling
#   make test2    # Runtime API profiling
#   make test3    # Overlay statistics
#   make test4    # sglang integration

MUSA_INC  ?= /home/shanfeng/workspace/MUSA-Runtime/musa_shared_include
MUSA_SDK  ?= /usr/local/musa-5.1.0/include
CXX       ?= g++
CXXFLAGS  := -shared -fPIC -O2
LDFLAGS   := -ldl -lpthread -lstdc++
TARGET    := liblatency_profiler.so

.PHONY: all profiler clean test1 test2 test3 test4 help

all: profiler

profiler: $(TARGET)

$(TARGET): api_latency_profiler.cpp cbid_names_driver.inc cbid_names_runtime.inc
	$(CXX) $(CXXFLAGS) -o $@ api_latency_profiler.cpp \
		-I$(MUSA_INC) -I$(MUSA_SDK) -I. $(LDFLAGS)
	@echo "[OK] $(TARGET) built"

clean:
	rm -f $(TARGET) tests/*.o tests/test_*

# ── Test targets (run inside container with MUSA driver) ─────────────────────

test1: profiler tests/test_driver_api
	@echo "=== Test 1: Driver API (muXXX) Profiling ==="
	MUSA_INJECTION64_PATH=$(CURDIR)/$(TARGET) tests/test_driver_api

test2: profiler tests/test_runtime_api
	@echo "=== Test 2: Runtime API (musaXXX) Profiling ==="
	MUSA_INJECTION64_PATH=$(CURDIR)/$(TARGET) tests/test_runtime_api

test3: profiler tests/test_overlay
	@echo "=== Test 3: Overlay Statistics ==="
	MUSA_INJECTION64_PATH=$(CURDIR)/$(TARGET) tests/test_overlay

test4:
	@echo "=== Test 4: sglang Integration ==="
	bash tests/test_sglang.sh

tests/test_driver_api: tests/test_driver_api.c
	gcc -o $@ $< -ldl

tests/test_runtime_api: tests/test_runtime_api.c
	gcc -o $@ $< -ldl

tests/test_overlay: tests/test_overlay.c
	gcc -o $@ $< -ldl

help:
	@echo "Targets:"
	@echo "  profiler  — build liblatency_profiler.so"
	@echo "  test1     — Driver API muXXX profiling"
	@echo "  test2     — Runtime API musaXXX profiling"
	@echo "  test3     — Overlay (cross-layer) statistics"
	@echo "  test4     — sglang integration"
	@echo "  clean     — remove build artifacts"
