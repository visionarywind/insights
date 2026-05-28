# ============================================================================
# Kernel Profiler Library — libmusaKernelProfiler.so
# ============================================================================
# 用途: 注入到 musa 驱动中, 通过 MUPTI Driver Hooks 精确 profile GPU kernel 耗时
# 加载: export MUSA_INJECTION64_PATH=/path/to/libmusaKernelProfiler.so
#
# 输出: stdout — 每个 kernel 完成后输出一行:
#   [PROFILER] 2026-05-19 14:32:01 | kernelName | Grid(128,1,1) Block(256,1,1)
#   [PROFILER]   CPU调度: 12.34us | GPU队列: 5.67us | GPU执行: 123.45us | 端到端: 141.46us
# ============================================================================

**Sisyphus**:
CXX = g++ -std=c++17 -fPIC -O2 -Wall -Wextra -Wno-unused-parameter
MUSA_ROOT = /home/shanfeng/linux-ddk/musa
INCLUDES = -I$(MUSA_ROOT)/src/musa_shared_include/ -I$(MUSA_ROOT)/src/musa_shared_include/mupti/

all: libmusaKernelProfiler.so

libmusaKernelProfiler.so: kernel_profiler.cpp kernel_profiler.h
	$(CXX) $(INCLUDES) -shared -o $@ kernel_profiler.cpp -ldl -lpthread

clean:
	rm -f libmusaKernelProfiler.so
