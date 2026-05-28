#pragma once

#include "export_table.h"
#include "musa.h"
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace KernelProfiler {

struct KernelRecord {
    uint64_t    correlationId;
    std::string kernelName;
    dim3        gridSize;
    dim3        blockSize;
    uint64_t    dynamicSmem;
    uint64_t    queued_time;
    uint64_t    submit_time;
    bool        has_submit_time;
    uint64_t    gpu_begin;
    uint64_t    gpu_end;

    KernelRecord() : correlationId(0), dynamicSmem(0),
                     queued_time(0), submit_time(0), has_submit_time(false),
                     gpu_begin(0), gpu_end(0) {}
};

extern MUpti::DriverExportTable*  g_DriverExport;
extern Driver::ExportTable*       g_DriverApiTable;
extern std::atomic<double>        g_GpuTimestampPeriodNs;
extern std::unordered_map<uint64_t, KernelRecord*> g_Records;
extern std::mutex                                   g_Mutex;

MUpti::Context* OnRegisterKernelV2(Musa::DispatchCommand* command);
MUpti::Context* OnRegisterGraphKernel(Musa::GraphCommand* gcmd, Musa::GraphNode* node);
void OnMarkKernelQueued(uint64_t correlationId);
void OnMarkKernelSubmitted(uint64_t correlationId);
void OnMarkCommandBeginEnd(MUpti::Context* context, Musa::Command* command);
void OnMarkGraphNodeBeginEndV2(MUpti::Context* context, Musa::GraphNode* node, Musa::GraphExec* exec);
void OnAssignKernelToKick(uint64_t kernelId, uint64_t submissionId);

void PrintKernelReport(const KernelRecord& rec);
std::string TimestampStr();

inline uint64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

} // namespace KernelProfiler
