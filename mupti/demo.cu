#include <mupti.h>
#include <musa_runtime.h>
#include <malloc.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <iostream>

#define CHECK_MUSA_ERROR(call) do { \
    musaError_t err = call; \
    if (err != musaSuccess) { \
        fprintf(stderr, "MUSA error at %s:%d: %s\n", \
                __FILE__, __LINE__, musaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_MUPTI_ERROR(call) do { \
    MUptiResult err = call; \
    if (err != MUPTI_SUCCESS) { \
        const char *errStr; \
        muptiGetResultString(err, &errStr); \
        fprintf(stderr, "MUPTI error at %s:%d: %s\n", \
                __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

__global__ void axpy(float *x, float *y, float a) {
    y[threadIdx.x] = a * x[threadIdx.x];
}

const char* GetMemcpyKindString(MUpti_ActivityMemcpyKind kind) {
    switch (kind) {
    case MUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        return "HtoD";
    case MUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        return "DtoH";
    case MUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
        return "HtoA";
    case MUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
        return "AtoH";
    case MUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
        return "AtoA";
    case MUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
        return "AtoD";
    case MUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
        return "DtoA";
    case MUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        return "DtoD";
    case MUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        return "HtoH";
    default:
        break;
    }

    return "<unknown>";
}

void printActivity(MUpti_Activity *record) {
    switch (record->kind) {
    case MUPTI_ACTIVITY_KIND_MEMCPY: {
        MUpti_ActivityMemcpy4 *memcpy = (MUpti_ActivityMemcpy4 *)record;
        printf(
            "MEMCPY %s(%d->%d) [ %llu - %llu ] device %u, context %u, stream %u, "
            "size %llu, correlation %u, graphId: %u, graphNodeId: %lu\n",
            GetMemcpyKindString((MUpti_ActivityMemcpyKind)memcpy->copyKind),
            memcpy->srcKind, memcpy->dstKind, 
            (unsigned long long)(memcpy->start),
            (unsigned long long)(memcpy->end),
            memcpy->deviceId, memcpy->contextId, memcpy->streamId,
            (unsigned long long)memcpy->bytes, memcpy->correlationId,
            memcpy->graphId, memcpy->graphNodeId);
        break;
    }
    case MUPTI_ACTIVITY_KIND_MEMSET: {
        MUpti_ActivityMemset3 *memset = (MUpti_ActivityMemset3 *)record;
        printf(
            "MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream "
            "%u, correlation %u, graphId: %u, graphNodeId: %lu\n",
            memset->value,
            (unsigned long long)(memset->start),
            (unsigned long long)(memset->end),
            memset->deviceId, memset->contextId, memset->streamId,
            memset->correlationId, memset->graphId, memset->graphNodeId);
        break;
    }
    case MUPTI_ACTIVITY_KIND_KERNEL:
    case MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const char *kindString =
            (record->kind == MUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
        MUpti_ActivityKernel6 *kernel = (MUpti_ActivityKernel6 *)record;
        printf(
            "%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, "
            "correlation %u, queued: %lu, submitted: %lu, graphId: %u, graphNodeId: %lu\n",
            kindString, kernel->name,
            (unsigned long long)(kernel->start),
            (unsigned long long)(kernel->end),
            kernel->deviceId, kernel->contextId, kernel->streamId,
            kernel->correlationId, kernel->queued, kernel->submitted,
            kernel->graphId, kernel->graphNodeId);
        printf(
            "    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static "
            "%u, dynamic %u)\n",
            kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX,
            kernel->blockY, kernel->blockZ, kernel->staticSharedMemory,
            kernel->dynamicSharedMemory);
        break;
    }
    case MUPTI_ACTIVITY_KIND_DRIVER: {
        MUpti_ActivityAPI *api = (MUpti_ActivityAPI *)record;
        MUpti_CallbackDomain domain = MUPTI_CB_DOMAIN_DRIVER_API;
        char const* name;
        muptiGetCallbackName(domain, api->cbid, &name);
        printf(
            "DRIVER API %s [ %llu - %llu ] cbid=%u, process %u, thread %u, "
            "correlation %u\n",
            name,
            (unsigned long long)(api->start),
            (unsigned long long)(api->end),
            api->cbid, api->processId, api->threadId, api->correlationId);
        break;
    }
    case MUPTI_ACTIVITY_KIND_RUNTIME: {
        MUpti_ActivityAPI *api = (MUpti_ActivityAPI *)record;
        MUpti_CallbackDomain domain = MUPTI_CB_DOMAIN_RUNTIME_API;
        char const* name;
        muptiGetCallbackName(domain, api->cbid, &name);
        printf(
            "RUNTIME API %s [ %llu - %llu ] cbid=%u, process %u, thread %u, "
            "correlation %u, status: %d\n",
            name,
            (unsigned long long)(api->start),
            (unsigned long long)(api->end),
            api->cbid, api->processId, api->threadId, api->correlationId, api->returnValue);
        break;
    }
    case MUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
        MUpti_ActivitySynchronization *sync = (MUpti_ActivitySynchronization *)record;
        printf(
            "SYNC [ %llu - %llu ] context %u, stream %u, event %u, type: %u, correlation %u\n",
            (unsigned long long)(sync->start),
            (unsigned long long)(sync->end),
            sync->contextId, sync->streamId, sync->musaEventId, sync->type, sync->correlationId);
        break;
    }
    default:
        printf("  <unknown>\n");
        break;
    }
}

void bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *buffer = static_cast<uint8_t*>(aligned_alloc(8, 256 * 10));
    *size = malloc_usable_size(*buffer);
    *maxNumRecords = 0;
}

void bufferCompleted(MUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    MUptiResult status;
    if (validSize > 0) {
        MUpti_Activity* record = nullptr;
        do {
            status = muptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == MUPTI_SUCCESS) {
                printActivity(record);
            } else if (status == MUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                exit(1);
            }
        } while (true);

        size_t dropped;
        CHECK_MUPTI_ERROR(muptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
    }

    free(buffer);
}

void initTrace() {
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_DRIVER));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_MEMSET));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_KERNEL));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK_MUPTI_ERROR(muptiActivityEnable(MUPTI_ACTIVITY_KIND_SYNCHRONIZATION));

    CHECK_MUPTI_ERROR(muptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void finiTrace() {
    CHECK_MUPTI_ERROR(muptiActivityFlushAll(1));

    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_DRIVER));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_MEMSET));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_KERNEL));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK_MUPTI_ERROR(muptiActivityDisable(MUPTI_ACTIVITY_KIND_SYNCHRONIZATION));

    CHECK_MUPTI_ERROR(muptiFinalize());
}

// 全局变量用于控制刷新线程
std::atomic<bool> flush_thread_active(true);

// 刷新函数，在单独的线程中运行
void periodic_flush() {
    while(flush_thread_active.load()) {
        muptiActivityFlushAll(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
}

int main(int argc, char *argv[]) {
    // build: mcc -o axpy_trace axpy_trace.cu -mtgpu -lmusart -lmupti
    initTrace();

    // 启动定期刷新线程
    std::thread flush_thread(periodic_flush);

    float a = 2.0f;
    const int kDataLen = 4;
    float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_y[kDataLen];

    // Copy input data to device.
    float *device_x;
    float *device_y;
    CHECK_MUSA_ERROR(musaMalloc(&device_x, kDataLen * sizeof(float)));
    CHECK_MUSA_ERROR(musaMalloc(&device_y, kDataLen * sizeof(float)));
    CHECK_MUSA_ERROR(musaMemcpy(device_x, host_x, kDataLen * sizeof(float), musaMemcpyHostToDevice));

    musaStream_t streams[10];
    for (int i = 0; i < 10; i++) {
        CHECK_MUSA_ERROR(musaStreamCreate(&(streams[i])));
    }

    void* args[] = { &device_x, &device_y, &a };
    // Launch the kernel.
    for (int i = 0; i < 10; i++) {
        CHECK_MUSA_ERROR(musaLaunchKernel((void*)axpy, dim3(1), dim3(kDataLen), args, 0, streams[i]));
    }

    // Copy output data to host.
    CHECK_MUSA_ERROR(musaDeviceSynchronize());

    CHECK_MUSA_ERROR(musaMemcpy(host_y, device_y, kDataLen * sizeof(float), musaMemcpyDeviceToHost));

    // Print the results.
    for (int i = 0; i < kDataLen; ++i) {
        std::cout << "y[" << i << "] = " << host_y[i] << "\n";
    }

    CHECK_MUSA_ERROR(musaFree(device_x));
    CHECK_MUSA_ERROR(musaFree(device_y));
    CHECK_MUSA_ERROR(musaDeviceReset());

    // 停止并等待刷新线程结束
    flush_thread_active = false;
    flush_thread.join();

    finiTrace();

    return 0;
}