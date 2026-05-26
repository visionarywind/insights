#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "musa.h"

static void print_status(const char* step, MUresult status) {
    std::printf("%-28s status=%d\n", step, static_cast<int>(status));
}

static bool require_success(const char* step, MUresult status) {
    print_status(step, status);
    return status == MUSA_SUCCESS;
}

int main() {
    MUresult status = MUSA_SUCCESS;
    MUdevice device = 0;
    MUcontext context = nullptr;
    MUstream stream = nullptr;

    if (!require_success("muInit", muInit(0))) {
        return 1;
    }
    if (!require_success("muDeviceGet", muDeviceGet(&device, 0))) {
        return 1;
    }
    if (!require_success("muCtxCreate", muCtxCreate(&context, 0, device))) {
        return 1;
    }
    if (!require_success("muStreamCreate", muStreamCreate(&stream, 0))) {
        return 1;
    }

    MUdeviceptr dptr = 0;
    if (!require_success("muMemAlloc", muMemAlloc(&dptr, 4096))) {
        return 1;
    }

    // 同步 memset 路径：
    // driver API -> Context::GeneralMemset -> Stream::CmdMemset -> MemsetCommand::Submit。
    status = muMemsetD32(dptr, 0x3f800000u, 4);
    print_status("muMemsetD32", status);

    // 同步 HtoD copy 路径：
    // driver API -> Context::GeneralMemcpy -> Stream::CmdCopyMemory -> Sync/AsyncMemcpyCommand。
    uint32_t host_values[4] = {1, 2, 3, 4};
    status = muMemcpyHtoD(dptr, host_values, sizeof(host_values));
    print_status("muMemcpyHtoD", status);

    // 指针属性查询路径：
    // muPointerGetAttributes 会逐个调用内部 imuapiPointerGetAttribute。
    MUmemorytype memory_type = MU_MEMORYTYPE_UNIFIED;
    MUdeviceptr device_pointer = 0;
    void* range_start = nullptr;
    size_t range_size = 0;
    int device_ordinal = -1;
    MUpointer_attribute attrs[] = {
        MU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        MU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        MU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        MU_POINTER_ATTRIBUTE_RANGE_SIZE,
        MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    };
    void* values[] = {
        &memory_type,
        &device_pointer,
        &range_start,
        &range_size,
        &device_ordinal,
    };
    status = muPointerGetAttributes(
        static_cast<unsigned int>(sizeof(attrs) / sizeof(attrs[0])), attrs, values, dptr);
    print_status("muPointerGetAttributes", status);
    std::printf("  memory_type=%d device_pointer=%#llx range_start=%p range_size=%zu device_ordinal=%d\n",
                static_cast<int>(memory_type),
                static_cast<unsigned long long>(device_pointer),
                range_start,
                range_size,
                device_ordinal);

    // 当前源码中的 prefetch 是参数校验路径，不创建 PrefetchCommand，也不向 stream 入队。
    status = muMemPrefetchAsync(dptr, 1024, device, stream);
    print_status("muMemPrefetchAsync", status);

    void* pinned_host = nullptr;
    status = muMemHostAlloc(&pinned_host, 4096, 0);
    print_status("muMemHostAlloc", status);
    if (status == MUSA_SUCCESS) {
        status = muMemFreeHost(pinned_host);
        print_status("muMemFreeHost", status);
    }

    void* registered_host = nullptr;
    if (posix_memalign(&registered_host, 4096, 4096) == 0) {
        status = muMemHostRegister(registered_host, 4096, 0);
        print_status("muMemHostRegister", status);
        if (status == MUSA_SUCCESS) {
            status = muMemHostUnregister(registered_host);
            print_status("muMemHostUnregister", status);
        }
        std::free(registered_host);
    } else {
        std::printf("%-28s status=posix_memalign_failed\n", "posix_memalign");
    }

    // 异步池化分配路径：
    // muMemAllocAsync -> Stream::CmdMemAlloc -> Stream::AsyncMemAlloc
    // -> MemoryPool::CreateMemory -> physical->Init -> MemoryPool::ModifyAccess
    // -> Stream::CmdPaging -> PagingCommand::Submit。
    MUdeviceptr async_ptr = 0;
    status = muMemAllocAsync(&async_ptr, 4096, stream);
    print_status("muMemAllocAsync", status);
    if (status == MUSA_SUCCESS) {
        status = muStreamSynchronize(stream);
        print_status("muStreamSynchronize alloc", status);

        // 异步释放路径：
        // muMemFreeAsync -> Stream::CmdMemFree -> Stream::AsyncMemFree
        // -> MemoryPool::DisableAccess -> Stream::CmdPaging
        // -> CallbackCommand::Submit -> DestroyPhysMemories/DestroyMemory。
        status = muMemFreeAsync(async_ptr, stream);
        print_status("muMemFreeAsync", status);
        status = muStreamSynchronize(stream);
        print_status("muStreamSynchronize free", status);
    }

    status = muMemFree(dptr);
    print_status("muMemFree", status);
    status = muStreamDestroy(stream);
    print_status("muStreamDestroy", status);
    status = muCtxDestroy(context);
    print_status("muCtxDestroy", status);
    return 0;
}
