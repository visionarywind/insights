# MUSA GPU Driver Kick 机制分析

> **Kick = 通知 GPU 硬件"有新工作"的最终信号**。本质是一次 MMIO 寄存器写操作。
>
> 分析基于源码，覆盖从用户态 API 到硬件寄存器写入的完整调用链。

---

## 1. 什么是 Kick？

Kick 是 GPU 驱动中**通知固件（firmware）有新命令需要处理**的硬件信号。它的本质是一次对 MMIO 寄存器（MHI region）的写入操作。固件轮询或中断响应这个信号后，会从 CCB（Command Circular Buffer）环形缓冲区中读取新的命令提交。

```
用户态                              内核态                          GPU 固件
─────                              ─────                          ────────
                                   ┌──────────┐
API 调用 → 命令缓冲区 → Kick ──────▶│  CCB 环   │──▶ 固件读取命令
                                   │ (环形队列)│
                                   └──────────┘
                                          ↑
                                Kick 更新 wrIdx
```

Kick 的**本质作用**是更新 CCB 的写指针（wrIdx），并通过写 doorbell 寄存器通知 GPU 固件"有新条目待处理"。

---

## 2. 三种 Kick 路径

MUSA 驱动支持三种 kick 路径，按性能从高到低排列：

| 路径 | 延迟 | 适用场景 | 开销来源 |
|------|------|----------|----------|
| **Doorbell 路径** | 极低 | 现代芯片（PH1+），CDM/CE/ACE 队列 | 用户态直接 MMIO 写，无 ioctl |
| **IOCTL 路径** | 中等 | 传统路径，无 doorbell 支持 | DRM ioctl 系统调用 |
| **Graph Kick 路径** | 低 | CUDA Graph 执行 | 批量 kick，引擎路由 |

### 2.1 Doorbell 路径（快速路径）

**适用条件**：`EnableUserQueue() && EnableDoorbell()` 均为 true，且 GPU 支持 doorbell 功能（PH1+）。

**Doorbell 生命周期**：

```
阶段 1: 获取 Doorbell（队列初始化时）
  Stream::Init()
    → queueCreateInfo.doorbell = userQueue && EnableDoorbell()   [stream.cpp:905]
    → HalDevice.CreateQueue(queueCreateInfo)
      → Queue::Init()
        → Device::AcquireDoorbell()                              [mtgpuQueue.cpp:267]
          → mtgpu_job_doorbell_acquire()                         [libdrm: mtgpu_job.c:451]
            → mmap() 匿名页 → DRM_MTGPU_JOB_ACQUIRE_DOORBELL ioctl → 内核
              → doorbell_acquire()  [doorbell.c:41]
                → 从 bitmap pool 分配 doorbell ID
                → rm_doorbell_map_to_user() — 将 MHI 寄存器 PA remap 到用户 VMA
            → 用户态 doorbell_reg = mmap_base + addr_offset

阶段 2A: 首次提交（带命令数据）
  Queue::LaunchCommandStreams()
    → Device::SubmitCommandsWithDoorbell(hContext, ..., hDoorbell)  [mtgpuQueue.cpp:561]
      → mtgpu_job_submit_with_doorbell()  [libdrm: mtgpu_job.c]
        → DRM_MTGPU_JOB_SUBMIT_WITH_DOORBELL ioctl
          → 内核：mtgpu_job_submit_with_doorbell_ioctl()  [mtgpu_job_v3.c:2589]
            → 构建 CCB item（带 doorbellId + withDoorbell=1）
            → 固件收到提交 + doorbell 信号，开始监控该 doorbell

阶段 2B: 后续提交（仅 ring doorbell）
  Queue::LaunchCommandStreams()
    → Device::RingDoorbell(hContext, hDoorbell)                   [mtgpuQueue.cpp:620]
      → mtgpu_job_doorbell_ring()  [libdrm: mtgpu_job.c:555]
        → *doorbell->doorbell_reg = 1 << (doorbell->doorbell_id % 32);
                                     ↑
                               直接用户态 MMIO 写！
```

**关键代码（libdrm-mt/mtgpu/mtgpu_job.c:568）**：
```c
*doorbell->doorbell_reg = 1 << (doorbell->doorbell_id % MTGPU_DOORBELL_REG_WIDTH);
```

这是整个 kick 路径中最快的一条——**绕过内核，用户态直接写硬件寄存器**。

### 2.2 IOCTL 路径（传统路径）

当不支持 doorbell 时（`hDoorbell == 0`），走传统 ioctl 路径：

```
Stream::AsyncSubmit()
  → HalQueue::Submit()
    → M3D::Queue::SubmitInternal()
      → Queue::LaunchCommandStreams()
        → Device::SubmitCommandsV3(...)                         [mtgpuQueue.cpp:589]
          → mtgpu_job_submit()  [libdrm: mtgpu_job.c]
            → DRM_IOCTL_MTGPU_CMD (SUBMIT_V3)
              ┌─────────────────────────────────────────────────────────────┐
              │ 内核态                                                       │
              │                                                             │
              │ mtgpu_ioctl()                       [mtgpu_ioctl.c:60]      │
              │   → mtgpu_job_submit_ioctl_v3()     [mtgpu_job_v3.c:1931]   │
              │     → 映射 cmd_type: COMPUTE→CDM, UNIVERSAL→UQ, CE→CE, ... │
              │     │                                                        │
              │     ├── Path A: Host Scheduler (drm_sched)                  │
              │     │   → mtgpu_sched_job_create_and_push()                 │
              │     │     → Scheduler 线程取出 job                           │
              │     │       → mtgpu_sched_job_run()  [mtgpu_sched.c:445]    │
              │     │         → rm_queue_ccb_submit() 或 mtgpu_fw_cmd_submit│
              │     │                                                        │
              │     └── Path B: FW-only (直接提交)                           │
              │         → mtgpu_fw_cmd_submit()       [mtgpu_fw_job.c:452]  │
              │           → mtgpu_fw_get_ccb_woff() — 获取 CCB 写位置       │
              │           → 写入 MTFW_CCB_ITEM (submission_va, size, ...)   │
              │           → os_wmb() — 内存屏障                              │
              │           → 更新 ccb_ctrl->wrIdx                             │
              │           → mtgpu_fw_kick(dev_node, queue_index)            │
              │             → MMIO 寄存器写 (见 §3)                          │
              └─────────────────────────────────────────────────────────────┘
```

### 2.3 Graph Kick 路径

CUDA Graph 使用独立的 kick 结构：

**关键文件**：`musa/src/musa/core/graph/graph1/graphExec.h/cpp`

```cpp
// MusaKick 结构体（graphExec.h）
struct MusaKick {
    KickType kickType;   // CDM, ACE, CE
    // ...
};

// Kick 类型分发
bool IsKickTypeCdm(MusaKick& kick);
bool IsKickTypeAce(MusaKick& kick);
bool IsKickTypeCe(MusaKick& kick);

// 索引更新
uint32_t GetCdmIndexAndUpdate(MusaKick& kick);
uint32_t GetCeIndexAndUpdate(MusaKick& kick);
uint32_t GetAceIndexAndUpdate(MusaKick& kick);

// 提交
result = AddKickToSubmission(MusaKick& kick, ...);
```

Graph 执行时，多个 node 的 kick 被批量收集，按引擎类型分组后统一提交。

---

## 3. 内核态 Kick 实现（硬件寄存器写）

内核态 kick 的入口是 `mtgpu_fw_kick()`，根据固件类型分发到不同硬件后端：

```c
// mtgpu_fw.c:61
void mtgpu_fw_kick(struct mtgpu_device_node *dev_node, u32 queue_index) {
    if (RM_FEATURE_IS_ENABLED(FW))
        dev_node->rm_fw_kick(dev_node->rm_fw_priv, queue_index);  // RM/Rust 路径
    else
        dev_node->fw_ops->kick(dev_node, queue_index);            // 传统路径
}
```

### 3.1 RM 路径（新架构）

RM（Resource Manager）启用时，kick 回调由 Rust RM 在初始化时注册：

```
mtgpu_rm_init()
  → RM_0002_CTRL_CMD_FW_GET_CALLBACK
    → dev_node->rm_fw_kick = rm_kick_callback  [mtgpu_rm.c:126]
```

### 3.2 传统路径：三种固件后端

| 固件类型 | 芯片 | Kick 函数 | 寄存器 |
|----------|------|-----------|--------|
| **MCU** | QY1, QY2 | `mcu_fw_kick()` | `MUSA_CR_MTS_SCHEDULE` |
| **RISC-V (HG+)** | HG, HS, LS | `riscv_fw_kick()` | `MT_CR_FEC_FTS_CORE_OS0_SCHEDULE_FTS_0` |
| **FEC** | 旧版 RISC-V | `mtgpu_fec_kick()` | softirq `MTGPU_SOFTIRQ_TO_FEC_1` |

#### MCU Kick（`mtgpu_meta.c:788`）
```c
static void mcu_fw_kick(struct mtgpu_device_node *dev_node, u32 type) {
    MCU_WRITE_REG32(MUSA_CR_MTS_SCHEDULE, type);
    // type = MTFW_FWIF_QUEUE_TYPE_CDM (3), CE (4), etc.
}
```

#### RISC-V Kick（`mtgpu_riscv.c:286`）
```c
static void riscv_fw_kick(struct mtgpu_device_node *dev_node, u32 type) {
    void *reg_base = mtdev->fec_fts_region.registers;
    uint32_t reg_val = ccb_to_schedule_val_tbl[type];
    // 映射表: GP→EVENT_0, UQ→EVENT_2, CDM→EVENT_3, CE→EVENT_4

    if (DEVICE_IS_HG_OR_LATER(mtdev))
        os_iowrite32(reg_val, reg_base + MT_CR_FEC_FTS_CORE_OS0_SCHEDULE_FTS_0);
    else
        mtgpu_fec_kick(dev);  // 旧版走 softirq
}
```

#### FEC Kick（`mtgpu/mtgpu_fec.c:20`）
```c
void mtgpu_fec_kick(struct device *dev) {
    mtgpu_raise_softirq(MTGPU_SOFTIRQ_TO_FEC_1);
    // softirq 触发 → mtgpu_fw_irq_work_func() → 处理 CCB
}
```

---

## 4. Doorbell 机制深入

### 4.1 Doorbell 是什么？

Doorbell 是 MHI（Message Host Interface）寄存器空间中的一个 bit。每个 doorbell 就是一个独立的通知通道。固件监控特定 doorbell bit，当该 bit 被置 1 时，固件知道对应队列有新工作。

### 4.2 Doorbell 寄存器地址计算

**文件**：`gr-kmd/mt-rm/modules/device/doorbell/doorbell.c:23`

```
物理地址 = IO region base + offset + (index / width) × stride

MCU (meta fw):
  phys_addr = IO_IDX_META_MHI.start + mcu_mhi_set_reg_offset
            + (index / mcu_mhi_set_reg_width) × mcu_mhi_reg_gruop_stride

FEC (riscv fw):
  phys_addr = IO_IDX_FEC_MHI.start + fec_mhi_set_reg_offset
            + (index / fec_mhi_set_reg_width) × fec_mhi_reg_gruop_stride
```

**LS/HS 芯片参数**（`doorbell_mgr_ls.c`）：
```
fec_mhi_set_reg_offset    = 0x104
fec_mhi_reg_gruop_count   = 0x40
fec_mhi_reg_gruop_stride  = 0x4
fec_mhi_set_reg_width     = 0x20  (32 bits)
```

每个 32-bit 寄存器包含 32 个 doorbell bit。Doorbell #N 对应 bit `N % 32`。

### 4.3 Doorbell 分配与管理

**文件**：`gr-kmd/mt-rm/modules/device/doorbell/doorbell.c:41`

```c
static int doorbell_acquire(struct gpu_device *gpu, uint64_t user_va,
                            struct doorbell *doorbell, uint32_t *addr_offset,
                            uint64_t *phys_addr) {
    // 1. 从 bitmap pool 中分配一个 free doorbell index
    index = rm_os_find_first_zero_bit(mgr->pool_bitmap, mgr->pool_size);

    // 2. 将 MHI 寄存器页 remap 到用户态 VMA
    rm_doorbell_map_to_user(gpu, user_va, index, addr_offset);
    //   → os_remap_pfn_range() — 用户态可直接访问物理寄存器

    // 3. 返回 doorbell ID 和物理地址偏移
    doorbell->id = index;
    *addr_offset = (index % reg_width) * 4;
    *phys_addr = doorbell_get_phys_addr(gpu, index);
}
```

### 4.4 用户态 Doorbell 结构

**文件**：`libdrm-mt/mtgpu/mtgpu_internal.h:117`

```c
struct mtgpu_doorbell {
    volatile uint32_t *doorbell_reg;  // 映射到 MHI 寄存器的用户态指针
    uint32_t doorbell_id;             // doorbell 编号
    uint64_t doorbell_addr_offset;    // 页内偏移
};
```

---

## 5. CCB 环形缓冲区

CCB（Command Circular Buffer）是 Host 与 GPU 固件之间的命令通信环形队列。

### 5.1 CCB 控制结构

**文件**：`shared_include/fwif/mtfw_fwif_types.h:72`

```c
typedef struct {
    uint64_t wrIdx;       // Host 写指针（驱动更新）
    uint64_t rdIdx;       // 固件读指针（GPU 固件更新）
    // ... 其他控制字段
} MTFW_RING_CTRL;
```

### 5.2 CCB 条目（提交项）

**文件**：`shared_include/fwif/mtfw_fwif_ccb.h:33`

```c
typedef struct {
    uint32_t withDoorbell;     // 是否使用 doorbell 通知
    uint32_t doorbellId;       // doorbell ID（若 withDoorbell=1）
    uint64_t submission_va;    // GPU VA of command buffer
    uint32_t submission_size;  // 命令缓冲区大小
    // ... page table info, semaphores, fences
} MTFW_SUBMISSION_PARA;
```

### 5.3 CCB 提交流程

```
mtgpu_fw_cmd_submit()                           [mtgpu_fw_job.c:452]
  1. mtgpu_fw_get_ccb_woff() — 获取下一个可用 CCB slot
  2. 写入 MTFW_CCB_ITEM（submission_va, size, page_table, semaphores,
                          withDoorbell, doorbellId）
  3. 写入 item 地址到 reqCcbQ cell
  4. os_wmb() — 确保数据对固件可见
  5. 更新 ccb_ctrl->wrIdx
  6. os_wmb() + os_ioread32() — 确保更新对固件可见
  7. mtgpu_fw_kick(dev_node, queue_index) — 通知固件
```

---

## 6. 引擎类型与队列映射

GPU 硬件有多个独立的命令处理引擎，每个有独立的 CCB 队列和 doorbell：

| 引擎 | 队列类型常量 | 值 | 用途 |
|------|-------------|-----|------|
| GP | `MTFW_FWIF_QUEUE_TYPE_GP` | 0 | 通用计算（旧） |
| UQ | `MTFW_FWIF_QUEUE_TYPE_UQ` | 1 | Universal Queue（图形） |
| TDM | `MTFW_FWIF_QUEUE_TYPE_TDM` | 2 | Tile Distribution Manager |
| CDM | `MTFW_FWIF_QUEUE_TYPE_CDM` | 3 | Compute Dispatch Manager |
| CE | `MTFW_FWIF_QUEUE_TYPE_CE` | 4 | Copy Engine（DMA 拷贝） |
| KMD_CE | `MTFW_FWIF_QUEUE_TYPE_KMD_CE` | 5 | KMD 内部拷贝引擎 |
| ACE | `MTFW_FWIF_QUEUE_TYPE_ACE` | 6 | Async Compute Engine |

**引擎选择路径**：

```
API 调用
  → Stream::AsyncSubmit()
    → GetHalEngine() — 确定引擎类型（CDM/CE/TDM/UQ）
      → HalQueue::Submit(engine)
        → M3D Queue Submit → CCB → fw_kick(queue_type)
```

---

## 7. 完整 Kick 调用链速查

```
用户态 API
  │
  ├── Doorbell 路径（快速）
  │   └─ musaLaunchKernel / musaMemcpyAsync / ...
  │       → Stream::AsyncSubmit()                              [stream.cpp]
  │         → HalQueue::Submit()
  │           → M3D Queue::SubmitInternal()
  │             → CmdBuffer::End()                             [cmdBuffer.cpp:生成 kick]
  │               → Queue::LaunchCommandStreams()              [mtgpuQueue.cpp:443]
  │                 ├─ 首次: SubmitCommandsWithDoorbell()      [→ KMD ioctl → CCB]
  │                 └─ 后续: RingDoorbell()                    [→ 用户态 MMIO 写]
  │                     → *doorbell_reg = 1 << (id % 32)      [mtgpu_job.c:568]
  │
  ├── IOCTL 路径（传统）
  │   └─ 同上至 Queue::LaunchCommandStreams()
  │       → SubmitCommandsV3()                                 [mtgpuQueue.cpp:589]
  │         → DRM_IOCTL_MTGPU_CMD                              [ioctl 系统调用]
  │           → mtgpu_ioctl()                                  [mtgpu_ioctl.c:60]
  │             → mtgpu_job_submit_ioctl_v3()                  [mtgpu_job_v3.c:1931]
  │               → mtgpu_fw_cmd_submit()                      [mtgpu_fw_job.c:452]
  │                 → 写 CCB, 更新 wrIdx
  │                   → mtgpu_fw_kick()                        [mtgpu_fw.c:61]
  │                     ├─ RM: dev_node->rm_fw_kick()
  │                     ├─ MCU: MCU_WRITE_REG32(MUSA_CR_MTS_SCHEDULE)
  │                     ├─ RISC-V: os_iowrite32(FTS_SCHEDULE_FTS_0)
  │                     └─ FEC: mtgpu_raise_softirq()
  │
  └── Graph Kick 路径
      └─ muGraphLaunch / muGraphInstantiate
          → GraphExec::Execute()
            → AddKickToSubmission(MusaKick)                    [graphExec.cpp]
              → 按 KickType 分发(CDM/ACE/CE)
                → CmdAcquire/CmdRelease
                  → 批量提交到对应引擎
```

---

## 8. 性能特征

| 路径 | Kick 延迟 | 关键开销 | 适用条件 |
|------|----------|----------|----------|
| Doorbell ring（追加） | ~100ns | 单次用户态 MMIO 写 | PH1+，CDM/CE/ACE，userQueue |
| Doorbell submit（首次） | ~1μs | ioctl + CCB 写入 | 同上 |
| IOCTL 提交 | ~2-5μs | ioctl 系统调用 | 所有平台 |
| Graph kick | ~100ns/node | 批量提交，单次 kick | Graph 模式 |

**关键优化点**：

1. **Doorbell 路径消除 ioctl 开销**：首次提交走 ioctl 建立 CCB 条目，后续提交仅 ring doorbell
2. **批量提交**：多个命令 buffer 可以在一次 CCB 写入中提交，配一次 kick
3. **Graph 模式**：多个 graph node 的 kick 合并提交，仅最后做一次 kick

---

## 9. 相关源文件索引

### 用户态 Kick 路径
| 文件 | 关键函数 | 行 |
|------|----------|-----|
| `musa/src/musa/core/stream.cpp` | `AsyncSubmit()`, doorbell enable | 895-920 |
| `musa/src/musa/core/stream.h` | `EnableDoorbell()` | 129 |
| `musa/src/hal/m3d/queue.cpp` | `Hal::M3d::Queue::Submit()` | — |
| `musa/src/hal/m3d/m3d/src/core/queue.cpp` | `SubmitInternal()`, `OsSubmit()` | — |
| `musa/src/hal/m3d/m3d/src/core/cmdBuffer.cpp` | `End()` — 生成 kick 信息 | — |
| `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuQueue.cpp` | `LaunchCommandStreams()` | 443-626 |
| `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuDevice.cpp` | `AcquireDoorbell()`, `RingDoorbell()`, `SubmitCommandsWithDoorbell()` | 1828-1869, 6124-6157 |
| `musa/src/musa/core/graph/graph1/graphExec.cpp` | `AddKickToSubmission()` | — |

### libdrm-mt 接口层
| 文件 | 关键函数 | 行 |
|------|----------|-----|
| `libdrm-mt/mtgpu/mtgpu_job.c` | `mtgpu_job_doorbell_acquire()` | 439-510 |
| 同上 | `mtgpu_job_submit_with_doorbell()` | 512-553 |
| 同上 | `mtgpu_job_doorbell_ring()` — **实际 MMIO 写** | 555-571 |
| `libdrm-mt/mtgpu/mtgpu_internal.h` | `struct mtgpu_doorbell` | 117-123 |

### 内核态 Kick 路径
| 文件 | 关键函数 | 行 |
|------|----------|-----|
| `gr-kmd/mtgpu-next/mtgpu_ioctl.c` | `mtgpu_ioctl()` — ioctl 分发 | 60 |
| `gr-kmd/mtgpu-next/mtgpu_job_v3.c` | `mtgpu_job_submit_ioctl_v3()` | 1931 |
| 同上 | `mtgpu_job_submit_with_doorbell_ioctl()` | 2589 |
| 同上 | `mtgpu_job_acquire_doorbell_ioctl()` | 2964 |
| `gr-kmd/mtgpu-next/mtgpu_fw_job.c` | `mtgpu_fw_cmd_submit()` — CCB 写入 | 452 |
| `gr-kmd/mtgpu-next/mtgpu_fw.c` | `mtgpu_fw_kick()` — kick 分发 | 61 |
| `gr-kmd/mtgpu-next/mtgpu_riscv.c` | `riscv_fw_kick()` — RISC-V kick | 286 |
| `gr-kmd/mtgpu-next/mtgpu_meta.c` | `mcu_fw_kick()` — MCU kick | 788 |
| `gr-kmd/mtgpu/mtgpu_fec.c` | `mtgpu_fec_kick()` — FEC kick (softirq) | 20 |
| `gr-kmd/mtgpu-next/scheduler/mtgpu_sched.c` | `mtgpu_sched_job_run()` | 445 |

### Doorbell 硬件层
| 文件 | 关键函数 | 行 |
|------|----------|-----|
| `gr-kmd/mt-rm/modules/device/doorbell/doorbell.c` | `doorbell_get_phys_addr()` | 23 |
| 同上 | `doorbell_acquire()` — bitmap 分配 + remap | 41 |
| `gr-kmd/mt-rm/modules/device/doorbell/doorbell_mgr.c` | HAL 分发表 | 22-27 |
| `gr-kmd/mt-rm/modules/device/doorbell/doorbell_mgr_ls.c` | LS/HS 寄存器参数 | 15-30 |
| `gr-kmd/mt-rm/os/linux/src/rm_os_doorbell.c` | `rm_doorbell_map_to_user()` | 18-60 |

### 固件接口定义
| 文件 | 内容 |
|------|------|
| `shared_include/fwif/mtfw_fwif_types.h` | `MTFW_RING_CTRL` — CCB 环控制结构 |
| `shared_include/fwif/mtfw_fwif_ccb.h` | `MTFW_CCB_ITEM`, `MTFW_SUBMISSION_PARA` — CCB 条目 |
| `shared_include/fwif/mtfw_fwif_queue.h` | Doorbell 队列相关定义 |
| `shared_include/linux/uapi/drm/mtgpu_drm.h` | DRM ioctl 结构体定义 |
