# MUSA Kick 机制：从命令提交到硬件通知

> Kick = 通知 GPU 固件"有新工作"的硬件信号。本质是**一次 MMIO 寄存器写操作**。
>
> 以一次 `musaLaunchKernel` 为线索，追踪从用户态到硬件寄存器的完整路径。

---

## 1. 什么是 Kick

```
     CPU 侧（驱动）                               GPU 侧
     ────────────                              ────────
                                           ┌─────────────┐
  ① 写命令到 cmd buffer                    │  cmd buffer  │
     (GPU 可见内存)                        │  (GPU VRAM)  │
                                           └──────┬──────┘
                                                  │
  ② 更新 CCB 写指针                                  │
     ccb_ctrl->wrIdx = 5;                    ┌──────▼──────┐
                                           │  CCB 环形队列 │
  ③ ★ Kick ★                                │ [0][1][2][3] │
     写 doorbell 寄存器                      │  [4]←新命令  │
     *doorbell_reg = 1 << (id % 32);        └──────┬──────┘
       │                                           │
       │                                  固件检测到 doorbell
       │                                  读取 CCB[wrIdx-1]
       │                                  执行新命令
       ▼
```

**Kick 就是第 ③ 步：敲一下"门铃"，告诉 GPU 固件"CCB 里有新东西，来看看"。**

---

## 2. 两种 Kick 路径

MUSA 驱动有两条 Kick 路径：

| | Doorbell 路径（快速） | IOCTL 路径（传统） |
|---|---|---|
| **怎么通知 GPU** | 用户态直接写 MMIO 寄存器 | 走系统调用到内核，内核写寄存器 |
| **延迟** | ~100ns（一次内存写） | ~2-5μs（系统调用） |
| **什么时候用** | 同 stream 上的**后续**提交 | 不支持 doorbell 的老芯片 / 首次提交 |
| **适用引擎** | CDM, CE, ACE | 所有引擎 |

---

## 3. 例子：一个 kernel 如何触发 Kick

假设用户代码：

```c
musaStream_t stream;
musaStreamCreate(&stream);
musaLaunchKernel(myKernel, grid, block, args, 0, stream);
musaDeviceSynchronize();  // 等 GPU 完成
```

从 Command 提交到 Kick 的完整链路：

```
musaLaunchKernel()
  │
  ▼
Stream::CmdLaunchKernel()                        [stream.cpp:1482]
  → new DispatchCommand(stream, CDM, ...)        [阶段 ①]
  → ResolveDependencyAndQueueCommand()            [阶段 ② 依赖解析]
  → QueueCommand()                                [阶段 ③ 入队]
    → m_CommandList.push(command)
    → 唤醒 AsyncSubmit 线程

                                 AsyncSubmit 线程
                                 ────────────────
Stream::AsyncSubmit()                            [stream.cpp:1141]
  → buildCommand()                               [阶段 ④]
    → CanMergeTo() → false（单命令，不合并）
    → Build()
      → 编码 PM4 包到 Hal::ICmdBuffer
      → 建立 semaphore 依赖
    → submitMergingList()
      → Submit()                                 [阶段 ⑥]
        → SubmitToQueue()                        [command.cpp:646]
          → 构建 wait/signal semaphore 列表
          → pQueue->Submit(submitInfo) ───────────────────┐
                                                          │
═══════════════════════ Kick 分界线 ═══════════════════════│
                                                          ▼
Hal::M3d::Queue::Submit()                      [hal/m3d/queue.cpp]
  → IM3d::Queue::Submit()
    → Queue::SubmitInternal()                  [m3d/core/queue.cpp]
      → CmdBuffer::End()                       [生成 kick 信息]
      → Queue::LaunchCommandStreams()          [m3d/core/.../mtgpuQueue.cpp:443]
         │
         ├── [如果支持 doorbell] ──→ ★ 走 Doorbell 路径（§4）
         │
         └── [不支持 doorbell]  ──→ ★ 走 IOCTL 路径（§5）
```

---

## 4. Doorbell 路径详解（快速路径）

### 4.1 前提：Doorbell 获取

Doorbell 不是每个提交都申请——在**创建 Stream 时**一次性获取，整个 Stream 生命周期复用。

```
Stream::Init()
  → queueCreateInfo.doorbell = userQueue && EnableDoorbell()   [stream.cpp:905]
  → HalDevice.CreateQueue(queueCreateInfo)
    → M3D Queue::Init()
      → Device::AcquireDoorbell()                              [mtgpuQueue.cpp:267]
```

**Device::AcquireDoorbell** 调用到 libdrm，执行三步：

```
mtgpu_job_doorbell_acquire()                       [libdrm/mtgpu_job.c:451]
  │
  ├─ ① mmap() 一页匿名内存
  │     ptr = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
  │                MAP_ANONYMOUS|MAP_SHARED, -1, 0);
  │     // 此时 ptr 只是一个普通的内存页，没有任何特殊含义
  │
  ├─ ② 调 DRM ioctl，让内核把 MHI 硬件寄存器映射到这个页上
  │     ioctl_args.in.user_va = (uint64_t)ptr;         // 告诉内核页的地址
  │     drmCommandWriteRead(fd, DRM_MTGPU_CMD, &args);
  │     // 内核收到后：
  │     //   1. 从 bitmap pool 分配一个 doorbell ID
  │     //   2. 用 remap_pfn_range 把 MHI 寄存器的物理地址映射到用户页
  │     ioctl_args.out.doorbell_handle = 5;            // 内核返回 doorbell ID=5
  │     ioctl_args.out.doorbell_addr_offset = 0x104;   // 寄存器在页内的偏移
  │
  └─ ③ 计算 doorbell_reg 指针
       doorbell->doorbell_id  = 5;
       doorbell->doorbell_reg = (uint32_t*)(ptr + 0x104);
       //  ↑ 此时 *doorbell_reg 指向的是 GPU 的 MHI 硬件寄存器！
       //    这个地址在 PCIe BAR 空间里，写它就是写硬件
```

**此时用户态有了一个指向 GPU 硬件寄存器的指针**。之后写这个指针就等于直接通知 GPU。

#### 物理地址的计算（内核侧 `doorbell.c:23`）

```
doorbell #5 的物理地址 = MHI_BASE + 0x104 + (5 / 32) * 4
                       = MHI_BASE + 0x104 + 0 * 4
                       = MHI_BASE + 0x104

doorbell #37 的物理地址 = MHI_BASE + 0x104 + (37 / 32) * 4
                        = MHI_BASE + 0x104 + 4
                        = MHI_BASE + 0x108
```

每个 32-bit 寄存器保存 32 个 doorbell 信号，doorbell #N 对应 bit `N % 32`。

### 4.2 首次提交：SubmitWithDoorbell

Stream 上的第一个命令需要走一次 ioctl，把**命令数据 + doorbell 关联**发送到内核：

```
Queue::LaunchCommandStreams()                      [mtgpuQueue.cpp:559]
  if (hDoorbell != 0) {                            // 当前 stream 有 doorbell
      Device::SubmitCommandsWithDoorbell(
          hContext,
          checkSemaphores, updateSemaphores,
          m_firstCmdStreamAddr,                    // GPU 命令缓冲区的 VA
          m_firstCmdStreamSize,
          hDoorbell);                              // doorbell handle
  }
```

这个调用走 `DRM_IOCTL_MTGPU_CMD` → 内核侧 `mtgpu_job_submit_with_doorbell_ioctl()`：

```
内核收到后做什么：
  1. 构建 CCB item，包含：
     - submission_va   = GPU 命令缓冲区地址
     - submission_size = 命令大小
     - withDoorbell = 1          ← 告诉固件：等 doorbell 信号
     - doorbellId    = 5         ← 固件等哪个 doorbell
  2. 写 CCB item 到 reqCcbQ
  3. 更新 CCB wrIdx
  4. 调用 mtgpu_fw_kick() 通知固件

固件收到后：
  - 读取 CCB item：这个提交关联 doorbell #5
  - 固件开始监控 doorbell #5
  - 等待 doorbell 信号才执行实际命令
```

**首次提交后，固件知道"doorbell #5 收到了才执行这批命令"。**

### 4.3 ★ 后续提交：直接 Ring Doorbell ★

这是 Kick 机制的核心——**后续提交不再走 ioctl**：

```
Stream 上的第二个 kernel：
  Queue::LaunchCommandStreams()
    // 命令数据已经在 GPU 可见内存里了（不需要再发 CCB item）
    // 只需要通知固件"可以开始了"
    Device::RingDoorbell(hContext, hDoorbell)      [mtgpuQueue.cpp:620]

      ↓ 调用 libdrm
      mtgpu_job_doorbell_ring(dev, jobCtx, doorbell_handle)
                                                      [mtgpu_job.c:554]
        struct mtgpu_doorbell *doorbell = (void*)doorbell_handle;

        // ★ 这一行就是整个 Kick ★
        *doorbell->doorbell_reg = 1 << (doorbell->doorbell_id % 32);
        //                     = 1 << (5 % 32)
        //                     = 1 << 5
        //                     = 0x20

        // N 行关键代码对比 ──────────────────────
        //                             │
        // Doorbell Kick:              │  IOCTL Kick (对比):
        // 1 次用户态内存写            │  drmCommandWriteRead()
        // 0 次系统调用                │  → syscall → 内核 → CCB → kick
        // 0 次内核态执行              │  → ~2-5μs
        // ~100ns                      │
```

**为什么后续提交不需要 ioctl？**

因为命令数据已经在 GPU 可见内存里了。固件在首次提交时已经拿到了命令缓冲区的地址，后续只需要"敲门"告诉固件"可以开始了"。

---

## 5. IOCTL 路径详解（传统路径）

当不支持 doorbell 时（老芯片或非 CDM/CE/ACE 引擎），每个提交都走 ioctl：

```
Queue::LaunchCommandStreams()
  if (hDoorbell == 0) {                            // 没有 doorbell
      Device::SubmitCommandsV3(
          checkSemaphores, updateSemaphores,
          m_firstCmdStreamAddr, m_firstCmdStreamSize,
          submissionFlags, ...);
  }
```

这调到了 `DRM_IOCTL_MTGPU_CMD` → 内核 `mtgpu_job_submit_ioctl_v3()`：

```c
// mtgpu_job_v3.c:1931
int mtgpu_job_submit_ioctl_v3(...) {
    // ① 映射 submission type → 固件命令类型
    //    COMPUTE → MTFW_SUBMISSION_CMD_GPU_CDM (104)
    //    CE      → MTFW_SUBMISSION_CMD_GPU_CE  (105)

    // ② 复制用户态 semaphore/fence 数据

    // ③ 两条路径：
    if (use_host_scheduler) {
        // Path A: 交给 DRM scheduler，调度线程异步处理
        mtgpu_sched_job_create_and_push();
    } else {
        // Path B: 直接提交
        mtgpu_fw_cmd_submit(dev_node, priority, queue_type,
                            sub_cmd, ccb_item_para, ...);
    }
}
```

### 5.1 `mtgpu_fw_cmd_submit` — 写 CCB + Kick

**文件**：`gr-kmd/mtgpu-next/mtgpu_fw_job.c:452`

```c
int mtgpu_fw_cmd_submit(dev_node, priority, queue_type, ...) {
    // ====== 步骤 1: 获取 CCB 写位置 ======
    ccb_ctrl = &mtfw_ccb_queue->reqCcbCtrl[PRIORITY_LOWEST];
    err = mtgpu_fw_get_ccb_woff(&ccb_ctrl_tmp, &new_woff);
    //  ↑ 环形队列：new_woff = (wrIdx + 1) % capacity
    //    如果环形队列满了 → 睡眠等待固件消费（最长等 5 秒）

    // ====== 步骤 2: 分配 CCB item ======
    global_index = CCB_POOL_INDEX(queue_type, pool_index);
    item_fw_addr = ccb_item_pool_mcu_addr + global_index * sizeof(MTFW_CCB_ITEM);

    // ====== 步骤 3: 填充 CCB item ======
    ccb_item->submissionCmd = sub_cmd;               // GPU_CDM = 104
    ccb_item->processId     = process_id;
    ccb_item->priority      = priority;
    ccb_item->hostPrivateData = (uintptr_t)job_item; // 用于跟踪
    ccb_item->ccbItemPara   = *ccb_item_para;        // 包含 submission_va, semaphores 等
    // 写入 GPU 可见内存
    os_memcpy_toio(ccb_item_pool + global_index, ccb_item, sizeof(MTFW_CCB_ITEM));

    // ====== 步骤 4: 写入 CCB 队列 ======
    ccb_qcell = &mtfw_ccb_queue->reqCcbQ[PRIORITY_LOWEST][ccb_ctrl_tmp.wrIdx];
    ccb_qcell->ccbItemAddr = item_fw_addr;       // 固件来读这个地址就知道新 item 在哪

    // ====== 步骤 5: 内存屏障 ======
    os_wmb();                                      // 确保前面的写对固件可见
    os_ioread32(&ccb_qcell->ccbItemAddr);          // 强制 PCIe 写完成
    os_wmb();

    // ====== 步骤 6: 更新写指针 ======
    ccb_ctrl->wrIdx = new_woff;                    // ★ 固件读到 wrIdx 变化就知道有新工作
    os_wmb();
    os_ioread32(&ccb_ctrl->wrIdx);                 // 强制 PCIe 写完成

    // ====== 步骤 7: Kick！ ======
    mtgpu_fw_kick(dev_node, queue_index);           // → §6

    return 0;
}
```

### 5.2 CCB 环形队列图示

```
CCB 环形队列（GPU 可见内存）
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ [0] │ [1] │ [2] │ [3] │ [4] │ [5] │ [6] │ [7] │
│已处理│已处理│已处理│已处 │ 空  │ 空  │ 空  │ 空  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    ▲rdIdx=3      ▲wrIdx=4

驱动写新提交:                        固件处理:
1. 写 CCB item 到 slot[4]           1. 检测 wrIdx 变化
2. ccb_qcell[4].itemAddr = 0x...    2. 读 ccb_qcell[4].itemAddr
3. wrIdx = 5                         3. 读 CCB item 内容
4. Kick → notify firmware            4. 执行命令
                                    5. rdIdx = 5
```

---

## 6. `mtgpu_fw_kick` — 写硬件寄存器

**文件**：`gr-kmd/mtgpu-next/mtgpu_fw.c:61`

```c
void mtgpu_fw_kick(struct mtgpu_device_node *dev_node, u32 queue_index) {
    if (RM_FEATURE_IS_ENABLED(FW))
        dev_node->rm_fw_kick(dev_node->rm_fw_priv, queue_index);   // RM/Rust 路径
    else
        dev_node->fw_ops->kick(dev_node, queue_index);             // 传统路径
}
```

### 6.1 传统路径：两种固件后端

```
mtgpu_fw_kick()
  └─ fw_ops->kick(dev_node, queue_index)

       ├── RISC-V 固件（PH1/PH1S/LS/HS/HG 芯片）
       │   → riscv_fw_kick()                              [mtgpu_riscv.c:286]
       │       uint32_t reg_val = ccb_to_schedule_val_tbl[queue_type];
       │       // 映射表: GP→EVENT_0, CDM→EVENT_3, CE→EVENT_4, UQ→EVENT_2
       │       os_iowrite32(reg_val, reg_base + MT_CR_FEC_FTS_CORE_OS0_SCHEDULE_FTS_0);
       │       //                                  ↑
       │       //                    FEC MHI 寄存器，通过 PCIe BAR 访问
       │       //                    写入这个寄存器→ GPU 固件被唤醒→读取 CCB
       │
       └── MCU 固件（QY1/QY2 老芯片）
           → mcu_fw_kick()                                [mtgpu_meta.c:788]
               MCU_WRITE_REG32(MUSA_CR_MTS_SCHEDULE, queue_type);
               //              ↑
               //    META MHI 寄存器，老芯片的固件调度寄存器
```

**实际效果**：写一个 PCIe BAR 地址的寄存器，GPU 固件收到中断/轮询到这个变化 → 读 CCB → 执行新命令。

### 6.2 RM 路径（新架构）

```c
// 初始化时（mtgpu_rm.c:126）
dev_node->rm_fw_kick = rm_kick_callback;  // Rust RM 注册的回调

// Kick 时
dev_node->rm_fw_kick(dev_node->rm_fw_priv, queue_index);
// → Rust RM 内部 → 最终也是写 MHI 寄存器
```

---

## 7. Doorbell vs IOCTL：完整对比

```
同一个 Stream 上的 3 个 kernel 提交：

  kernel_1:
    IOCTL 路径:
      musaLaunchKernel → ... → SubmitCommandsV3
        → DRM_IOCTL_MTGPU_CMD (系统调用)
          → 内核: CCB write + wrIdx + fw_kick → 硬件寄存器写
        ← 返回
    耗时: ~3μs

  kernel_2 (同 stream):
    Doorbell 路径:
      musaLaunchKernel → ... → RingDoorbell
        → *doorbell_reg = 1 << (doorbell_id % 32)    ← 只有这一行！
        ← 返回
    耗时: ~100ns（快了 30 倍）

  kernel_3 (同 stream):
    Doorbell 路径（同样）:
      *doorbell_reg = 1 << (doorbell_id % 32)
    耗时: ~100ns
```

**Doorbell 快在哪里**：

| | IOCTL 路径 | Doorbell 路径 |
|---|---|---|
| 系统调用 | 1 次 `drmCommandWriteRead` | **0 次** |
| 内核/用户态切换 | 2 次（进一次，出一声） | **0 次** |
| CCB 写入 | 每次都要 | 首次才要 |
| 硬件寄存器写 | 内核态 | **用户态直接写** |

---

## 8. 关键实现文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `libdrm-mt/mtgpu/mtgpu_job.c:568` | 568 | **Doorbell ring 的一行代码** |
| `libdrm-mt/mtgpu/mtgpu_job.c:451` | 451 | Doorbell acquire（mmap + ioctl remap） |
| `musa/core/stream.cpp:905` | 905 | `EnableDoorbell()` 条件判定 |
| `hal/m3d/.../mtgpuQueue.cpp:559` | 559 | `SubmitCommandsWithDoorbell`（首次） |
| `hal/m3d/.../mtgpuQueue.cpp:620` | 620 | `RingDoorbell`（后续） |
| `gr-kmd/mtgpu-next/mtgpu_fw_job.c:452` | 452 | `mtgpu_fw_cmd_submit`（CCB 写 + Kick） |
| `gr-kmd/mtgpu-next/mtgpu_fw.c:61` | 61 | `mtgpu_fw_kick`（分发） |
| `gr-kmd/mtgpu-next/mtgpu_riscv.c:286` | 286 | `riscv_fw_kick`（RISC-V 硬件写） |
| `gr-kmd/mtgpu-next/mtgpu_meta.c:788` | 788 | `mcu_fw_kick`（MCU 硬件写） |
| `gr-kmd/mt-rm/.../doorbell/doorbell.c:23` | 23 | 物理地址计算 |
| `gr-kmd/mt-rm/.../doorbell/doorbell_mgr_ls.c:15` | 15 | LS 芯片 doorbell 寄存器参数 |
| `gr-kmd/mt-rm/.../rm_os_doorbell.c:18` | 18 | `remap_pfn_range`（MHI → 用户页） |
| `shared_include/fwif/mtfw_fwif_types.h:72` | 72 | `MTFW_RING_CTRL`（CCB 环控制） |
