# linux-ddk 项目模块依赖与交互分析

> 生成时间：2026-05-18 | 基于源码分析

---

## 目录

1. [项目全景：linux-ddk 有哪些模块](#1-项目全景linux-ddk-有哪些模块)
2. [musa 内部构建依赖链](#2-musa-内部构建依赖链)
3. [musa 对外部模块的依赖](#3-musa-对外部模块的依赖)
4. [模块间通信机制](#4-模块间通信机制)
5. [运行时交互时序图](#5-运行时交互时序图)
6. [关键文件速查表](#6-关键文件速查表)

---

## 1. 项目全景：linux-ddk 有哪些模块

linux-ddk 是一个 **GPU 驱动全栈 monorepo**，包含从 firmware 到用户态 API 的所有层。

### 1.1 模块一览

```
linux-ddk/                              # MUSA GPU Driver Platform
│
├── 🧠 musa/                 【UMD】用户态计算驱动 — 本分析的核心
│   最终产物: libmusa.so (CUDA 兼容 GPU 驱动)
│   角色: 实现 MUSA API、运行时核心、HAL、M3D GPU 接口
│
├── 🔧 gr-kmd/               【KMD】内核态驱动
│   最终产物: mtgpu.ko 等内核模块
│   角色: GPU 硬件管理、MMU/IOMMU 页表、调度、命令提交
│   └── 子模块: mt-rm/ (资源管理), services/ (服务层), mtgpu/ (设备),
│               mtgpu-next/, mtsnd/, mtvpu/, common/, hwdefs/
│
├── 🔗 libdrm-mt/            【库】DRM 接口封装库
│   最终产物: libdrm_mt.so
│   角色: 封装 Linux DRM ioctl 调用，是 UMD↔KMD 的通信胶水
│   └── 关键文件: mtgpu/mtgpu_bo.c, mtgpu/mtgpu_vm.c, mtgpu/mtgpu_job.c
│
├── 🎮 gr-umd/               【UMDs】用户态图形/视频驱动
│   最终产物: libGL*.so, libVulkan*.so, libOpenCL*.so 等
│   角色: OpenGL/Vulkan/OpenCL/媒体编解码等图形 API 实现
│   └── 与 musa 关系: 通过 KMD 共享 GPU 硬件; musaGL 互操作
│
├── 📐 shared_include/       【头文件】共享硬件定义
│   最终产物: 纯头文件 (无链接产物)
│   角色: GPU 寄存器定义(hwdefs/), 像素格式, FW接口, IPC接口
│   └── 与 musa 关系: musa 编译时包含 hwdefs/ 以解析 GPU 寄存器
│
├── 📦 gpu-fw/               【固件】GPU 固件镜像
│   最终产物: .bin 固件文件
│   角色: GPU 微码 (各种 gen 的 build target)
│   └── 与 musa 关系: 运行时由 KMD 加载到 GPU，musa 间接依赖
│
├── 🗂️ grallocBoMgr/         【库】Android Graphics Allocator
│   最终产物: gralloc 库
│   角色: Android 图形缓冲分配器/管理器
│   └── 与 musa 关系: 弱依赖，共享 M3D 依赖
│
├── 🖼️ ogl/                  【ICD】OpenGL ICD
│   最终产物: libGL*.so (OpenGL 可安装客户端驱动)
│   角色: OpenGL 实现
│   └── 与 musa 关系: GL-CUDA interop 场景
│
├── 🎯 vulkan/               【ICD】Vulkan ICD
│   最终产物: libvulkan*.so
│   角色: Vulkan 实现
│   └── 与 musa 关系: Vulkan-CUDA interop 场景
│
├── 📺 mt-video-drv/         【驱动】视频解码驱动
├── 🎬 mt-media-driver/      【驱动】媒体驱动
├── 🖥️ xf86-video-mtgpu/    【DDX】X.org 显示驱动
├── 🗜️ mthreads-gmi/        【库】MThreads GMI (GPU Management Interface)
└── 🧪 libgfxc/             【库】图形编译器支持库
```

### 1.2 依赖强度矩阵

|          | musa | gr-kmd | libdrm-mt | shared_include | gpu-fw | gr-umd |
|----------|:----:|:------:|:---------:|:--------------:|:------:|:------:|
| **musa**  | —    | 🔴 强   | 🔴 强     | 🟡 中         | 🟢 弱  | 🟢 弱  |
| **gr-kmd**| 🔴 强 | —      | 🔴 强     | 🟡 中         | 🔴 强  | 🟢 弱  |
| **libdrm-mt** | 🔴 强 | 🔴 强 | —         | 🟡 中         | 🟢 弱  | 🟢 弱  |
| **gr-umd**| 🟢 弱 | 🔴 强   | 🔴 强     | 🟡 中         | 🔴 强  | —     |

> 🔴 强: 编译+链接+运行时依赖 | 🟡 中: 编译时头文件依赖 | 🟢 弱: 仅运行时间接依赖

---

## 2. musa 内部构建依赖链

musa 自己是一个 **分层构建的 CMake 项目**，从源码到 `libmusa.so` 有 5 个链接层：

### 2.1 构建产物链

```
ddk_build.sh -m 1
  └─ cd musa/build && cmake .. && make -j32

     编译顺序 (自底向上):
     ┌────────────────────────────────────────────────────┐
     │ [1] util.a         静态库 (musa 内部工具)          │
     │     src/util/*.cpp: 数学, bitops, 线程工具         │
     ├────────────────────────────────────────────────────┤
     │ [2] m3d.a + scpc.a 静态库 (三方 GPU SDK, 源码编译) │
     │     src/hal/m3d/m3d/src/**/*.cpp                   │
     │     ~85 个 CMakeLists, 数百个源文件                 │
     ├────────────────────────────────────────────────────┤
     │ [3] halM3d.a       静态库 (HAL 桥接层)             │
     │     src/hal/m3d/*.cpp                               │
     │     链接: m3d + scpc + util                        │
     ├────────────────────────────────────────────────────┤
     │ [4] musaCore.a     静态库 (运行时核心)             │
     │     src/musa/core/**/*.cpp                          │
     │     链接: halM3d                                   │
     ├────────────────────────────────────────────────────┤
     │ [5] libmusa.so     动态库 (最终产物)               │
     │     src/driver/**/*.cpp                             │
     │     链接: musaCore (PRIVATE, 静态连入)              │
     └────────────────────────────────────────────────────┘
```

### 2.2 musa 源码组织结构

```
musa/src/
├── driver/              # 【层0】驱动 API 层
│   ├── mu_entry.cpp     #   导出表初始化
│   ├── internal.cpp     #   入口表注册 (addEntryPoints)
│   ├── mu_vmm.cpp       #   虚拟内存管理
│   ├── mu_memory.cpp    #   内存分配/释放
│   ├── mu_wrappers_generated.cpp  # 公开 API wrapper (22000+ 行)
│   ├── mu_module.cpp    #   模块加载
│   └── mupti/, mugdb/   #   PTI 工具接口 / GDB 调试
│
├── musa/core/           # 【层1】运行时核心
│   ├── context.cpp      #   Context: 命令生命周期管理
│   ├── stream.cpp       #   Stream: 命令队列抽象
│   ├── device.cpp       #   Device: 引擎属性, 队列族
│   ├── memory.cpp       #   Memory: CPU 地址映射, 物理内存句柄
│   ├── platform.cpp     #   Platform: 全局单例, 内存追踪
│   ├── command/         #   命令层
│   │   ├── memsetCommand.cpp   # 引擎选择: CDM→CE→TDM→DMA→CPU
│   │   ├── dispatchCommand.cpp # GPU kernel dispatch
│   │   ├── memcpyCommand.cpp   # 内存拷贝命令
│   │   └── barrierCommand.cpp  # 同步屏障
│   ├── copyManager2/    #   拷贝引擎后端
│   └── graph/, node/    #   CUDA Graph 支持
│
├── hal/                 # 【层2】HAL 抽象层
│   ├── halDevice.h      #   抽象接口: 设备能力查询
│   ├── halMemory.h      #   抽象接口: Map/Unmap, 物理地址
│   ├── halQueue.h       #   抽象接口: 命令提交, 同步
│   └── m3d/             #   M3D HAL 实现
│       ├── platform.cpp #   桥接: Hal::Platform → M3D::Platform
│       ├── device.cpp   #   队列族构建 (CDM, CE, DMA)
│       ├── memory.cpp   #   Hal::IMemory → M3D::GpuMemory
│       ├── virtualMemory.cpp # VA reservation → M3D CreateGpuMemory
│       └── m3d/         #   M3D SDK (三方, 从源码编译)
│           └── src/core/os/drm/mthreads/
│               ├── mtgpuDevice.cpp  # DRM 设备初始化
│               └── mtgpuMemory.cpp  # AllocateOrPinMemory
│
├── musa_shared_include/ # 【共享头文件层】
│   ├── musa.h           #   公开 API 声明 (27479 行)
│   ├── export_table.h   #   导出表结构体定义 (ADDMEMBER 宏)
│   ├── generated_musa_meta.h  # 工具回调元数据
│   └── mupti/           #   PTI 跟踪接口
│
└── util/                # 工具函数
```

---

## 3. musa 对外部模块的依赖

### 3.1 libdrm-mt — 运行时通信胶水

**依赖类型**: 🔴 运行时链接依赖 (动态库)

**musa 如何依赖它**:
- musa 通过 M3D SDK 调用 libdrm-mt 封装的 ioctl
- 关键调用都在 `m3d/src/core/os/drm/mthreads/` 中：

```
mtgpuDevice.cpp:
  mtgpu_device_init()      → libdrm-mt: 打开 /dev/dri/cardX
  AssignVirtualAddress()   → libdrm-mt: DRM_IOCTL_MTGPU_VM_BIND

mtgpuMemory.cpp:
  AllocateOrPinMemory()    → libdrm-mt: DRM_IOCTL_MTGPU_BO_ALLOC
  VirtualReserve()         → libdrm-mt: DRM_IOCTL_MTGPU_VM_RESERVE
  MapToHost()              → libdrm-mt: mmap BO
```

**libdrm-mt 的关键 API** (声明在 `libdrm-mt/mtgpu/mtgpu.h`):

| API | 功能 | musa 调用路径 |
|-----|------|-------------|
| `mtgpu_bo_alloc()` | 分配 GPU buffer object | 内存分配 |
| `mtgpu_bo_free()` | 释放 BO | 内存释放 |
| `mtgpu_bo_mmap()` | 映射 BO 到用户态地址 | CPU 访问 GPU 内存 |
| `mtgpu_vm_bind()` | 绑定 VA 到 BO | 虚拟地址映射 |
| `mtgpu_vm_unbind()` | 解绑 VA | 释放虚拟地址 |
| `mtgpu_job_submit()` | 提交命令到 GPU | 命令执行 |

### 3.2 gr-kmd — 内核态驱动

**依赖类型**: 🔴 运行时系统依赖 (内核模块)

**musa 如何依赖它**:
- musa 不直接链接 gr-kmd
- 通过 `libdrm-mt` → `ioctl()` → Linux DRM 子系统 → `gr-kmd` 内核模块
- gr-kmd 注册为 DRM 驱动（名称为 `"mtgpu"`）

```
libdrm-mt: ioctl(fd, DRM_IOCTL_MTGPU_*, ...)
  ↓ syscall
Linux kernel: DRM ioctl dispatch
  ↓
gr-kmd/mtgpu/: MT GPU DRM 驱动处理 ioctl
  ↓
gr-kmd/mt-rm/: Resource Manager — GPU VA/内存分配, 调度
  ↓
gr-kmd/services/: 设备内存、MMU、调度服务
```

**gr-kmd 子模块**:

| 子目录 | 角色 | 输出 |
|--------|------|------|
| `mtgpu/` | MT GPU DRM 驱动 (设备注册, ioctl) | mtgpu.ko |
| `mtgpu-next/` | 新一代 GPU 驱动 | mtgpu-next.ko |
| `mt-rm/` | Resource Manager (RM) — 资源分配核心 | mt-rm 组件 |
| `services/` | 内核服务层 (设备内存, MMU, 调度) | 内核服务 |
| `mtsnd/` | Sound 驱动 | mtsnd.ko |
| `mtvpu/` | VPU (Video Processing Unit) 驱动 | mtvpu.ko |
| `common/`, `hwdefs/` | 共享代码和寄存器定义 | — |

### 3.3 shared_include — 编译时头文件依赖

**依赖类型**: 🟡 编译时头文件依赖

**musa 如何依赖它**:
- musa 的 CMakeLists.txt 通过 `-I` 引入 `shared_include/`
- 主要用于 `shared_include/hwdefs/` — GPU 硬件寄存器位域定义

```
shared_include/
├── hwdefs/       # ← musa 主要使用的部分
│   # QY1/QY2/PH1/LS/HS/HG 等 GPU 家族的寄存器定义
├── fwif/         # 固件接口 (KMD/UCODE 通信)
├── rmif/         # RM 接口定义
├── basic_types.h # 基础类型
├── bitops.h      # 位操作
├── pixel_formats.h # 像素格式 (musa 纹理相关)
└── ...
```

### 3.4 gpu-fw — 运行时固件依赖

**依赖类型**: 🟢 运行时间接依赖

**musa 如何依赖它**: musa 不直接加载固件。流程是：
```
KMD 初始化 → 从 /lib/firmware/ 加载 gpu-fw 产物 → 上传到 GPU
```

musa 运行时通过 KMD 间接依赖 GPU 固件：如果固件未加载，GPU 无法执行命令。

### 3.5 gr-umd — 运行时互操作

**依赖类型**: 🟢 运行时互操作 (共享 GPU)

**musa 如何依赖它**: 无编译/链接依赖，但运行时两者通过以下方式协同：
- 共享 KMD（同一个 GPU 设备）
- KMD 负责计算+图形任务的调度隔离
- `musaGL` 互操作 (musa 可以导入 OpenGL buffer)

---

## 4. 模块间通信机制

### 4.1 通信路径总图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户应用程序                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ CUDA App │  │ OpenGL   │  │ Vulkan   │  │ OpenCL   │        │
│  │          │  │ App      │  │ App      │  │ App      │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       ▼             ▼             ▼             ▼               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │libmusa  │  │libGL    │  │libVulkan│  │libOpenCL│            │
│  │.so      │  │.so      │  │.so      │  │.so      │            │
│  │(musa)   │  │(gr-umd) │  │(gr-umd) │  │(gr-umd) │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │             │             │             │               │
│       │     ┌───────┴─────────────┴─────────────┘               │
│       │     │  (共享 GPU 硬件, 通过 KMD 仲裁)                    │
│       │     │                                                   │
├───────┼─────┼───────────────────────────────────────────────────┤
│ 用户态 │     │                                                   │
│       │     │                                                   │
│       ▼     ▼                                                   │
│  ┌──────────────┐                                               │
│  │  libdrm_mt   │  ← UMD↔KMD 通信胶水                           │
│  │  .so         │    封装 ioctl(fd, DRM_IOCTL_MTGPU_*, ...)     │
│  └──────┬───────┘                                               │
│         │                                                       │
├─────────┼───────────────────────────────────────────────────────┤
│ 内核态  │  ioctl() syscall                                      │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  gr-kmd      │  ← MT GPU 内核模块                            │
│  │  (各种 .ko)  │    处理 ioctl, 管理 GPU 资源                  │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  GPU 硬件    │  ← 固件由 KMD 加载 (gpu-fw)                   │
│  │  + 固件      │                                              │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 关键 ioctl 接口

| ioctl 命令 | 作用 | musa 调用场景 |
|-----------|------|-------------|
| `DRM_IOCTL_MTGPU_BO_ALLOC` | 分配 GPU Buffer Object | `muMemAlloc`, `muMemCreate` |
| `DRM_IOCTL_MTGPU_BO_FREE` | 释放 BO | `muMemFree`, `muMemRelease` |
| `DRM_IOCTL_MTGPU_BO_MMAP` | 映射 BO 到用户态虚拟地址 | `GetHostPointer()` |
| `DRM_IOCTL_MTGPU_VM_BIND` | 绑定 VA 到 BO | `muMemMap` |
| `DRM_IOCTL_MTGPU_VM_UNBIND` | 解绑 VA | `muMemUnmap` |
| `DRM_IOCTL_MTGPU_VM_RESERVE` | 预留 VA 范围 | `muMemAddressReserve` |
| `DRM_IOCTL_MTGPU_JOB_SUBMIT` | 提交 GPU 命令 (command buffer) | `Stream::Submit()` |
| `DRM_IOCTL_MTGPU_DEV_QUERY` | 查询设备能力 | `InitPlatform()` |

### 4.3 musa 不依赖 MUSA-Runtime

**重要澄清**: `linux-ddk/musa/` 和 `/home/shanfeng/MUSA-Runtime/` 是**两个独立的项目**：

| 项目 | 产物 | 角色 |
|------|------|------|
| `linux-ddk/musa/` | `libmusa.so` | 驱动实现 (HAL, M3D, KMD 通信) |
| `MUSA-Runtime/` | `libmusart.so` | 运行时包装层 (CUDA Runtime API 兼容) |

MUSA-Runtime 链接到 libmusa.so 以获得实际驱动能力。两者的关系是：

```
用户 CUDA 应用
  → libmusart.so     (MUSA-Runtime: musaMemcpy, musaMalloc 等 runtime API)
    → libmusa.so     (linux-ddk/musa: muMemcpy, muMemAlloc 等 driver API)
      → libdrm_mt.so (ioctl 封装)
        → gr-kmd.ko  (内核驱动)
```

---

## 5. 运行时交互时序图

### 5.1 完整调用流程：内存分配 (`muMemAlloc`)

```
用户代码                   musa (libmusa.so)              libdrm-mt            gr-kmd (内核)
  │                             │                           │                    │
  │ muMemAlloc(&ptr, size)      │                           │                    │
  ├────────────────────────────►│                           │                    │
  │                             │                           │                    │
  │              ┌─ Driver API Layer ─────────────────────┐ │                    │
  │              │ mu_wrappers_generated.cpp               │ │                    │
  │              │   → muapiMemAlloc()                     │ │                    │
  │              │     → Platform::MemAlloc()              │ │                    │
  │              └────────────────┬────────────────────────┘ │                    │
  │                               │                          │                    │
  │              ┌─ MUSA Core ───┴────────────────────────┐ │                    │
  │              │ memory.cpp: Init(createInfo)            │ │                    │
  │              │   → Context::GeneralAlloc()             │ │                    │
  │              │     → Hal::CreateMemory()               │ │                    │
  │              └────────────────┬────────────────────────┘ │                    │
  │                               │                          │                    │
  │              ┌─ HAL/M3D ──────┴────────────────────────┐ │                    │
  │              │ platform.cpp: CreateMemory()            │ │                    │
  │              │   → m3dDevice->CreateGpuMemory()        │ │                    │
  │              │     → gpuMemory.cpp: Init()             │ │                    │
  │              └────────────────┬────────────────────────┘ │                    │
  │                               │                          │                    │
  │              ┌─ M3D OS/DRM ───┴────────────────────────┐ │                    │
  │              │ mtgpuMemory.cpp: AllocateOrPinMemory()   │ │                    │
  │              │   → mtgpu_bo_alloc()  ────────────────────┤                    │
  │              │                      ┌────────────────────┘                    │
  │              │                      │ mtgpu_bo.c: mtgpu_bo_alloc()            │
  │              │                      │   → drmIoctl(fd,                        │
  │              │                      │       DRM_IOCTL_MTGPU_BO_ALLOC, &req)   │
  │              │                      │─────────────────────────────────────────►│
  │              │                      │                          ioctl syscall   │
  │              │                      │                          ┌───────────────┤
  │              │                      │                          │ mtgpu DRM 驱动│
  │              │                      │                          │  → RM 分配    │
  │              │                      │                          │  → MMU 映射   │
  │              │                      │                          │  → 返回 GEM   │
  │              │                      │                          │    handle     │
  │              │                      │◄─────────────────────────┤               │
  │              │                      │   返回 BO handle         │               │
  │              │   ←──────────────────┤                          │               │
  │              │    BO + VA                                        │               │
  │              └──────────────────────┘                          │               │
  │                               │                          │                    │
  │              ┌─ 返回指针 ─────┴──────────────────────────┐                    │
  │              │ *ptr = pMemory->GetDevicePointer()        │                    │
  │              │   → 注册到 MemoryTracker                  │                    │
  │              └────────────────┬──────────────────────────┘                    │
  │                               │                          │                    │
  │  return musaSuccess           │                          │                    │
  │◄──────────────────────────────┤                          │                    │
```

### 5.2 虚拟地址预留 (`muMemAddressReserve`)

```
用户代码            musa (libmusa.so)               libdrm-mt + gr-kmd
  │                       │                                │
  │ muMemAddressReserve   │                                │
  │  (&ptr, size, 0, 0, 0)│                                │
  ├──────────────────────►│                                │
  │                       │                                │
  │  driver/mu_vmm.cpp    │                                │
  │  muapiMemAddressReserve│                                │
  │   ├─ 校验: size>0, alignment=2^n                      │
  │   ├─ 校验: UVA 支持                                   │
  │   ├─ 校验: size % granularity == 0                    │
  │   └─ Platform::CreateMemory()                         │
  │       └─ Memory::VirtualAlloc()                        │
  │           └─ Hal::CreateMemory(memoryTypeVirtual)      │
  │               └─ VirtualMemory::InitInternal()         │
  │                    ├─ 验证 heap size                   │
  │                    ├─ GpuMemoryCreateInfo{              │
  │                    │    virtualAlloc=1,                 │
  │                    │    svmAlloc=1,                     │
  │                    │    globalGpuVa=1 }                 │
  │                    └─ m3dDevice->CreateGpuMemory() ────►│
  │                       └─ mtgpuDevice.cpp:              │
  │                          AssignVirtualAddress() ───────►│
  │                             ioctl(VM_RESERVE)           │
  │                                ← ─ ─ ─ ─ ─ ─ VA range  │
  │                       *ptr = reserved VA               │
  │                                                         │
  │  return SUCCESS, *ptr=0x7f...                           │
  │◄────────────────────────────────────────────────────────┤
```

### 5.3 GPU 命令提交 (Memset 示例)

```
用户代码          musa (libmusa.so)                     libdrm-mt + gr-kmd
  │                     │                                     │
  │ muMemsetD8Async     │                                     │
  │  (ptr, 0, size, s)  │                                     │
  ├────────────────────►│                                     │
  │                     │                                     │
  │  driver/mu_memory.cpp                                     │
  │  muapiMemsetD8Async  │                                     │
  │   → Context::GeneralMemset()                              │
  │      ├─ 创建 GraphMemsetNode                              │
  │      └─ stream->CmdMemset()                               │
  │          └─ MemsetCommand ctor                            │
  │              ├─ 引擎选择: CDM→CE→TDM→DMA→CPU              │
  │              └─ Build() → 编码命令缓冲                    │
  │                                                           │
  │  ResolveDependencyAndQueueCommand()                       │
  │   └─ HalQueue::Submit()                                   │
  │       └─ Queue::Submit()                                  │
  │           └─ mtgpu_job_submit() ────────────────────────► │
  │               ioctl(DRM_IOCTL_MTGPU_JOB_SUBMIT)           │
  │                                        ┌──────────────────┤
  │                                        │ KMD 调度器       │
  │                                        │  → 将命令缓冲    │
  │                                        │    提交到 GPU    │
  │                                        │    硬件队列      │
  │                                        │                  │
  │                                        │ GPU 执行 memset  │
  │                                        │  → 写完成后      │
  │                                        │    触发 fence    │
  │                     ◄──────────────────┘                  │
  │                     fence signal                         │
  │                                                           │
  │  return SUCCESS                                           │
  │◄──────────────────────────────────────────────────────────┤
```

### 5.4 项目构建时序

```
ddk_build.sh -a 0 -m 1  (只构建 musa, Release 模式)
│
├─ [1] 构建 gpu-fw (如果 BUILD_FW=1)
│     编译 GPU 固件 → *.bin 文件
│
├─ [2] 构建 libdrm-mt (BUILD_LIBDRM=1)
│     cd libdrm-mt && meson build && ninja
│     产物: libdrm_mt.so
│
├─ [3] 构建 gr-kmd (BUILD_KMD=1)
│     cd gr-kmd && make
│     产物: mtgpu.ko, ... 
│
├─ [4] 构建 musa (BUILD_MUSA=1)  ← 本分析核心
│     ┌─────────────────────────────────────────┐
│     │ cd musa/build                           │
│     │ cmake ..                                │
│     │   -DDDK_2_0=ON                          │
│     │   -DLIBDRM_PATH=../libdrm-mt            │
│     │   -DSHARED_INCLUDE_PATH=../shared_include│
│     │                                         │
│     │ make -j32                                │
│     │   ├─ [4a] libutil.a      (工具库)       │
│     │   ├─ [4b] libm3d.a       (GPU SDK)      │
│     │   ├─ [4b] libscpc.a      (编译器接口)    │
│     │   ├─ [4c] libhalM3d.a    (HAL 桥接)     │
│     │   ├─ [4d] libmusaCore.a  (运行时核心)   │
│     │   └─ [4e] libmusa.so     (最终产物)     │
│     └─────────────────────────────────────────┘
│
├─ [5] 构建 gr-umd (BUILD_UMD=1)
│     编译 OpenGL/Vulkan/OpenCL 等用户态图形驱动
│
└─ 产物汇集到 build/ 目录 → 打包为 .deb
```

---

## 6. 关键文件速查表

### musa 与外部模块的接口文件

| 文件 | 所属模块 | 作用 |
|------|---------|------|
| `libdrm-mt/mtgpu/mtgpu.h` | libdrm-mt | 所有 DRM ioctl wrap 函数的声明 |
| `libdrm-mt/mtgpu/mtgpu_bo.c` | libdrm-mt | BO alloc/free/mmap 实现 |
| `libdrm-mt/mtgpu/mtgpu_vm.c` | libdrm-mt | VA bind/unbind/reserve 实现 |
| `libdrm-mt/mtgpu/mtgpu_job.c` | libdrm-mt | GPU job submit 实现 |
| `shared_include/hwdefs/` | shared_include | GPU 硬件寄存器定义 |
| `gr-kmd/mtgpu/` | gr-kmd | MT GPU DRM 驱动 ioctl handler |
| `gr-kmd/mt-rm/` | gr-kmd | Resource Manager (VA/物理内存分配) |
| `gpu-fw/build_gen4/` | gpu-fw | GPU 固件 (QY2/S4000) |
| `gpu-fw/build_gen7/` | gpu-fw | GPU 固件 (PH1/S5000) |

### musa 内部关键文件 (按层)

| 层 | 文件 | 做什么 |
|----|------|--------|
| Driver API | `driver/mu_vmm.cpp` | muMemAddressReserve, muMemMap, muMemCreate |
| Driver API | `driver/mu_memory.cpp` | muMemAlloc, muMemFree, memcpy, memset |
| Driver API | `driver/internal.cpp` | API 入口注册 (addEntryPoints) |
| Driver API | `driver/mu_wrappers_generated.cpp` | 所有公开 API 的 wrapper 实现 |
| MUSA Core | `core/platform.cpp` | Platform 单例, CreateMemory |
| MUSA Core | `core/context.cpp` | Context 管理, GeneralMemset, ResolveDependency |
| MUSA Core | `core/stream.cpp` | Stream 管理, CmdMemset, CmdDispatch |
| MUSA Core | `core/memory.cpp` | Memory 对象, GetHostPointer, VirtualAlloc |
| MUSA Core | `core/device.cpp` | Device 初始化, 引擎属性, 拷贝管理器 |
| MUSA Core | `core/command/memsetCommand.cpp` | Memset 命令封装, 引擎选择 |
| HAL | `hal/m3d/platform.cpp` | CreateMemory 桥接 |
| HAL | `hal/m3d/virtualMemory.cpp` | VA reservation → M3D CreateGpuMemory |
| HAL | `hal/m3d/device.cpp` | 队列族构建 |
| M3D DRM | `m3d/.../mtgpuDevice.cpp` | DRM 设备初始化, VA 管理器 |
| M3D DRM | `m3d/.../mtgpuMemory.cpp` | AllocateOrPinMemory, VirtualReserve |
| M3D DRM | `m3d/.../mtgpuDevice.cpp:1536` | AssignVirtualAddress → KMD IOCTL |

### 构建系统文件

| 文件 | 作用 |
|------|------|
| `ddk_build.sh` | 顶层构建入口 (所有模块调度) |
| `musa/CMakeLists.txt` | musa 项目根配置 |
| `musa/src/driver/CMakeLists.txt` | musa_dynamic → libmusa.so |
| `musa/src/musa/core/CMakeLists.txt` | musaCore 静态库 |
| `musa/src/hal/m3d/CMakeLists.txt` | halM3d 静态库 (链接 m3d+scpc) |
| `gr-kmd/Makefile` | KMD 构建 |
| `libdrm-mt/meson.build` | libdrm-mt 构建 (meson) |
| `gr-umd/CMakeLists.txt` | 图形 UMD 构建 |

---

## 附录：平台差异速查

| 特性 | QY2 iGPU (S4000) | QY2 dGPU | PH1 (S5000) |
|------|-----------------|----------|-------------|
| CDM 引擎 | ✗ | ✗ | ✓ |
| CE 引擎 | ✗ | ✓ | ✓ |
| DMA 引擎 | ✗ | ✓ | ✓ |
| CPU 回退 | ✓ | ✓ | ✓ |
| CE padding (QY2 only) | +32B | +32B | 无 |
| supportCopyEngine | false | true | true |
| DmaIpLevel | None | — | — |
| 固件 | build_gen4 | build_gen4 | build_gen7 |
