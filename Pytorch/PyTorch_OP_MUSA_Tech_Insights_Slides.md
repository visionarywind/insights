# PyTorch OP、SGLang DeepSeek V4 与 MUSA 适配技术洞察 - PPT 大纲

## 1. PyTorch OP、SGLang DeepSeek V4 与 MUSA 适配技术洞察
基于 OP 链路、框架源码与 MUSA 用例的工程分析
- 面向 AIInfra / 推理框架 / 后端适配工程师
- 关注点：语义正确、后端命中、Graph 稳定、同步边界、生产热路径

## 2. 为什么从 OP 视角切入
- PyTorch API 只是入口，真正决定质量的是 dispatch、kernel、layout、sync 和框架位置
- 同一个 OP 在 eager、compile、Graph、训练通信中风险完全不同
- MUSA 适配不能停留在“能跑”，还要确认是否命中目标后端并适合热路径
- 这套分析把单 OP、框架源码、MUSA 用例和跨后端经验串成一条链路

## 3. 分享主线
- 1. PyTorch 常见 OP：GPU / CPU / Sync / Dynamic Shape / Graph
- 2. OP 在 SGLang、vLLM、Megatron 中如何进入真实链路
- 3. SGLang DeepSeek V4：metadata、Graph、HC/MHC、MoE 源码拆解
- 4. Eager / Compile / Graph 下同一 OP 的不同风险
- 5. MUSA 视角下的 OP 使用与排查方法

## 4. 证据范围
- 81 个可运行 MUSA 用例保留输入、完整代码和 stdout 输出
- 第 2-5 章提供 SGLang / vLLM / Megatron 源码链路和关键 OP 位置
- 第 6-7 章把源码 OP 链抽成 MUSA 最小场景，覆盖 Graph / compile / 推理流程
- 第 8 章是跨后端工程归纳，用于形成可复用排查清单
- 性能结论仍需要 profiler、kernel timeline、吞吐/延迟数据继续补充

## 5. 一条 OP 从 API 到热路径的链路
- PyTorch API / Tensor OP
- Dispatcher / DispatchKey
- Backend kernel: CPU / CUDA / MUSA / Composite
- Framework usage: SGLang / vLLM / Megatron
- Hot path: attention / KV cache / MoE / sampling / communication
- MUSA checks: layout / graph-safe / no CPU sync / native kernel
- 检查 OP 不能只看 API 名字，要沿着链路追到真实执行位置和后端实现

## 6. PyTorch OP 分类框架
| 类别 | 代表 OP | 主要风险 |
| --- | --- | --- |
| GPU OP | empty, view, matmul, topk, softmax | layout、fallback、小 OP 链 |
| CPU OP | dict/list, tokenizer, CPU tensor | 控制面侵入热路径 |
| Sync OP | item, cpu, synchronize, work.wait | 隐式同步、overlap 失效 |
| Dynamic Shape | nonzero, unique, masked_select | shape 不稳定、Graph 失败 |
| Graph OP | copy_, fill_, zero_, graph.replay | 地址/shape/路径不固定 |

## 7. GPU OP：不是“在 GPU 上跑”就算完成
- 创建初始化：empty/new_empty 用于固定 buffer，但 empty 后必须完整写入
- Shape/Layout：view/reshape/contiguous 决定 custom kernel 能否正确读数据
- 数学激活：silu/softmax/rsqrt 可表达语义，生产路径通常需要融合
- 线性代数/路由：linear/matmul/topk 的关键是 dtype、layout 和后端 kernel 命中
- dtype/device：to/float/bfloat16 常在边界处集中处理，避免隐式拷贝

## 8. CPU OP 与 Sync OP：推理服务的时序边界
- CPU OP 应负责 request queue、scheduler、KV block、tokenizer、协议输出
- device 热路径只消费 CPU 规划后的固定 tensor metadata
- .item/.tolist/.cpu 是隐式同步，适合小结果边界，不适合每 token decode
- stream/event 用于局部依赖；全设备 synchronize 应只出现在 benchmark/debug 边界
- 通信 wait 的时机决定计算-通信 overlap 是否成立

## 9. Dynamic Shape 与 Graph 的核心冲突
- Dynamic Shape OP 的输出长度依赖数据值，Graph 需要固定 shape 和固定地址
- nonzero/unique/masked_select 适合调试和 CPU planner，不适合 replay 热路径
- 生产推理通常把动态性收敛到 bucket、padding、mask、page table、top-k
- Graph replay 前通过 copy_/fill_/zero_ 更新固定 buffer，replay 内保持路径稳定
- 如果 capture 时是低效 fallback，Graph 只会把低效路径固定下来

## 10. SGLang：四层 OP 分工
| 层次 | 关键 OP / 对象 | 作用 |
| --- | --- | --- |
| 模型语义层 | linear, silu, topk, sum | 表达 forward / fallback reference |
| metadata 层 | seq_lens, page_table, out_cache_loc | 把动态请求压成 tensor 说明书 |
| Graph replay 层 | copy_, foreach_copy_, fill_, zero_ | 更新固定 buffer 内容 |
| fused backend 层 | FlashMLA, DeepGEMM, TileLang, MUSA native | 承接生产热路径性能 |

## 11. SGLang DeepSeek V4 调用流程
- CPU scheduler 选择 batch / bucket / KV slot
- 构造 CPU mirror 与 device metadata
- Attention prepare 整理 Q/KV layout
- DSV4 metadata copy 到固定对象
- Graph runner replay_prepare 更新 fixed buffers
- graph.replay 执行 attention / HC / MoE / logits
- Sampling 只同步小结果
- 核心模式：动态决策在 Graph 外，Graph 内只消费固定 shape、固定地址、固定路径的 tensor

## 12. DSV4 Metadata：动态请求如何变成固定输入
- DSV4 metadata 是 attention backend 的“说明书”：seq lens、positions、page table、out loc 等
- Graph replay 时 metadata 对象地址保持不变，只用 copy_ 更新 tensor 字段内容
- seq_lens_cpu 负责 max/item 等 CPU 决策，避免对 MUSA tensor 做隐式同步
- padding 到 capture bucket，保证 replay 中 shape 固定
- 风险点：dtype 不匹配、padding 未初始化、replay 内替换 tensor 对象

## 13. Graph Runner：copy ladder 是必要入口，也是性能风险
- _grouped_foreach_copy_ 按 dtype 分组，减少 Python 循环和 launch 管理成本
- populate_from_forward_batch 先 fill_/zero_ padding，再 copy 真实 batch
- replay_prepare 选择 capture bucket，更新 input_ids、positions、seq_lens、metadata
- graph.replay 只复用 capture 路径，不负责修复低效 OP 链
- 排查重点：copy 数量、buffer 复用、padding 完整性、capture 内后端 kernel 支持

## 14. HC/MHC 与 MoE SwiGLU：fallback 语义不等于生产路径
- HC/MHC fallback: flatten -> square -> mean -> rsqrt -> F.linear -> sum 清晰表达语义
- decode 小 batch 下碎 OP 链会带来多次 launch 和 HBM 读写
- MUSA 生产路径应优先 TileLang / MUSA native / fused kernel
- MoE SwiGLU 重点看 empty buffer 是否完整写入、view 是否满足 contiguous 契约
- SwishGLU / clamp / grouped GEMM 要和 CUDA fused path 数值与 layout 对齐

## 15. vLLM 对比：PagedAttention metadata 是中心
- vLLM 的关键不是普通 attention API，而是 CPU scheduler 生成的 PagedAttention metadata
- block_tables 记录 sequence 使用哪些 KV block
- slot_mapping 记录 token 写入/读取 KV cache 的具体位置
- seq_lens_tensor 进入 attention backend，Graph replay 需要固定 dtype/shape/layout
- MUSA 适配要确认 decode 主路径命中 PagedAttention custom kernel，而不是 padding + SDPA fallback

## 16. Megatron 对比：训练 OP 的核心是通信与状态
- Megatron 的 OP 不只表达数学计算，还参与 TP/PP/DP/EP 并行编排
- TP linear: all_gather -> matmul -> reduce_scatter/all_reduce，关键是 overlap
- PP schedule: send/recv/wait 时机决定 pipeline bubble 和显存峰值
- MoE dispatcher: topk/view/permute/alltoall/combine 必须满足 padding、dtype、layout 契约
- Optimizer offload 引入 D2H/H2D、pin memory、CPU optimizer kernel 和参数回写边界

## 17. Eager / Compile / Graph：按 OP 特征选择
| 执行模式 | 适合放什么 | 主要风险 |
| --- | --- | --- |
| Eager | 动态控制、scheduler、fallback reference | 小 OP 链、Python 调度、隐式同步 |
| torch.compile | 稳定 OP 组合、局部图优化 | dynamic shape guard、graph break、模板 fallback |
| Graph | 固定 shape / 地址 / 路径的 decode replay | capture 不支持、allocator、新 tensor、低效路径被固定 |

## 18. 普通 Graph、Piecewise Graph 与 torch.compile
- 普通 Graph：整段固定路径 capture，收益大，约束最硬
- Piecewise Graph：把 attention、MLP/MoE、logits 等稳定片段分开 capture
- torch.compile：适合先优化稳定 PyTorch OP 组合，不等于一定命中高性能后端
- 三者可组合：compile 生成 callable，Graph 在固定 bucket 下 capture callable 执行
- 动态 Python 控制、tokenizer、I/O、日志、.item/.cpu 应留在 Graph 外

## 19. MUSA 视角下的 OP 排查五件事
| 排查项 | 代表 OP | 看什么 |
| --- | --- | --- |
| 后端命中 | matmul, softmax, SwishGLU | MuDNN / TileLang / custom / fallback |
| layout 契约 | view, reshape, contiguous | stride、contiguous、隐式 copy |
| dtype 支持 | to(fp16/bf16/fp8) | scale layout、量化 kernel 支持 |
| 同步边界 | item, cpu, wait, synchronize | host wait、D2H、stream 空洞 |
| Graph 安全 | copy_, graph.replay | capture 内 kernel、allocator、固定对象 |

## 20. 排查项与证据来源
- 单 OP 语义：附录 A 的完整代码与 MUSA stdout 能证明输入输出符合预期
- 框架位置：第 2-5 章源码链路能定位 OP 在控制面、metadata、fallback 或热路径
- Graph 稳定性：第 6/7 章最小用例能验证固定 shape、固定地址、replay 路径
- 后端命中：需要 profiler、后端日志和 kernel timeline 继续确认
- 性能结论：需要 benchmark、吞吐、延迟、显存和带宽数据补齐

## 21. MUSA 最小用例覆盖的推理场景
- 源码 OP 链：buffer copy、metadata copy、graph replay、HC fallback、SwiGLU clamp
- Prefill metadata：positions、req_pool_indices、out_cache_loc 展开
- Decode graph replay：固定 batch bucket、padding、copy_ 更新输入
- KV cache 写入：slot_mapping -> page_idx/page_offset -> cache write/read
- MoE routing 与 sampling：topk、softmax、combine、device-side argmax

## 22. 常见根因归纳
- 功能正确但后端错误：静默 CPU/Composite/PyTorch fallback
- shape 正确但 layout 错误：stride、contiguous、FP8 scale layout 不满足 kernel 契约
- eager 能跑但 Graph 不稳定：动态 shape、allocator、替换 tensor 对象、capture 不支持
- 同步 API 隐藏在热路径：item/tolist/cpu/synchronize/work.wait
- 通信功能可用但 overlap 失效：async collective 后立即 wait

## 23. 落地检查清单
- 先看 OP 语义：shape、dtype、device、stride、广播、in-place 行为是否符合预期
- 再看后端路径：MUSA native/custom/MuDNN/MCCL/TileLang 还是 fallback
- 再看框架位置：启动期、控制面、debug、fallback reference，还是每 token 热路径
- 再看执行模式：eager、compile、Graph、distributed overlap 是否满足同一组约束
- 最后看失败策略：不支持的 dtype/layout/shape 应 fail-closed 或受控 fallback

## 24. 结论
- 基于 OP 分类、源码链路和 MUSA 用例分析，OP 是连接模型语义、框架调度、后端 kernel、图捕获和硬件运行时的最小工程单元
- “能跑”只说明功能路径打通；“适合热路径”还要看 dispatch、layout、sync、dynamic shape、Graph 和后端质量
- SGLang / vLLM / Megatron 的共同模式是：PyTorch 保留语义，热路径交给后端
- MUSA 适配的核心不是 cuda -> musa 字符串替换，而是后端能力、Graph 兼容、layout 契约和同步边界的全链路验证

## 25. 关键判断口诀
- 能不能进 Graph：不看名字，看是否改变地址、产生动态 shape、触发 CPU 同步、走不支持 capture 的后端
- 性能瓶颈第一刀：不看大 GEMM，先看 copy ladder 和小 element-wise 链
- CPU/GPU 边界：device tensor 上的 item/tolist/cpu 是隐式 barrier
- MUSA 适配核心：JIT 机制、FP8 layout、capture 兼容性、clang/NVCC 方言差异都要验证
- OP 工程质量由真实框架链路中的 dispatch、内存语义、同步行为、图捕获能力和后端实现共同决定
