# PyTorch OP 大模型推理系列

本目录把原始长文 `PyTorch_OP_DeepSeekV4_MUSA_Sharing_Guide.md` 拆成三篇长文。拆分依据是内容边界，不是压缩改写：原文中的主线分析、源码注释、MUSA 验证输出和完整 OP 用例都保留在三篇文章中。

## 阅读顺序

1. [`01_pytorch_op_fundamentals.md`](01_pytorch_op_fundamentals.md)：PyTorch OP 分类、性能问题、基础张量组织类 OP 用例。
2. [`02_transformer_execution_op_cases.md`](02_transformer_execution_op_cases.md)：Transformer OP 主线、核心计算 OP、CPU/同步/动态形状类用例。
3. [`03_deepseek_sglang_musa_practice.md`](03_deepseek_sglang_musa_practice.md)：DeepSeek V4-Pro、SGLang、Graph 用例、MUSA 模块验证和总结。

## 原文映射

| 新文件 | 对应原文章节 | 内容范围 |
|--------|--------------|----------|
| `01_pytorch_op_fundamentals.md` | §1-§2、附录 A.1.1-A.1.5 | OP 分类、注意事项、创建/更新/layout/索引/序列组织类 OP |
| `02_transformer_execution_op_cases.md` | §3、附录 A.1.6-A.4 | Transformer 主线、数学激活、线性代数、dtype/device、CPU、同步、动态形状 |
| `03_deepseek_sglang_musa_practice.md` | §4-§5、附录 A.5、附录 B、附录 C、§6 | DeepSeek V4-Pro、SGLang、Graph、MUSA 验证、回顾总结 |
