# MUSA driver/API 性能建模（模拟）
对 3 个主流模型推理/训练进行 MUSA driver/API 性能建模，建立可与 profiling 数据对齐的性能预测与瓶颈分析能力

## KR1: 基于3个模型推理/训练建立性能模型，覆盖trace中累积耗时Top90%的driver API
## KR2: 建模结果与 profiling 数据对齐：Top50 kernel按累计耗时排序的 Top-K overlap达到90%以上
## KR3: 针对一个SDK版本发布3个模型的性能建模分析报告，包含 API 热点、kernel 热点和优化建议


# MUSA 行为分析 & CTS 沉淀（真实执行）
对 3 个主流模型推理/训练进行 MUSA行为分析，抽象模型行为并沉淀到 MUSA CTS。

## KR1: 针对dense和MoE模型抽象不少于5个行为特征用例，覆盖attention、MLP、KV cache，expert routing关键行为路径
## KR2: 建立MUSA行为用例，覆盖trace中出现的memory、stream/event、graph等关键API达到90%以上
## KR3: 建立3个模型推理/训练的MUSA行为基线，针对一个 SDK 版本发布行为分析报告，分析跟上一版本的执行性能差异点
