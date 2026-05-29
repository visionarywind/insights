# Weekly Report Skill

## Scope

Use this skill when the user asks to generate, polish, or structure a weekly report in Chinese.

Typical triggers:

- 写周报
- 生成周报
- 整理周报
- 周报 skill
- 写本周工作总结
- 写T5T

## Output Format

Always use this structure unless the user asks for a different format:

```markdown
# My Top 5 Things

## My OKR

- **<OKR方向/职责域 1>**：<目标描述>，<关键衡量标准或验收目标>
- **<OKR方向/职责域 2>**：<目标描述>，<关键衡量标准或验收目标>
- **<OKR方向/职责域 3>**：<目标描述>，<关键衡量标准或验收目标>

## Executive Summary

<5 条以内本周最重要进展，每条建议包含：项目/平台/阶段 + 做了什么 + 当前状态/结果 + 风险或下一步>

# 未来计划

## 1. <下周/近期计划 1>

## 2. <下周/近期计划 2>

## 3. <下周/近期计划 3>

# 我要求助

<需要协同、资源、评审、机器、权限、排期或决策支持的事项；没有则写“暂无”>

# 情报分享及其他要点

<跨团队信息、风险提醒、技术观察、阻塞变化、CI/机器/版本状态、值得同步的上下文；没有则写“暂无”>
```

## Style

- `My OKR` 固定使用 bullet list，每条格式为 `- **方向/职责域**：目标描述，衡量标准`。
- OKR 方向建议使用短标签，例如 `GPU Driver问题闭环`、`musa_benchmarks性能看护`、`MTCC工具链洞察`。
- 使用中文。
- 保持周报风格：事实清晰、结果导向、可被 manager 快速扫读。
- 不写空泛表述，例如“持续推进”“积极沟通”，除非后面跟具体对象和结果。
- 每条 Executive Summary 尽量包含状态词：已完成、已合入、验证中、调试中、CI pending、对齐中、阻塞于。
- 项目前缀建议保留，例如 `[CrossTeam]`、`[CI]`。
- 如果用户给的是零散 bullet，主动归并为 Top 5，不要机械照抄。
- 如果信息不足，不要编造具体数据；用“待补充”标注。

## Content Heuristics

When organizing weekly content:

1. Prefer shipped/validated/merged work over exploratory work.
2. Prefer work with user/customer/CI impact over internal cleanup.
3. Group related items into one bullet if they belong to the same project.
4. Put blockers and asks under `我要求助` instead of burying them in Executive Summary.
5. Put general observations, risks, and cross-team context under `情报分享及其他要点`.

## Example

```markdown
# My Top 5 Things

## My OKR

- **性能建模**：完成driver软件建模分析，完成模型行为模式建模
- **行为抽象**：沉淀模型行为模式，完成模型行为模式建模

## Executive Summary

[MUPTI] 调研项目
[Driver] 引入设计

# 未来计划

1. 深入理解MUPTI

2. 开发ModelEvent

3. 沉淀cts

# 我要求助

暂无

# 情报分享及其他要点

暂无
```

## Interaction Pattern

If the user provides raw weekly notes, immediately rewrite them into the target format.

If the user only says “生成周报” and provides no notes, ask for one of the following inputs:

- 本周完成事项
- 下周计划
- 需要求助的事项
- 其他需要同步的信息

If local git history or task records are relevant and the user asks to “基于当前工作生成周报”, inspect the available local/remote records first, then draft the report. Do not invent confidential business results or performance numbers that are not present in the source material.
