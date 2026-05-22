#!/usr/bin/env python3
"""
统一 PyTorch MUSA 输出格式
将所有 MUSA stdout 输出统一为：
    x = Tensor(shape=(2, 3), dtype=int64, device=musa:0)
        value=[[0, 0, 0], [0, 0, 0]]
"""

import re

def convert_output_line(line):
    """转换单行输出为统一格式"""
    line_stripped = line.strip()
    if not line_stripped:
        return None

    # 跳过简单格式（stride = (3, 1)，is_contiguous = False，offset = 2 等）
    if not ('shape=' in line_stripped and 'dtype=' in line_stripped):
        return line_stripped

    # 处理 fmt 函数格式（空格分隔）：
    # input_ids shape=(4,), dtype=int64, device=musa:0, value=[11, 12, 13, 0]
    fmt_pattern = r'^([\w.]+)\s+shape=\(([^)]*)\),\s+dtype=([\w.]+),\s+device=([\w:]+),\s+value=(.+)$'
    fmt_match = re.match(fmt_pattern, line_stripped)
    if fmt_match:
        name, shape, dtype, device, value = fmt_match.groups()
        value = value.strip()
        # 检测 value 是否为简单值（单行，不需要换行）
        if not (value.startswith('[[') or value.startswith('[')):
            return f"{name} = Tensor(shape=({shape}), dtype={dtype}, device={device}, value={value})"
        else:
            return f"{name} = Tensor(shape=({shape}), dtype={dtype}, device={device})\n    value={value}"

    # 处理已有 Tensor 格式（等号连接）：
    # x = Tensor(shape=(2, 3), dtype=int64, device=musa:0, value=[[0, 0, 0], [0, 0, 0]])
    tensor_pattern = r'^([\w.]+)\s+=\s+Tensor\(shape=\(([^)]*)\),\s+dtype=([\w.]+),\s+device=([\w:]+)(?:,\s+value=(.+))?\)$'
    tensor_match = re.match(tensor_pattern, line_stripped)
    if tensor_match:
        name, shape, dtype, device, value = tensor_match.groups()
        if value:
            value = value.strip()
            if not (value.startswith('[[') or value.startswith('[')):
                return f"{name} = Tensor(shape=({shape}), dtype={dtype}, device={device}, value={value})"
            else:
                return f"{name} = Tensor(shape=({shape}), dtype={dtype}, device={device})\n    value={value}"
        else:
            return f"{name} = Tensor(shape=({shape}), dtype={dtype}, device={device})"

    # 无法识别的格式，保持原样
    return line_stripped

def process_file(input_path):
    """处理整个文件"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 找到 MUSA 运行结果标记
        if 'MUSA 运行结果' in line and 'stdout' in line:
            # 保留标记行
            output_lines.append(line)
            i += 1

            # 跳过空行
            if i < len(lines) and lines[i].strip() == '':
                output_lines.append(lines[i])
                i += 1

            # 找到 ```text 开始
            if i < len(lines) and lines[i].strip() == '```text':
                output_lines.append(lines[i])  # ```text
                i += 1

                # 收集所有 tensor 输出行，直到遇到 ```
                while i < len(lines) and lines[i].strip() != '```':
                    converted = convert_output_line(lines[i])
                    if converted:
                        # 处理多行格式（已转换的格式可能包含换行符）
                        for converted_line in converted.split('\n'):
                            output_lines.append(converted_line)
                    i += 1

                # 保留结束标记
                if i < len(lines):
                    output_lines.append(lines[i])  # ```
                i += 1
        else:
            output_lines.append(line)
            i += 1

    return '\n'.join(output_lines)

if __name__ == '__main__':
    input_file = '/home/mtuser/workspace/Insights/Pytorch/PyTorch_Ops_SGLang_DeepSeekV4_MUSA_Consolidated_Guide.md'

    # 读取原文件
    with open(input_file, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 备份
    backup_file = input_file + '.backup_v3'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"已备份原文件到 {backup_file}")

    # 处理
    result = process_file(input_file)

    # 写回文件
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"处理完成，已保存到 {input_file}")
    print(f"转换行数统计：")
    print(f"  - 原始文件行数: {len(original_content.split(chr(10)))}")
    print(f"  - 处理后行数: {len(result.split(chr(10)))}")
