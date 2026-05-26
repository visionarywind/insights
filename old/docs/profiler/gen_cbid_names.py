#!/usr/bin/env python3
"""Generate CBID→name lookup tables from MUPTI headers."""
import re, sys
from pathlib import Path

def extract_cbids(header_path: str, enum_name: str, prefix: str) -> dict:
    """Extract CBID enum: {cbid_number: api_name}"""
    text = Path(header_path).read_text()
    cbid_to_name = {}
    # Match: PREFIX_someName   = number,
    pattern = re.compile(rf'{enum_name}_(\w+)\s*=\s*(\d+)')
    for m in pattern.finditer(text):
        name = m.group(1)
        num = int(m.group(2))
        if name == "INVALID":
            continue
        # Extract the API name from the CBID name
        # "musaDriverGetVersion_v3020" → "musaDriverGetVersion"
        # "muInit" → "muInit"
        # Strip version suffix like _v3020, _v2, _v3 etc.
        api_name = re.sub(r'_v\d+$', '', name)
        # If the original prefix is "mu" and api_name starts with "mu", keep as-is
        # If the prefix is "musa", the api_name already starts with "musa"
        cbid_to_name[num] = api_name
    return cbid_to_name

def generate(header_dir: str):
    driver_h = f"{header_dir}/mupti_driver_cbid.h"
    runtime_h = f"{header_dir}/mupti_runtime_cbid.h"

    driver_cbids = extract_cbids(driver_h, "MUPTI_DRIVER_TRACE_CBID", "mu")
    runtime_cbids = extract_cbids(runtime_h, "MUPTI_RUNTIME_TRACE_CBID", "musa")

    max_d = max(driver_cbids.keys()) if driver_cbids else 0
    max_r = max(runtime_cbids.keys()) if runtime_cbids else 0

    lines = []
    lines.append("// Auto-generated from MUPTI CBID headers. Do not edit.")
    lines.append("// Maps CBID number → authoritative API name from MUPTI enum.")
    lines.append("")
    lines.append(f"#define MAX_DRIVER_CBID {max_d}")
    lines.append(f"#define MAX_RUNTIME_CBID {max_r}")
    lines.append("")
    lines.append("static const char* kDriverCbidNames[] = {")
    for i in range(max_d + 1):
        name = driver_cbids.get(i, "unknown")
        lines.append(f'    /* {i:>4} */ "{name}",')
    lines.append("};")
    lines.append("")
    lines.append("static const char* kRuntimeCbidNames[] = {")
    for i in range(max_r + 1):
        name = runtime_cbids.get(i, "unknown")
        lines.append(f'    /* {i:>4} */ "{name}",')
    lines.append("};")
    lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    header_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/shanfeng/workspace/MUSA-Runtime/musa_shared_include/mupti"
    out = generate(header_dir)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cbid_names.inc"
    Path(out_path).write_text(out)
    print(f"Generated {out_path} ({len(out.split(chr(10)))} lines, {out.count('{')} entries)")
