from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "Pytorch" / "PyTorch_OP_DeepSeekV4_MUSA_Sharing_Guide.md"
OUT = Path(__file__).resolve().parent / "cases"


def slugify(title: str) -> str:
    title = re.sub(r"`([^`]+)`", r"\1", title)
    title = re.sub(r"[^0-9A-Za-z_.-]+", "_", title).strip("_")
    return title[:80] or "example"


def nearest_heading(text: str, pos: int) -> str:
    prefix = text[:pos].splitlines()
    for line in reversed(prefix):
        if line.startswith("##### ") or line.startswith("#### ") or line.startswith("### "):
            return line.lstrip("#").strip()
    return "example"


def expected_stdout(after: str) -> str | None:
    marker = "MUSA 运行结果"
    marker_idx = after.find(marker)
    if marker_idx < 0:
        return None
    fence_idx = after.find("```text", marker_idx)
    if fence_idx < 0:
        return None
    start = fence_idx + len("```text")
    end = after.find("```", start)
    if end < 0:
        return None
    return after[start:end].strip("\n")


def main() -> None:
    text = DOC.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()

    manifest = []
    pos = 0
    idx = 0
    while True:
        start = text.find("```python", pos)
        if start < 0:
            break
        code_start = start + len("```python")
        end = text.find("```", code_start)
        if end < 0:
            raise RuntimeError(f"unclosed python fence near {start}")
        code = text[code_start:end].strip("\n") + "\n"
        after = text[end : end + 2000]
        expected = expected_stdout(after)
        if expected is not None:
            idx += 1
            line = text[:start].count("\n") + 1
            heading = nearest_heading(text, start)
            stem = f"{idx:03d}_line{line}_{slugify(heading)}"
            script = OUT / f"{stem}.py"
            expect = OUT / f"{stem}.expected.txt"
            script.write_text(code)
            expect.write_text(expected + "\n")
            manifest.append(
                {
                    "id": idx,
                    "line": line,
                    "heading": heading,
                    "script": script.name,
                    "expected": expect.name,
                }
            )
        pos = end + 3

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"extracted {len(manifest)} runnable examples to {OUT}")


if __name__ == "__main__":
    main()
