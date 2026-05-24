from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent
CASES = BASE / "cases"
RESULTS = Path(os.environ.get("RESULTS_DIR", BASE / "results"))


def normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CASES / "manifest.json").read_text())
    if len(sys.argv) > 1:
        selected = {int(arg) for arg in sys.argv[1:]}
        manifest = [item for item in manifest if int(item["id"]) in selected]
    records = []
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    for item in manifest:
        script = CASES / item["script"]
        expected = normalize((CASES / item["expected"]).read_text())
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(BASE),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            status = "PASS" if proc.returncode == 0 and normalize(proc.stdout) == expected else "FAIL"
            stdout = normalize(proc.stdout)
            stderr = normalize(proc.stderr)
            returncode = proc.returncode
            error = ""
        except subprocess.TimeoutExpired as exc:
            status = "TIMEOUT"
            stdout = normalize(exc.stdout or "")
            stderr = normalize(exc.stderr or "")
            returncode = None
            error = "timeout"

        elapsed = round((time.time() - start) * 1000, 3)
        record = {
            **item,
            "status": status,
            "returncode": returncode,
            "elapsed_ms": elapsed,
            "stdout": stdout,
            "stderr": stderr,
            "expected": expected,
            "error": error,
        }
        records.append(record)
        (RESULTS / f"{item['id']:03d}.stdout.txt").write_text(stdout + ("\n" if stdout else ""))
        (RESULTS / f"{item['id']:03d}.stderr.txt").write_text(stderr + ("\n" if stderr else ""))
        print(f"{item['id']:03d} {status} line={item['line']} heading={item['heading']} elapsed_ms={elapsed}")

    summary = {
        "total": len(records),
        "pass": sum(r["status"] == "PASS" for r in records),
        "fail": sum(r["status"] == "FAIL" for r in records),
        "timeout": sum(r["status"] == "TIMEOUT" for r in records),
        "records": records,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    md = [
        "# PyTorch OP 文档用例 MUSA 验证结果",
        "",
        f"- Total: {summary['total']}",
        f"- PASS: {summary['pass']}",
        f"- FAIL: {summary['fail']}",
        f"- TIMEOUT: {summary['timeout']}",
        "",
        "| ID | Status | Line | Heading | Elapsed ms |",
        "|----|--------|------|---------|------------|",
    ]
    for r in records:
        md.append(f"| {r['id']:03d} | {r['status']} | {r['line']} | {r['heading']} | {r['elapsed_ms']} |")
    failing = [r for r in records if r["status"] != "PASS"]
    if failing:
        md += ["", "## Failures", ""]
        for r in failing:
            md += [
                f"### {r['id']:03d} line {r['line']} {r['heading']}",
                "",
                "Expected:",
                "```text",
                r["expected"],
                "```",
                "Actual stdout:",
                "```text",
                r["stdout"],
                "```",
                "Stderr:",
                "```text",
                r["stderr"],
                "```",
                "",
            ]
    (RESULTS / "validation_report.md").write_text("\n".join(md) + "\n")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
