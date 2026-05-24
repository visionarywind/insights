# Mochi SGLang MUSA Validation

- Date: 2026-05-24
- Host: `10.18.32.25`
- Container: `mochi-sglang`
- Image: `registry.mthreads.com/mcconline/inference/sglang:deepseek-v4-s5000-4.3.5-torch2.9.0-20260515`
- PyTorch: `2.9.0`
- Device: `MTT S5000`
- Device count: `8`

Validation command inside the container:

```bash
cd /tmp/musa_doc_validation_20260524
RESULTS_DIR=results_current python3 run_doc_examples.py
```

Result:

- Total examples: `81`
- PASS: `81`
- FAIL: `0`
- TIMEOUT: `0`

Full report: `results_current/validation_report.md`
