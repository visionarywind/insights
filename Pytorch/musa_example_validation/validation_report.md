# MUSA Example Validation Report

## Summary

- Document: `Insights/Pytorch/PyTorch_OP_DeepSeekV4_MUSA_Sharing_Guide.md`
- Environment: remote `mochi-sglang` container on MUSA host
- Runnable Python examples: 81
- Passed: 81
- Failed: 0
- Validation method: extracted Python code fences whose first non-empty line is `import torch`, prepended `import torch_musa`, and executed each example in an isolated Python process with a 45s timeout.

## Artifacts

- `results.json`: per-example return code, stdout, stderr, and source line range.
- `summary.txt`: compact pass/fail summary.
- `progress.txt`: final progress snapshot from the remote run.

## Result

All extracted runnable MUSA examples passed.

## Source Line Ranges

- Example 1: lines 696-818, returncode=0
- Example 2: lines 843-873, returncode=0
- Example 3: lines 891-928, returncode=0
- Example 4: lines 946-968, returncode=0
- Example 5: lines 987-1007, returncode=0
- Example 6: lines 1025-1046, returncode=0
- Example 7: lines 1167-1197, returncode=0
- Example 8: lines 1212-1268, returncode=0
- Example 9: lines 1284-1306, returncode=0
- Example 10: lines 1417-1420, returncode=0
- Example 11: lines 1438-1442, returncode=0
- Example 12: lines 1459-1463, returncode=0
- Example 13: lines 1480-1483, returncode=0
- Example 14: lines 1499-1502, returncode=0
- Example 15: lines 1518-1521, returncode=0
- Example 16: lines 1545-1550, returncode=0
- Example 17: lines 1567-1574, returncode=0
- Example 18: lines 1593-1597, returncode=0
- Example 19: lines 1613-1617, returncode=0
- Example 20: lines 1633-1638, returncode=0
- Example 21: lines 1663-1667, returncode=0
- Example 22: lines 1684-1688, returncode=0
- Example 23: lines 1707-1711, returncode=0
- Example 24: lines 1728-1732, returncode=0
- Example 25: lines 1749-1753, returncode=0
- Example 26: lines 1772-1776, returncode=0
- Example 27: lines 1795-1799, returncode=0
- Example 28: lines 1816-1820, returncode=0
- Example 29: lines 1837-1841, returncode=0
- Example 30: lines 1858-1862, returncode=0
- Example 31: lines 1885-1889, returncode=0
- Example 32: lines 1906-1912, returncode=0
- Example 33: lines 1931-1936, returncode=0
- Example 34: lines 1954-1959, returncode=0
- Example 35: lines 1977-1982, returncode=0
- Example 36: lines 2000-2006, returncode=0
- Example 37: lines 2024-2027, returncode=0
- Example 38: lines 2049-2052, returncode=0
- Example 39: lines 2068-2072, returncode=0
- Example 40: lines 2088-2094, returncode=0
- Example 41: lines 2113-2118, returncode=0
- Example 42: lines 2136-2141, returncode=0
- Example 43: lines 2159-2163, returncode=0
- Example 44: lines 2182-2186, returncode=0
- Example 45: lines 2209-2213, returncode=0
- Example 46: lines 2230-2234, returncode=0
- Example 47: lines 2251-2255, returncode=0
- Example 48: lines 2272-2277, returncode=0
- Example 49: lines 2295-2298, returncode=0
- Example 50: lines 2314-2317, returncode=0
- Example 51: lines 2333-2336, returncode=0
- Example 52: lines 2352-2355, returncode=0
- Example 53: lines 2371-2374, returncode=0
- Example 54: lines 2390-2393, returncode=0
- Example 55: lines 2409-2412, returncode=0
- Example 56: lines 2428-2431, returncode=0
- Example 57: lines 2447-2450, returncode=0
- Example 58: lines 2472-2478, returncode=0
- Example 59: lines 2497-2502, returncode=0
- Example 60: lines 2520-2525, returncode=0
- Example 61: lines 2543-2548, returncode=0
- Example 62: lines 2566-2571, returncode=0
- Example 63: lines 2589-2593, returncode=0
- Example 64: lines 2611-2614, returncode=0
- Example 65: lines 2631-2634, returncode=0
- Example 66: lines 2650-2653, returncode=0
- Example 67: lines 2675-2679, returncode=0
- Example 68: lines 2696-2700, returncode=0
- Example 69: lines 2717-2721, returncode=0
- Example 70: lines 2738-2742, returncode=0
- Example 71: lines 2779-2783, returncode=0
- Example 72: lines 2802-2806, returncode=0
- Example 73: lines 2823-2827, returncode=0
- Example 74: lines 2846-2850, returncode=0
- Example 75: lines 2868-2920, returncode=0
- Example 76: lines 2964-2968, returncode=0
- Example 77: lines 2983-3030, returncode=0
- Example 78: lines 3059-3074, returncode=0
- Example 79: lines 3095-3109, returncode=0
- Example 80: lines 3130-3145, returncode=0
- Example 81: lines 3225-3245, returncode=0
