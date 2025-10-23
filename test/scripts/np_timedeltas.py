#!/usr/bin/env python
import numpy as np

# Base periods to generate from, as (value, base_unit)
# Avoid calendar 'Y'/'M' as base for numpy timedelta (ambiguous); handle those separately in tests.
base_periods = [
    (-1_000_000_000, "ns"),
    (-1_000_000, "us"),
    (-1_000, "ms"),
    (-3600, "s"),
    (-60, "m"),
    (-1, "h"),
    (-1, "D"),
    (-1, "W"),
    (0, "ns"),
    (1, "ns"),
    (1, "us"),
    (1, "ms"),
    (1, "s"),
    (60, "s"),
    (3600, "s"),
    (1, "m"),
    (1, "h"),
    (1, "D"),
    (7, "D"),
    (2, "W"),
    (1_000, "ms"),
    (1_000_000, "us"),
    (1_000_000_000, "ns"),
]

# Target units to cast to, including sub-ns
targets = ["W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"]

# Output format:
# "<VALUE> <BASE_UNIT> -> <TARGET_UNIT> <INT_VALUE>"
for v, b in base_periods:
    tb = np.timedelta64(v, b)
    for t in targets:
        try:
            out = int(tb.astype(f"timedelta64[{t}]").astype("int64"))
            print(f"{v} {b} -> {t} {out}")
        except Exception:
            # Some casts may be invalid in older numpy; skip
            pass

# Calendar-like units for numpy timedelta: 'M' (months) and 'Y' (years) exist, but semantics differ.
# We only provide identity conversions here to validate unit-count semantics where appropriate.
for v in [-100, -12, -1, 0, 1, 12, 100]:
    print(f"{v} M -> M {v}")
for v in [-100, -1, 0, 1, 100]:
    print(f"{v} Y -> Y {v}")
