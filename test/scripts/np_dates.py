#!/usr/bin/env python
import numpy as np

dates = [
    "1969-12-31",
    "1970-01-01",
    "1970-01-02",
    "1999-12-31",
    "2000-02-29",
    "1900-01-01",
    "2100-01-01",
]

# Note: skip 'W' to avoid week-anchor semantics; test uses floor(days/7) semantics.
units = ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"]

# Output format: "<ISO_DATE> <UNIT> <INT_VALUE>"
for s in dates:
    for u in units:
        d = np.datetime64(s, u)
        print(f"{s} {u} {int(d.astype('int64'))}")
