#!/usr/bin/env python
import numpy as np

datetimes = [
    "1969-12-31T23:00:00",
    "1969-12-31T23:59:59",
    "1970-01-01T00:00:00",
    "1970-01-01T00:00:01",
    "1970-01-01T01:00:00",
    "1999-12-31T23:59:59",
    "2000-02-29T12:34:56",
    "1900-01-01T00:00:00",
    "2100-01-01T00:00:00",
]

# Note: skip 'W' due to week-anchor semantics.
units = ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"]

# Output format: "<ISO_DATETIME> <UNIT> <INT_VALUE>"
for s in datetimes:
    for u in units:
        d = np.datetime64(s, u)
        print(f"{s} {u} {int(d.astype('int64'))}")
