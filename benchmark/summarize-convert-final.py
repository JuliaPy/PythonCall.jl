"""Aggregate three serial before/after rounds; keep per-trial TSVs as raw data."""
import csv
import statistics
import sys
from pathlib import Path

root = Path(__file__).parent
experiment = sys.argv[1] if len(sys.argv) > 1 else ""
common_prefix = "convert-final" + ("-" + experiment if experiment else "")
expanded_prefix = "convert-expanded-" + (experiment or "final")
fields = ("group", "case", "mode", "before_ns", "after_ns", "change_percent",
          "before_p50_ns", "after_p50_ns", "before_low_ns", "before_high_ns",
          "after_low_ns", "after_high_ns", "before_bytes", "before_allocs",
          "after_bytes", "after_allocs")
summary = []
for group, prefix in (("common", common_prefix), ("expanded", expanded_prefix)):
    data = {}
    for side in ("before", "after"):
        rounds = []
        environments = []
        for round_number in (1, 2, 3):
            stem = f"{prefix}-{side}-r{round_number}"
            with (root / f"{stem}.tsv").open() as source:
                rounds.append({(r["case"], r["mode"]): r for r in
                               csv.DictReader(source, delimiter="\t")
                               if r.get("status", "measured") == "measured"})
            environments.append((root / f"{stem}-environment.txt").read_text())
        assert all(e == environments[0] for e in environments), (group, side)
        assert all(set(r) == set(rounds[0]) for r in rounds), (group, side)
        data[side] = rounds
    keys = list(data["before"][0])
    keys.extend(k for k in data["after"][0] if k not in keys)
    for key in keys:
        result = dict.fromkeys(fields, "")
        result.update(group=group, case=key[0], mode=key[1])
        for side in ("before", "after"):
            if key not in data[side][0]:
                continue
            rows = [r[key] for r in data[side]]
            times = [float(r["min_ns"]) for r in rows]
            result[side + "_ns"] = statistics.median(times)
            result[side + "_p50_ns"] = statistics.median(float(r["median_ns"]) for r in rows)
            result[side + "_low_ns"] = min(times)
            result[side + "_high_ns"] = max(times)
            for field in ("bytes", "allocs"):
                assert len({r[field] for r in rows}) == 1, (group, side, key, field)
                result[side + "_" + field] = int(rows[0][field])
        if result["before_ns"] != "" and result["after_ns"] != "":
            result["change_percent"] = 100 * (result["after_ns"] / result["before_ns"] - 1)
        summary.append(result)

with (root / ("convert-" + (experiment or "final") + "-summary.tsv")).open("w") as destination:
    writer = csv.DictWriter(destination, fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
print(f"Summarized {len(summary)} cases/modes from three before/after rounds")
