"""Print the README's results table from the saved 120-cell ablation aggregate.

Median over seeds 0-2 of development macro-F1 (trusted_val), severe-class 3_00_0 F1,
and mean forgetting, per encoder arm and learner. No model is loaded or run.
Usage: python scripts/make_tables.py
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

CELLS = Path(__file__).resolve().parents[1] / "ssl_hdc_pilot/results/mechanism_ablation/aggregate_all_120/cells.csv"
ENCODER = {"s": "ViT-S/16", "b": "ViT-B/16", "l": "ViT-L/16"}
MEMORY = {"median_ncm": "all samples"}  # every other learner keeps one vector per class


def arm_parts(arm):
    """'p3_b_adapted_d2000' -> ('ViT-B/16', 'adapted', 'hdc D=2000' suffix)."""
    _phase, enc, regime, *dim = arm.split("_")
    return ENCODER[enc], regime, f" D={dim[0][1:]}" if dim else ""


groups = defaultdict(list)
with open(CELLS, newline="") as f:
    for row in csv.DictReader(f):
        encoder, regime, dim = arm_parts(row["arm"])
        learner = row["learner"] + (dim or (" D=10000" if row["learner"] == "hdc" else ""))
        groups[(encoder, regime, learner)].append(row)

ORDER = ["ncm", "ranpac", "sign_ncm", "median_ncm", "hdc D=1000", "hdc D=2000", "hdc D=10000"]
key = lambda g: (list(ENCODER.values()).index(g[0]), g[1], ORDER.index(g[2]))  # noqa: E731

print("| Encoder | Regime | Learner | Memory | Macro-F1 | 3_00_0 F1 | Mean forgetting |")
print("|---|---|---|---|---|---|---|")
for (encoder, regime, learner), rows in sorted(groups.items(), key=lambda kv: key(kv[0])):
    med = lambda k: statistics.median(float(r[k]) for r in rows)  # noqa: E731
    memory = MEMORY.get(learner, "1 vector/class")
    print(f"| {encoder} | {regime} | {learner} | {memory} | {med('macro_f1'):.3f} "
          f"| {med('f1__3_00_0'):.3f} | {med('mean_forgetting'):.3f} |")
