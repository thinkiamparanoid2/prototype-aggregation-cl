"""Session-leakage audit for the DACON crop-disease data.

Each DACON sample ships an image and the greenhouse sensor log recorded around it. Photos taken
during one recording share that log, so they are near-duplicates of the same plants and
conditions. Two session keys per sample:
  exact  = SHA-1 of the raw <id>/<id>.csv sensor log
  window = first and last timestamp of that log

  python scripts/session_audit.py build <raw Data dir with train/ val/ test/> <session_keys.csv>
  python scripts/session_audit.py overlap <session_keys.csv> <dir with train.csv val.csv test.csv>

`overlap` reports, for each non-train split, the share of samples whose session also appears in
train. Split CSVs need an `image` column holding the DACON sample id. When they also have a
`label` column it adds a no-learning baseline: predict the majority train label of the sample's
session (a sample whose session is unseen counts as wrong).
"""
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


def build(raw_root, out):
    rows = []
    for split in ("train", "val", "test"):
        for d in sorted((raw_root / split).iterdir()):
            raw = (d / f"{d.name}.csv").read_bytes()
            stamps = [ln.split(",", 1)[0] for ln in raw.decode("utf-8-sig").strip().splitlines()[1:]]
            rows.append({"image": d.name, "split": split, "exact": hashlib.sha1(raw).hexdigest(),
                         "window": f"{min(stamps)}|{max(stamps)}"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} keys to {out}")


def overlap(keys_csv, split_dir):
    with open(keys_csv, newline="") as f:
        keys = {r["image"]: r for r in csv.DictReader(f)}

    def rows(split):
        with open(split_dir / f"{split}.csv", newline="") as f:
            return [(r["image"], r.get("label")) for r in csv.DictReader(f)]

    train = rows("train")
    for key in ("exact", "window"):
        labels = defaultdict(Counter)
        for i, label in train:
            labels[keys[i][key]][label] += 1
        print(f"[{key}] sessions in train: {len(labels)}")
        for split in ("val", "test"):
            mine = rows(split)
            shared = sum(keys[i][key] in labels for i, _ in mine)
            line = f"  {split}: {shared}/{len(mine)} = {100 * shared / len(mine):.1f}% share a session with train"
            if mine[0][1] is not None:
                hits = sum(labels[keys[i][key]].most_common(1)[0][0] == label
                           for i, label in mine if keys[i][key] in labels)
                line += f"; session-lookup accuracy {100 * hits / len(mine):.1f}%"
            print(line)


if __name__ == "__main__":
    cmd, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    build(a, b) if cmd == "build" else overlap(a, b)
