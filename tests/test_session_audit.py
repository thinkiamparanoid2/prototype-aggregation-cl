"""session_audit.py on a tiny synthetic DACON layout."""
import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "session_audit.py"
HEADER = "측정시각,내부 온도 1 평균\n"


def _sample(root, split, sid, stamps):
    d = root / split / sid
    d.mkdir(parents=True)
    (d / f"{sid}.csv").write_text(HEADER + "".join(f"{t},20.0\n" for t in stamps), encoding="utf-8")


def _split_csv(root, split, rows):
    with open(root / f"{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        w.writerows(rows)


def test_build_then_overlap_counts_shared_sessions_and_lookup(tmp_path):
    raw = tmp_path / "raw"
    a = ["2021-07-01 08:00:00", "2021-07-01 08:10:00"]
    b = ["2021-07-02 08:00:00", "2021-07-02 08:10:00"]
    c = ["2021-07-03 08:00:00", "2021-07-03 08:10:00"]
    _sample(raw, "train", "1", a)
    _sample(raw, "train", "2", b)
    _sample(raw, "val", "3", a)   # same session as train 1
    _sample(raw, "test", "4", b)  # same session as train 2
    _sample(raw, "test", "5", c)  # unseen session
    keys = tmp_path / "keys.csv"
    subprocess.run([sys.executable, str(SCRIPT), "build", str(raw), str(keys)], check=True)

    _split_csv(tmp_path, "train", [("1", "x"), ("2", "y")])
    _split_csv(tmp_path, "val", [("3", "x")])
    _split_csv(tmp_path, "test", [("4", "z"), ("5", "y")])
    out = subprocess.run([sys.executable, str(SCRIPT), "overlap", str(keys), str(tmp_path)],
                         check=True, capture_output=True, text=True, encoding="utf-8").stdout
    assert "val: 1/1 = 100.0% share a session with train; session-lookup accuracy 100.0%" in out
    assert "test: 1/2 = 50.0% share a session with train; session-lookup accuracy 0.0%" in out
