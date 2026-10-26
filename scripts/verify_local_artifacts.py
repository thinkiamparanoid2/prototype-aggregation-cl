"""Read-only checks of migrated files; never trains or runs inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ssl_hdc_pilot.scripts.checkpoint_encoder import inspect_checkpoint_export
from transformers import AutoConfig, AutoImageProcessor


def main() -> None:
    pilot = ROOT / "ssl_hdc_pilot"
    assert hashlib.sha256((pilot / "stream_spec.json").read_bytes()).hexdigest() == (
        "62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc"
    )
    for partition, expected in (("train", 3396), ("val", 1132), ("test", 1132)):
        with (pilot / "trusted_splits" / f"{partition}.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected, (partition, len(rows))
        assert len({row["label"] for row in rows}) == 19
        assert all((ROOT / row["image_path"]).is_file() for row in rows)
        print(f"Trusted {partition}: {len(rows)} image paths resolve inside HDC", flush=True)

    with (pilot / "results" / "unlabeled_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 51906
    assert all((ROOT / row["image_path"]).is_file() for row in rows)
    print("Unlabeled pool: all 51,906 local image paths resolve", flush=True)

    results = pilot / "results"
    for rung, dimension in (("vits16", 384), ("vitb16", 768), ("vitl16", 1024)):
        model_id = f"facebook/dinov3-{rung}-pretrain-lvd1689m"
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        assert config.hidden_size == dimension
        AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        for epoch in range(6):
            provenance = inspect_checkpoint_export(results / f"arm_d_{rung}" / f"checkpoint_epoch{epoch}")
            assert provenance.epoch == epoch and provenance.model_id == model_id
        print(f"{rung}: offline model config/processor and all six checkpoint exports validated", flush=True)

    summaries = list((results / "continual").rglob("*_summary.json"))
    histories = list((results / "continual").rglob("*_history.csv"))
    assert len(summaries) == len(histories) == 108, (len(summaries), len(histories))
    for history in histories:
        with history.open(newline="", encoding="utf-8-sig") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == 976, history
    for summary in summaries:
        json.loads(summary.read_text(encoding="utf-8-sig"))
    print("Continual evidence: 108 summaries, 108 histories, 976 rows each", flush=True)
    print("LOCAL ARTIFACT CHECKS PASSED - no training or inference performed", flush=True)


if __name__ == "__main__":
    main()
