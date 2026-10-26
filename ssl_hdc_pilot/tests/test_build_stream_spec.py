from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ssl_hdc_pilot.scripts.build_stream_spec import build_stream_spec
from ssl_hdc_pilot.scripts import stream_runner


PROVENANCE = {
    "split_partition": "train",
    "split_policy": "option-a-19class",
    "split_seed": "20260716",
    "supervisor_approved": "true",
    "provenance_version": "ssl-hdc-trusted-split-v1",
    "source_csv": "Data/test.csv",
}


def _labels() -> list[str]:
    healthy = [f"{crop}_00_0" for crop in range(1, 5)]
    disease = [f"1_a{index}_1" for index in range(1, 16)]
    return [*healthy, *disease]


def _write_approved_option_a(root: Path) -> None:
    root.mkdir()
    labels = _labels()
    retained_counts = {label: 16 for label in labels}
    retained_counts.update(
        {
            "1_a1_1": 10,
            "1_a2_1": 10,
            "1_a3_1": 12,
            "1_a4_1": 14,
        }
    )
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [], "val": [], "test": []
    }
    for label in labels:
        train_count = 4 if label in {"1_a1_1", "1_a2_1", "1_a3_1", "1_a4_1"} else 8
        remaining = retained_counts[label] - train_count
        counts = {"train": train_count, "val": remaining // 2, "test": remaining - remaining // 2}
        for partition, count in counts.items():
            for index in range(count):
                prefix = "" if partition == "train" else f"{partition}_"
                image_id = f"{prefix}{label}_{index:02d}"
                split_rows[partition].append(
                    {
                        "image": image_id,
                        "label": label,
                        **PROVENANCE,
                        "split_partition": partition,
                        "image_path": f"Data/test/{image_id}/{image_id}.jpg",
                    }
                )
    fieldnames = ["image", "label", *PROVENANCE, "image_path"]
    for partition, rows in split_rows.items():
        with (root / f"{partition}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "policy": "option-a-19class",
        "seed": 20260716,
        "retained_class_count": 19,
        "retained_count": sum(retained_counts.values()),
        "split_counts": {key: len(value) for key, value in split_rows.items()},
    }
    (root / "distribution_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_generator_applies_locked_novel_holdback_seed_and_threshold_rules(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)

    first = build_stream_spec(
        split_dir=split_dir,
        output_path=tmp_path / "first.json",
        summary_log_path=tmp_path / "first.log",
    )
    second = build_stream_spec(
        split_dir=split_dir,
        output_path=tmp_path / "second.json",
        summary_log_path=tmp_path / "second.log",
    )

    protocol = first["protocol"]
    assert [item["class_label"] for item in protocol["class_introductions"]] == [
        "1_a4_1",
        "1_a3_1",
        "1_a1_1",
        "1_a2_1",
    ]
    assert set(protocol["initially_known_classes"]) == set(_labels()) - {
        "1_a1_1",
        "1_a2_1",
        "1_a3_1",
        "1_a4_1",
    }
    assert [item["at_step"] for item in protocol["class_introductions"]] == [
        9,
        18,
        27,
        36,
    ]
    assert first["seeds"]["stream_order"] == [0, 1, 2]
    assert protocol["confidence_thresholds"] == [0.05, 0.1, 0.2]
    assert protocol["label_budget"]["count_low_confidence"] is True
    assert protocol["label_budget"]["count_novel_registration"] is True
    assert first["baseline"]["ewc_decay"] == 0.9
    assert first["baseline"]["ewc_lambda"] == 10.0
    assert first == second

    manifests = protocol["bootstrap"]["held_back_image_ids_by_seed"]
    for seed in ("0", "1", "2"):
        held_back = manifests[seed]
        assert len(held_back) == 30
        counts = {label: 0 for label in protocol["initially_known_classes"]}
        for image_id in held_back:
            label = image_id.rsplit("_", 1)[0]
            counts[label] += 1
        assert set(counts.values()) == {2}
    assert manifests["0"] != manifests["1"]


def test_generator_prints_and_logs_same_human_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    summary_path = tmp_path / "summary.log"

    build_stream_spec(
        split_dir=split_dir,
        output_path=tmp_path / "stream_spec.json",
        summary_log_path=summary_path,
    )

    printed = capsys.readouterr().out
    assert printed == summary_path.read_text(encoding="utf-8")
    assert "Novel classes (entry order):" in printed
    assert "Stream seeds: 0, 1, 2" in printed
    assert "Confidence thresholds: 0.05, 0.10, 0.20" in printed


def test_generator_refuses_split_without_supervisor_approved_provenance(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "unapproved"
    _write_approved_option_a(split_dir)
    train_path = split_dir / "train.csv"
    text = train_path.read_text(encoding="utf-8").replace(
        ",true,ssl-hdc-trusted-split-v1,", ",false,ssl-hdc-trusted-split-v1,"
    )
    train_path.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="supervisor-approved provenance"):
        build_stream_spec(
            split_dir=split_dir,
            output_path=tmp_path / "stream_spec.json",
            summary_log_path=tmp_path / "summary.log",
        )


def test_generated_spec_is_accepted_by_the_official_runner_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    output = tmp_path / "stream_spec.json"
    build_stream_spec(
        split_dir=split_dir,
        output_path=output,
        summary_log_path=tmp_path / "summary.log",
    )
    monkeypatch.setattr(stream_runner, "DEFAULT_STREAM_SPEC", output)

    spec = stream_runner.load_stream_spec(output, smoke_mode=False)

    assert spec.stream_order_seeds == (0, 1, 2)
    assert spec.expected_stream_length == 46
    assert spec.bootstrap_confirmed_per_class is None
    assert spec.held_back_image_ids_by_seed is not None


def test_official_runner_refuses_legacy_hand_authored_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    output = tmp_path / "stream_spec.json"
    build_stream_spec(
        split_dir=split_dir,
        output_path=output,
        summary_log_path=tmp_path / "summary.log",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["protocol"].pop("stream_length")
    payload["protocol"]["bootstrap"] = {
        "confirmed_per_class": 1,
        "counts_toward_budget": False,
    }
    payload["protocol"]["label_budget"] = {"max_total_requests": 100}
    output.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(stream_runner, "DEFAULT_STREAM_SPEC", output)

    with pytest.raises(RuntimeError, match="generated Section 6a"):
        stream_runner.load_stream_spec(output, smoke_mode=False)


def test_official_runner_recomputes_locked_novel_class_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    output = tmp_path / "stream_spec.json"
    build_stream_spec(
        split_dir=split_dir,
        output_path=output,
        summary_log_path=tmp_path / "summary.log",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    displaced_novel = payload["protocol"]["class_introductions"][2]["class_label"]
    replacement_known = "1_a5_1"
    payload["protocol"]["initially_known_classes"].remove(replacement_known)
    payload["protocol"]["initially_known_classes"].append(displaced_novel)
    payload["protocol"]["initially_known_classes"].sort()
    payload["protocol"]["class_introductions"][2]["class_label"] = replacement_known
    output.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(stream_runner, "DEFAULT_STREAM_SPEC", output)
    monkeypatch.setattr(stream_runner, "DEFAULT_APPROVED_SPLIT_DIR", split_dir)
    spec = stream_runner.load_stream_spec(output, smoke_mode=False)
    rows = stream_runner._load_split_rows(split_dir / "train.csv", smoke_mode=False)

    with pytest.raises(RuntimeError, match="rarity rules"):
        stream_runner._validate_official_section6a(
            spec=spec, split_csv=split_dir / "train.csv", split_rows=rows
        )


def test_official_runner_recomputes_deterministic_per_class_holdback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    output = tmp_path / "stream_spec.json"
    build_stream_spec(
        split_dir=split_dir,
        output_path=output,
        summary_log_path=tmp_path / "summary.log",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    held = payload["protocol"]["bootstrap"]["held_back_image_ids_by_seed"]["0"]
    removed = next(image_id for image_id in held if image_id.startswith("1_00_0_"))
    replacement = next(
        f"2_00_0_{index:02d}" for index in range(8) if f"2_00_0_{index:02d}" not in held
    )
    held[held.index(removed)] = replacement
    output.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(stream_runner, "DEFAULT_STREAM_SPEC", output)
    monkeypatch.setattr(stream_runner, "DEFAULT_APPROVED_SPLIT_DIR", split_dir)
    spec = stream_runner.load_stream_spec(output, smoke_mode=False)
    rows = stream_runner._load_split_rows(split_dir / "train.csv", smoke_mode=False)

    with pytest.raises(RuntimeError, match="25% per-known-class"):
        stream_runner._validate_official_section6a(
            spec=spec, split_csv=split_dir / "train.csv", split_rows=rows
        )


def test_official_runner_refuses_noncanonical_approved_train_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir = tmp_path / "approved"
    _write_approved_option_a(split_dir)
    output = tmp_path / "stream_spec.json"
    build_stream_spec(
        split_dir=split_dir,
        output_path=output,
        summary_log_path=tmp_path / "summary.log",
    )
    monkeypatch.setattr(stream_runner, "DEFAULT_STREAM_SPEC", output)
    monkeypatch.setattr(stream_runner, "DEFAULT_APPROVED_SPLIT_DIR", split_dir)
    spec = stream_runner.load_stream_spec(output, smoke_mode=False)
    rows = stream_runner._load_split_rows(split_dir / "train.csv", smoke_mode=False)

    with pytest.raises(RuntimeError, match="canonical approved Option-A"):
        stream_runner._validate_official_section6a(
            spec=spec, split_csv=tmp_path / "copied_train.csv", split_rows=rows
        )
