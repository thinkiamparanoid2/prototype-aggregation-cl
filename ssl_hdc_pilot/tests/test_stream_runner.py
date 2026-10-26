from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssl_hdc_pilot.scripts.stream_runner import load_stream_spec, run_study
from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _write_smoke_study(root: Path, embedding_dim: int) -> tuple[Path, Path, Path]:
    labels = ["known_a"] * 3 + ["known_b"] * 3 + ["novel_c"] * 3
    image_ids = [f"image_{index}" for index in range(len(labels))]
    embeddings = np.zeros((len(labels), embedding_dim), dtype=np.float32)
    for index, label in enumerate(labels):
        class_index = {"known_a": 0, "known_b": 1, "novel_c": 2}[label]
        embeddings[index, class_index] = 4.0
        embeddings[index, 3] = float(index) / 100.0

    embedding_path = root / f"embeddings_{embedding_dim}.npz"
    np.savez_compressed(
        embedding_path,
        image_ids=np.asarray(image_ids, dtype=np.str_),
        embeddings=embeddings,
    )
    split_path = root / "trusted_smoke.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(
            {"image": image_id, "label": label}
            for image_id, label in zip(image_ids, labels, strict=True)
        )

    spec = {
        "schema_version": 1,
        "tag": f"synthetic_dim_{embedding_dim}",
        "smoke": True,
        "execution": {"device": "cpu"},
        "seeds": {
            "stream_order": [17],
            "projection": 19,
            "learner": 23,
            "replay": 29,
        },
        "protocol": {
            "initially_known_classes": ["known_a", "known_b"],
            "class_introductions": [{"class_label": "novel_c", "at_step": 2}],
            "bootstrap": {"confirmed_per_class": 1, "counts_toward_budget": True},
            "confidence_thresholds": [2.0],
            "label_budget": {"max_total_requests": 20},
            "rolling_window": 3,
        },
        "hdc": {"hypervector_dim": 10_000},
        "baseline": {
            "head_type": "linear",
            "hidden_dim": None,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "update_steps": 1,
            "ewc_lambda": 10.0,
            "ewc_decay": 1.0,
            "replay_buffer_size": 3,
            "replay_batch_size": 2,
        },
        "metrics": {
            "stabilization_window": 2,
            "stabilization_min_accuracy": 0.5,
            "stabilization_consecutive_windows": 1,
            "ece_bins": 5,
        },
    }
    spec_path = root / "synthetic_stream_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return embedding_path, split_path, spec_path


def test_official_path_refuses_missing_or_smoke_stream_spec(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stream spec is required"):
        load_stream_spec(tmp_path / "stream_spec.json", smoke_mode=False)

    _, _, smoke_spec = _write_smoke_study(tmp_path, 384)
    with pytest.raises(RuntimeError, match="official path refuses smoke"):
        load_stream_spec(smoke_spec, smoke_mode=False)


def test_prototype_only_path_still_refuses_a_missing_stream_spec(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stream spec is required"):
        run_study(
            embeddings_path=tmp_path / "missing_embeddings.npz",
            split_csv=tmp_path / "missing_split.csv",
            stream_spec_path=tmp_path / "missing_spec.json",
            output_dir=tmp_path / "refused",
            smoke_mode=False,
            learner_names=("ncm", "ranpac"),
        )


def test_runner_refuses_labels_embedded_in_npz(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 384)
    with np.load(embeddings, allow_pickle=False) as payload:
        image_ids = payload["image_ids"].copy()
        vectors = payload["embeddings"].copy()
    np.savez_compressed(
        embeddings,
        image_ids=image_ids,
        embeddings=vectors,
        labels=np.asarray(["hidden"] * len(image_ids)),
    )

    with pytest.raises(RuntimeError, match="label-free"):
        run_study(
            embeddings_path=embeddings,
            split_csv=split,
            stream_spec_path=spec,
            output_dir=tmp_path / "refused",
            smoke_mode=True,
            clock=_StepClock(),
        )


def test_spec_accepts_multiple_classes_activated_at_one_step(tmp_path: Path) -> None:
    _, _, spec_path = _write_smoke_study(tmp_path, 384)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["protocol"]["class_introductions"].append(
        {"class_label": "novel_d", "at_step": 2}
    )
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    spec = load_stream_spec(spec_path, smoke_mode=True)

    assert [item.at_step for item in spec.class_introductions] == [2, 2]


def test_every_preregistered_stream_order_seed_is_run(tmp_path: Path) -> None:
    embeddings, split, spec_path = _write_smoke_study(tmp_path, 384)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["seeds"]["stream_order"] = [17, 18]
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    artifacts = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec_path,
        output_dir=tmp_path / "seed_sweep",
        smoke_mode=True,
        clock=_StepClock(),
    )

    summaries = [
        json.loads(artifact.summary_json.read_text(encoding="utf-8"))
        for artifact in artifacts
    ]
    assert len(artifacts) == 12
    assert {summary["stream_seed"] for summary in summaries} == {17, 18}


def test_runner_can_select_only_ewc_cells(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 384)

    artifacts = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec,
        output_dir=tmp_path / "ewc_only",
        smoke_mode=True,
        ewc_only=True,
        clock=_StepClock(),
    )

    assert len(artifacts) == 1
    summary = json.loads(artifacts[0].summary_json.read_text(encoding="utf-8"))
    assert summary["learner"] == "ewc"
    assert "__ewc__" in artifacts[0].run_tag


def test_runner_can_select_only_new_prototype_cells(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 384)

    artifacts = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec,
        output_dir=tmp_path / "prototypes_only",
        smoke_mode=True,
        learner_names=("ncm", "ranpac"),
        clock=_StepClock(),
    )

    summaries = [
        json.loads(artifact.summary_json.read_text(encoding="utf-8"))
        for artifact in artifacts
    ]
    assert len(artifacts) == 2
    assert {summary["learner"] for summary in summaries} == {"ncm", "ranpac"}
    assert all(
        "__ncm__" in item.run_tag or "__ranpac__" in item.run_tag
        for item in artifacts
    )
    configurations = {
        summary["learner"]: summary["learner_configuration"]
        for summary in summaries
    }
    assert configurations["ncm"]["embedding_variant"] == "l2_normalized"
    assert configurations["ranpac"]["projection_dim"] == 10_000
    assert configurations["ranpac"]["activation"] == "relu"
    assert configurations["ranpac"]["decorrelation"] is False
    assert configurations["ranpac"]["ridge"] is False


@pytest.mark.parametrize(
    ("learner_names", "ewc_only", "message"),
    [
        (("unknown",), False, "unknown continual learners"),
        (("ncm", "ncm"), False, "must not contain duplicates"),
        (("ncm",), True, "cannot be combined"),
    ],
)
def test_runner_rejects_invalid_learner_selections(
    tmp_path: Path,
    learner_names: tuple[str, ...],
    ewc_only: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_study(
            embeddings_path=tmp_path / "unused_embeddings.npz",
            split_csv=tmp_path / "unused_split.csv",
            stream_spec_path=tmp_path / "unused_spec.json",
            output_dir=tmp_path / "unused_output",
            smoke_mode=True,
            learner_names=learner_names,
            ewc_only=ewc_only,
        )


def test_novel_registration_requests_and_counts_a_label_without_low_confidence(
    tmp_path: Path,
) -> None:
    embeddings, split, spec_path = _write_smoke_study(tmp_path, 384)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["protocol"]["confidence_thresholds"] = [0.0]
    payload["protocol"]["label_budget"] = {
        "max_total_requests": 20,
        "count_low_confidence": True,
        "count_novel_registration": True,
    }
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    artifacts = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec_path,
        output_dir=tmp_path / "novel_registration",
        smoke_mode=True,
        clock=_StepClock(),
    )

    history = list(csv.DictReader(artifacts[0].history_csv.open(encoding="utf-8")))
    novel_rows = [row for row in history if row["true_label"] == "novel_c"]
    assert novel_rows[0]["step"] == "2"
    assert novel_rows[0]["label_requested"] == "True"
    assert novel_rows[0]["class_registered"] == "True"


def test_fixed_spec_produces_byte_identical_semantic_outputs(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 384)

    first = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec,
        output_dir=tmp_path / "first",
        smoke_mode=True,
        clock=_StepClock(),
    )
    second = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec,
        output_dir=tmp_path / "second",
        smoke_mode=True,
        clock=_StepClock(),
    )

    assert [artifact.run_tag for artifact in first] == [artifact.run_tag for artifact in second]
    for left, right in zip(first, second, strict=True):
        assert left.history_csv.read_bytes() == right.history_csv.read_bytes()
        assert left.summary_json.read_bytes() == right.summary_json.read_bytes()


@pytest.mark.parametrize("embedding_dim", [384, 1024])
def test_all_learners_are_dim_agnostic_and_report_non_accuracy_metrics(
    tmp_path: Path, embedding_dim: int
) -> None:
    root = tmp_path / str(embedding_dim)
    root.mkdir()
    embeddings, split, spec = _write_smoke_study(root, embedding_dim)

    artifacts = run_study(
        embeddings_path=embeddings,
        split_csv=split,
        stream_spec_path=spec,
        output_dir=root / "results",
        smoke_mode=True,
        clock=_StepClock(),
    )

    summaries = [json.loads(artifact.summary_json.read_text(encoding="utf-8")) for artifact in artifacts]
    assert {summary["learner"] for summary in summaries} == {
        "hdc",
        "ncm",
        "ranpac",
        "naive",
        "ewc",
        "replay",
    }
    for summary in summaries:
        assert summary["embedding_dim"] == embedding_dim
        assert summary["final_evaluation_samples"] == 9
        assert "macro_f1" in summary["prequential_metrics"]
        assert "macro_f1" in summary["final_metrics"]
        assert "macro_recall" in summary["final_metrics"]
        assert "per_class_recall" in summary["final_metrics"]
        assert "forgetting" in summary
        assert "new_class_uptake" in summary
        assert "update_compute_seconds" in summary
        assert "label_budget" in summary


def test_opt_in_final_states_preserve_legacy_outputs_and_provenance(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 4)
    legacy = run_study(embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
                       output_dir=tmp_path / "legacy", smoke_mode=True, clock=_StepClock())
    preserved = run_study(embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
                          output_dir=tmp_path / "states", smoke_mode=True, clock=_StepClock(),
                          save_final_state=True)
    for old, new in zip(legacy, preserved, strict=True):
        assert old.final_state_npz is None
        assert old.history_csv.read_bytes() == new.history_csv.read_bytes()
        assert old.summary_json.read_bytes() == new.summary_json.read_bytes()
        checkpoint = load_learner_checkpoint(new.final_state_npz)
        summary = json.loads(new.summary_json.read_text())
        assert checkpoint.context["run_tag"] == new.run_tag
        assert checkpoint.context["stream_spec_sha256"] == summary["stream_spec_sha256"]
        assert checkpoint.context["progress"]["processed_stream_events"] == 7
        assert checkpoint.context["progress"]["labels_charged"] == 9
        assert checkpoint.context["progress"]["confirmed_novel_classes"] == ["novel_c"]
        assert len(checkpoint.context["input_hashes"]["embeddings_sha256"]) == 64
        assert checkpoint.context["code_hashes"]
        assert checkpoint.learner.known_classes
        assert checkpoint.learner.predict(torch.tensor([4., 0., 0., 0.])).label is not None
    assert len(list((tmp_path / "legacy").glob("*.npz"))) == 0
    assert len(list((tmp_path / "states").glob("*.npz"))) == 6


def test_existing_state_file_is_not_overwritten_by_runner(tmp_path: Path) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 4)
    output = tmp_path / "protected"
    output.mkdir()
    path = output / "synthetic_dim_4__embeddings_4__stream_17__hdc__margin_2_final_state.npz"
    path.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_study(embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
                  output_dir=output, smoke_mode=True, save_final_state=True)
    assert path.read_bytes() == b"existing artifact"
    assert list(output.iterdir()) == [path]
