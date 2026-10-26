"""Post-hoc, read-only aggregator for the forgetting-grid and mechanism-ablation
comparative artifacts.

This intentionally does not import or alter evaluate_snapshots.py,
stream_runner.py, budget_experiment.py, experiment_a.py, continual_hdc.py,
continual_prototypes.py, continual_baselines.py, or the stream spec, and it
never loads a learner and never scores anything. It only reads already-saved
manifest.json / metrics.json / integrity_report.json / label_events.json /
uptake-repair JSON files from disk. Model: uptake_repair.py, which establishes
the same pattern for the same artifact set.

There is no `per_class_f1` field anywhere in the saved artifacts. It is
derived here from the saved confusion matrix: for class i, TP = cm[i][i],
FN = rowsum(i) - TP, FP = colsum(i) - TP, F1 = 2*TP / (2*TP + FP + FN),
defined as 0.0 when that denominator is 0. See per_class_f1_from_confusion().

This script must never open ssl_hdc_pilot/trusted_splits/test.csv. Every file
read goes through _read_json/_sha256, both of which refuse that path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

DEFAULT_VALIDATION_ROOT = Path("ssl_hdc_pilot/results/longitudinal_validation_development")
DEFAULT_CAPTURE_ROOT = Path("ssl_hdc_pilot/results/budget_longitudinal_development")

# Fixed regardless of --validation-root: this is where uptake_repair.py's
# per-cell post-hoc novel-class files live, and their filenames match the
# validation/capture cell directory names exactly (see uptake_repair.py).
DEFAULT_UPTAKE_ROOT = Path("ssl_hdc_pilot/results/forgetting_grid_development/uptake_repair")

# Hard prohibition: never read the trusted test split, from this script or any
# path it is handed.
FORBIDDEN_TEST_SPLIT_PARTS = ("trusted_splits", "test.csv")

# A cell directory name has the shape:
#   <date>_<tag>_b<budget>_<arm...>_<learner...>_s<seed>_t<threshold>_attempt<NN>
# arm and learner cannot always be split apart from the name alone (an
# ablation learner such as "sign_ncm" also contains an underscore), so the
# split point is resolved against manifest.json's own "learner" field in
# parse_cell_identity() below, which is authoritative. This regex only
# recognises candidate cell directories among everything else living in the
# results directories (logs, plan files, archived pre-grid captures with a
# different naming shape, etc.).
CELL_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9]+_b\d+_[a-z0-9_]+_s\d+_t\d+_attempt\d+$")

# Supervision is matched at the stated budget for these three learners in
# every cell; naive/EWC/replay are not label-matched (see AGENTS.md /
# RESULTS_LOG.md) and must never share an unqualified comparison table with
# them. per_class.csv and the win-count/"best control" logic below therefore
# only ever compare within this set.
LABEL_MATCHED_LEARNERS = ("hdc", "ncm", "ranpac")
NOT_LABEL_MATCHED_LEARNERS = ("naive", "ewc", "replay")

SCALAR_FIELDS = (
    "macro_f1", "accuracy", "macro_recall", "ece",
    "mean_forgetting", "labels_consumed", "allowance",
)

# The recorded "adapted-arm advantage is concentrated in two classes" finding
# in RESULTS_LOG.md is specifically: HDC's per-class F1 (mean over seeds) on
# the adapted arm, against the per-class *best* of NCM/RanPAC-style (also
# mean over seeds) -- not a fixed single control learner. This reproduces the
# recorded +0.0635 / +0.0069 macro deltas; see test_aggregate_grid.py.
CONCENTRATION_ARM = "adapted"
CONCENTRATION_FOCUS_LEARNER = "hdc"
CONCENTRATION_CONTROL_LEARNERS = ("ncm", "ranpac")
CONCENTRATION_TOP_N = 2


class TrustedTestSplitAccessError(RuntimeError):
    """Raised if anything in this module is asked to open the trusted test split."""


def _guard_no_test_split(path: Path) -> None:
    parts = path.parts
    if any(parts[i:i + 2] == FORBIDDEN_TEST_SPLIT_PARTS for i in range(len(parts) - 1)) or path.name == "test.csv":
        raise TrustedTestSplitAccessError(f"refusing to open trusted test split: {path}")


def _read_json(path: Path) -> Any:
    _guard_no_test_split(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    _guard_no_test_split(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_class_f1_from_confusion(classes: list[str], confusion_matrix: list[list[int]]) -> dict[str, float]:
    """Derive per-class F1 from a saved confusion matrix.

    Rows = true class in `classes` order (matches snapshots[-1]["classes"]).
    TP = cm[i][i]; FN = rowsum(i) - TP; FP = colsum(i) - TP;
    F1 = 2*TP / (2*TP + FP + FN), defined as 0.0 when that denominator is 0.
    """
    n = len(classes)
    result: dict[str, float] = {}
    for i, cls in enumerate(classes):
        tp = confusion_matrix[i][i]
        row_sum = sum(confusion_matrix[i])
        col_sum = sum(confusion_matrix[r][i] for r in range(n))
        fn = row_sum - tp
        fp = col_sum - tp
        denom = 2 * tp + fp + fn
        result[cls] = (2 * tp / denom) if denom else 0.0
    return result


def parse_cell_identity(dir_name: str, *, manifest_learner: str, manifest_seed: int) -> dict[str, Any]:
    """Parse (arm, learner, seed, budget, threshold, attempt) from a cell directory
    name, verifying learner/seed against the manifest rather than trusting the
    name alone. Any mismatch, or a name that cannot be parsed, is a hard error.
    """
    parts = dir_name.split("_")
    budget_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"b\d+", p)), None)
    seed_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"s\d+", p)), None)
    if budget_idx is None or seed_idx is None or seed_idx <= budget_idx:
        raise ValueError(f"cannot locate budget/seed tokens in cell directory name: {dir_name!r}")
    if seed_idx + 2 >= len(parts):
        raise ValueError(f"cell directory name is missing threshold/attempt tokens: {dir_name!r}")
    threshold_token, attempt_token = parts[seed_idx + 1], parts[seed_idx + 2]
    if not re.fullmatch(r"t\d+", threshold_token) or not re.fullmatch(r"attempt\d+", attempt_token):
        raise ValueError(f"unexpected threshold/attempt tokens in cell directory name: {dir_name!r}")

    directory_seed = int(parts[seed_idx][1:])
    if directory_seed != manifest_seed:
        raise ValueError(
            f"{dir_name}: directory seed {directory_seed} does not match manifest stream_seed {manifest_seed}")

    learner_tokens = manifest_learner.split("_")
    middle = parts[budget_idx + 1:seed_idx]
    # Mechanism-ablation Phase 3 names carry a hypervector dimension token between
    # the learner and the seed (..._hdc_d1000_s0_...). The caller verifies it
    # against the capture manifest's ablation_hypervector_dim.
    dim = None
    if middle and re.fullmatch(r"d\d+", middle[-1]):
        dim = int(middle[-1][1:])
        middle = middle[:-1]
    if middle[-len(learner_tokens):] != learner_tokens:
        raise ValueError(
            f"{dir_name}: directory name does not end with manifest learner {manifest_learner!r} "
            f"before the seed token")
    arm_tokens = middle[:-len(learner_tokens)] if len(learner_tokens) else middle
    if not arm_tokens:
        raise ValueError(f"{dir_name}: no arm token found before learner {manifest_learner!r}")

    arm = "_".join(arm_tokens)
    return {
        # The dimension stays in the arm key so seed medians never pool across D.
        "arm": arm if dim is None else f"{arm}_d{dim}",
        "learner": manifest_learner,
        "seed": manifest_seed,
        "budget": int(parts[budget_idx][1:]),
        "threshold": int(threshold_token[1:]),
        "attempt": int(attempt_token[len("attempt"):]),
    }


def discover_cells(validation_root: Path) -> list[Path]:
    if not validation_root.is_dir():
        return []
    return sorted(p for p in validation_root.iterdir() if p.is_dir() and CELL_NAME_RE.match(p.name))


def load_cell(*, validation_dir: Path, capture_root: Path, uptake_root: Path | None) -> dict[str, Any]:
    manifest = _read_json(validation_dir / "manifest.json")
    metrics = _read_json(validation_dir / "metrics.json")
    integrity = _read_json(validation_dir / "integrity_report.json")
    if integrity.get("status") != "PASS":
        raise ValueError(f"{validation_dir}: validation integrity_report status is not PASS")

    identity = parse_cell_identity(
        validation_dir.name, manifest_learner=manifest["learner"], manifest_seed=manifest["stream_seed"])

    capture_dir = capture_root / validation_dir.name
    if not capture_dir.is_dir():
        raise ValueError(f"{validation_dir}: no matching capture directory at {capture_dir}")
    capture_manifest = _read_json(capture_dir / "manifest.json")
    capture_integrity = _read_json(capture_dir / "integrity_report.json")
    if capture_integrity.get("status") != "PASS":
        raise ValueError(f"{capture_dir}: capture integrity_report status is not PASS")
    if (capture_manifest.get("learner") != manifest["learner"]
            or capture_manifest.get("stream_seed") != manifest["stream_seed"]):
        raise ValueError(f"{validation_dir}: capture manifest (learner/stream_seed) disagrees with validation manifest")
    dim_match = re.search(r"_d(\d+)_s\d+_", validation_dir.name)
    directory_dim = int(dim_match.group(1)) if dim_match else None
    if directory_dim != capture_manifest.get("ablation_hypervector_dim"):
        raise ValueError(
            f"{validation_dir}: directory dimension {directory_dim} does not match capture manifest "
            f"ablation_hypervector_dim {capture_manifest.get('ablation_hypervector_dim')}")

    label_events = _read_json(capture_dir / "label_events.json")
    accounting = label_events["accounting"]

    snapshot = metrics["snapshots"][-1]
    snapshot_metrics = snapshot["metrics"]
    classes = snapshot["classes"]
    confusion_matrix = snapshot_metrics["confusion_matrix"]
    per_class_f1 = per_class_f1_from_confusion(classes, confusion_matrix)

    forgetting = metrics["forgetting"]

    uptake = None
    uptake_path = None
    if uptake_root is not None:
        candidate = uptake_root / f"{validation_dir.name}.json"
        if candidate.is_file():
            uptake = _read_json(candidate)
            uptake_path = candidate

    return {
        "cell": validation_dir.name,
        "arm": identity["arm"],
        "learner": identity["learner"],
        "seed": identity["seed"],
        "attempt": identity["attempt"],
        "budget": identity["budget"],
        "threshold": identity["threshold"],
        "macro_f1": snapshot_metrics["macro_f1"],
        "accuracy": snapshot_metrics["accuracy"],
        "macro_recall": snapshot_metrics["macro_recall"],
        "ece": snapshot_metrics["ece"],
        "mean_forgetting": forgetting["mean_forgetting"],
        "labels_consumed": accounting["labels_consumed"],
        "allowance": accounting["allowance"],
        "classes": classes,
        "per_class_f1": per_class_f1,
        "per_class_forgetting": forgetting["per_class"],
        "uptake": uptake,
        "validation_dir": validation_dir,
        "capture_dir": capture_dir,
        "uptake_path": uptake_path,
        "validation_integrity_sha256": _sha256(validation_dir / "integrity_report.json"),
        "capture_integrity_sha256": _sha256(capture_dir / "integrity_report.json"),
    }


def load_all_cells(*, validation_root: Path, capture_root: Path, uptake_root: Path | None) -> list[dict[str, Any]]:
    cells = [
        load_cell(validation_dir=d, capture_root=capture_root, uptake_root=uptake_root)
        for d in discover_cells(validation_root)
    ]
    class_orderings = {tuple(c["classes"]) for c in cells}
    if len(class_orderings) > 1:
        raise ValueError("cells disagree on class order; refusing to aggregate across mismatched class sets")
    return cells


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_cells_table(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    class_order = list(cells[0]["classes"]) if cells else []
    header = ["cell", "arm", "learner", "seed", "attempt", "budget", "threshold",
              "macro_f1", "accuracy", "macro_recall", "ece", "mean_forgetting",
              "labels_consumed", "allowance"]
    header += [f"f1__{c}" for c in class_order]
    header += [f"forgetting__{c}" for c in class_order]
    rows = []
    for cell in sorted(cells, key=lambda c: (c["arm"], c["learner"], c["seed"])):
        row = [cell["cell"], cell["arm"], cell["learner"], cell["seed"], cell["attempt"],
               cell["budget"], cell["threshold"], cell["macro_f1"], cell["accuracy"],
               cell["macro_recall"], cell["ece"], cell["mean_forgetting"],
               cell["labels_consumed"], cell["allowance"]]
        row += [cell["per_class_f1"].get(c) for c in class_order]
        row += [cell["per_class_forgetting"].get(c, {}).get("forgetting") for c in class_order]
        rows.append(row)
    return header, rows


def build_uptake_table(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    header = ["cell", "arm", "learner", "seed", "novel_class", "N_available",
              "original_examples_to_threshold", "original_status",
              "repaired_examples_to_threshold", "repaired_status",
              "terminal_window_examples", "terminal_accuracy"]
    rows = []
    for cell in sorted(cells, key=lambda c: (c["arm"], c["learner"], c["seed"])):
        if not cell["uptake"]:
            continue
        for class_label, record in sorted(cell["uptake"]["classes"].items()):
            rows.append([
                cell["cell"], cell["arm"], cell["learner"], cell["seed"], class_label,
                record["N_available"],
                record["original_three_x_20_at_080"]["examples_to_threshold"],
                record["original_three_x_20_at_080"]["status"],
                record["repaired_three_x_10_at_080"]["examples_to_threshold"],
                record["repaired_three_x_10_at_080"]["status"],
                record["terminal_accuracy"]["window_examples"],
                record["terminal_accuracy"]["accuracy"],
            ])
    return header, rows


def build_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["arm"], cell["learner"]), []).append(cell)

    group_summaries = []
    for (arm, learner), members in sorted(groups.items()):
        members_by_seed = sorted(members, key=lambda c: c["seed"])
        entry: dict[str, Any] = {"arm": arm, "learner": learner, "n_seeds": len(members_by_seed)}
        for field in SCALAR_FIELDS:
            per_seed = {c["seed"]: c[field] for c in members_by_seed}
            entry[field] = {"median": _median(list(per_seed.values())), "per_seed": per_seed}
        group_summaries.append(entry)

    arms = sorted({cell["arm"] for cell in cells})
    win_counts: dict[str, Any] = {"metric": "macro_f1", "learners": list(LABEL_MATCHED_LEARNERS)}
    for arm in arms:
        by_learner_seed: dict[str, dict[int, float]] = {
            learner: {c["seed"]: c["macro_f1"] for c in cells if c["arm"] == arm and c["learner"] == learner}
            for learner in LABEL_MATCHED_LEARNERS
        }
        common_seeds = set.intersection(*(set(v) for v in by_learner_seed.values())) if all(by_learner_seed.values()) else set()
        tally = {learner: 0 for learner in LABEL_MATCHED_LEARNERS}
        for seed in sorted(common_seeds):
            best = max(LABEL_MATCHED_LEARNERS, key=lambda learner: by_learner_seed[learner][seed])
            tally[best] += 1
        win_counts[arm] = {"seeds_considered": len(common_seeds), "wins": tally}

    return {
        "label_matched_learners": list(LABEL_MATCHED_LEARNERS),
        "not_label_matched_learners": list(NOT_LABEL_MATCHED_LEARNERS),
        "groups": group_summaries,
        "win_counts": win_counts,
    }


def build_per_class_table(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """Mean per-class F1 over seeds, per (arm, learner) restricted to the
    label-matched learners (hdc/ncm/ranpac); delta is against the per-class
    best of the *other* label-matched learners in that arm.
    """
    class_order = list(cells[0]["classes"]) if cells else []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        if cell["learner"] in LABEL_MATCHED_LEARNERS:
            groups.setdefault((cell["arm"], cell["learner"]), []).append(cell)

    mean_f1: dict[tuple[str, str], dict[str, float]] = {}
    for (arm, learner), members in groups.items():
        mean_f1[(arm, learner)] = {
            cls: statistics.mean(m["per_class_f1"][cls] for m in members) for cls in class_order
        }

    header = ["arm", "learner", "class", "mean_f1", "delta_vs_best_of_other_label_matched"]
    rows = []
    arms = sorted({arm for arm, _ in groups})
    for arm in arms:
        learners_here = sorted(learner for a, learner in groups if a == arm)
        for learner in learners_here:
            others = [other for other in learners_here if other != learner]
            for cls in class_order:
                value = mean_f1[(arm, learner)][cls]
                if others:
                    best_other = max(mean_f1[(arm, other)][cls] for other in others)
                    delta = value - best_other
                else:
                    delta = None
                rows.append([arm, learner, cls, value, delta])
    return header, rows


def build_concentration(cells: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        if cell["arm"] == CONCENTRATION_ARM and cell["learner"] in (
                (CONCENTRATION_FOCUS_LEARNER,) + CONCENTRATION_CONTROL_LEARNERS):
            groups.setdefault(cell["learner"], []).append(cell)

    required = (CONCENTRATION_FOCUS_LEARNER,) + CONCENTRATION_CONTROL_LEARNERS
    if not all(learner in groups for learner in required):
        return {
            "arm": CONCENTRATION_ARM, "focus_learner": CONCENTRATION_FOCUS_LEARNER,
            "control_learners": list(CONCENTRATION_CONTROL_LEARNERS),
            "status": "insufficient_cells", "classes_considered": 0,
        }

    class_order = list(groups[CONCENTRATION_FOCUS_LEARNER][0]["classes"])
    mean_f1 = {
        learner: {cls: statistics.mean(c["per_class_f1"][cls] for c in members) for cls in class_order}
        for learner, members in groups.items()
    }

    per_class_delta = {}
    for cls in class_order:
        best_control = max(mean_f1[learner][cls] for learner in CONCENTRATION_CONTROL_LEARNERS)
        per_class_delta[cls] = mean_f1[CONCENTRATION_FOCUS_LEARNER][cls] - best_control

    ranked = sorted(per_class_delta.items(), key=lambda kv: kv[1], reverse=True)
    top_classes = [cls for cls, _ in ranked[:CONCENTRATION_TOP_N]]
    macro_delta_all = statistics.mean(per_class_delta.values())
    remaining = [v for cls, v in per_class_delta.items() if cls not in top_classes]
    macro_delta_excluding_top = statistics.mean(remaining) if remaining else None
    wins = sum(1 for v in per_class_delta.values() if v >= 0)

    return {
        "arm": CONCENTRATION_ARM,
        "focus_learner": CONCENTRATION_FOCUS_LEARNER,
        "control_learners": list(CONCENTRATION_CONTROL_LEARNERS),
        "status": "ok",
        "classes_considered": len(class_order),
        "per_class_mean_f1": mean_f1,
        "per_class_delta_vs_best_control": per_class_delta,
        "top_contributing_classes": top_classes,
        "macro_delta_including_top": macro_delta_all,
        "macro_delta_excluding_top": macro_delta_excluding_top,
        "focus_at_or_above_best_control_count": wins,
        "focus_at_or_above_best_control_total": len(class_order),
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_provenance(*, validation_root: Path, capture_root: Path, uptake_root: Path, cells: list[dict[str, Any]]) -> str:
    lines = [
        "# Aggregator provenance",
        "",
        "Read-only, post-hoc aggregation over saved grid/ablation artifacts. No",
        "capture, evaluation, training, or scoring was launched by this script;",
        "it never loads a learner and never opens the trusted test split.",
        "",
        f"- validation root: `{validation_root}`",
        f"- capture root: `{capture_root}`",
        f"- uptake-repair root: `{uptake_root}`",
        f"- cells aggregated: {len(cells)}",
        "",
        "## Per-class F1 convention",
        "",
        "There is no `per_class_f1` field in any saved artifact. It is derived from",
        "`snapshots[-1][\"metrics\"][\"confusion_matrix\"]`, rows/cols in",
        "`snapshots[-1][\"classes\"]` order: for class i, TP = cm[i][i],",
        "FN = rowsum(i) - TP, FP = colsum(i) - TP, F1 = 2*TP / (2*TP + FP + FN),",
        "defined as 0.0 when that denominator is 0.",
        "",
        "## Standing qualifiers",
        "",
        "- `trusted_val` is a development split; this is development-only evidence,",
        "  not held-out evaluation, and no test partition was ever accessed.",
        "- RanPAC-style is an ablation of the aggregation rule, not published RanPAC.",
        "- naive/EWC/replay are NOT label-matched (17-79 labels consumed against the",
        "  prototype heads' 400 in every cell) and must never share an unqualified",
        "  comparison table with hdc/ncm/ranpac-style. per_class.csv and the",
        "  concentration/win-count analyses in this output are restricted to the",
        "  label-matched learners for that reason.",
        "",
        "## Per-cell integrity receipts (SHA-256 of integrity_report.json)",
        "",
    ]
    for cell in sorted(cells, key=lambda c: (c["arm"], c["learner"], c["seed"])):
        lines.append(
            f"- `{cell['cell']}`: validation={cell['validation_integrity_sha256']} "
            f"capture={cell['capture_integrity_sha256']}"
        )
    lines.append("")
    return "\n".join(lines)


def plan(*, validation_root: Path, capture_root: Path, uptake_root: Path, output_dir: Path | None) -> dict[str, Any]:
    cell_dirs = discover_cells(validation_root)
    return {
        "mode": "plan_only",
        "would_read": {
            "validation_root": str(validation_root),
            "capture_root": str(capture_root),
            "uptake_root": str(uptake_root),
            "cell_directories": [d.name for d in cell_dirs],
            "cell_count": len(cell_dirs),
        },
        "would_write": {
            "output_dir": str(output_dir) if output_dir else None,
            "files": ["cells.csv", "summary.json", "per_class.csv", "concentration.json",
                      "PROVENANCE.md", "uptake.csv"],
        },
        "note": "no files were read beyond directory listing, and nothing was written; pass --execute to run",
    }


def run(*, validation_root: Path, capture_root: Path, uptake_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    cells = load_all_cells(validation_root=validation_root, capture_root=capture_root, uptake_root=uptake_root)

    cells_header, cells_rows = build_cells_table(cells)
    summary = build_summary(cells)
    per_class_header, per_class_rows = build_per_class_table(cells)
    concentration = build_concentration(cells)
    uptake_header, uptake_rows = build_uptake_table(cells)
    provenance = build_provenance(
        validation_root=validation_root, capture_root=capture_root, uptake_root=uptake_root, cells=cells)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "cells.csv", cells_header, cells_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "per_class.csv", per_class_header, per_class_rows)
    _write_json(output_dir / "concentration.json", concentration)
    (output_dir / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
    _write_csv(output_dir / "uptake.csv", uptake_header, uptake_rows)

    return {
        "mode": "executed",
        "output_dir": str(output_dir),
        "cell_count": len(cells),
        "files_written": ["cells.csv", "summary.json", "per_class.csv", "concentration.json",
                           "PROVENANCE.md", "uptake.csv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="otherwise print the plan and write nothing")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps(
            plan(validation_root=args.validation_root, capture_root=args.capture_root,
                 uptake_root=DEFAULT_UPTAKE_ROOT, output_dir=args.output_dir),
            indent=2, sort_keys=True))
        return

    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    report = run(validation_root=args.validation_root, capture_root=args.capture_root,
                 uptake_root=DEFAULT_UPTAKE_ROOT, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
