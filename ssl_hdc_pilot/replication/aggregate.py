"""Read-only, post-hoc aggregator over finished replication cells.

Reads whatever cells ``run_batch.py`` has finished under
``ssl_hdc_pilot/results/replication/cells/<Dataset>/<arm>/<learner>_s<seed>/``
plus the original study's already-saved B400 seed-0..2 grid cells (the
"Original" dataset rows), and writes per-dataset x arm x learner comparison
tables plus a concise COMPARISON.md to an output directory.

It never loads a learner, never opens a sealed test split, and never
launches anything -- it only reads already-saved metrics.json /
integrity_report.json / snapshot_index.json / prediction files (new cells)
or reuses ``ssl_hdc_pilot.scripts.aggregate_grid`` (Original cells) for the
identical validation/capture artifact family that script already knows how
to parse safely.

Statistics
==========
Per (dataset, arm, learner): mean/sd over available seeds of macro-F1,
mean_forgetting, accuracy, ECE, labels_consumed, and novel-class terminal
accuracy. HDC vs. best-control (max of NCM/RanPAC-style) win/tie/loss is the
seed-wise argmax rule already used by aggregate_grid.build_summary's
win_counts (a tie is an exact match to within 1e-9); the paired-bootstrap
interval resamples the per-seed (HDC - best_control) macro-F1 differences
with replacement (default 10,000 resamples, a fixed RNG seed for
reproducibility) and reports the 2.5/97.5 percentile interval of the
resampled mean. With <=5 seeds this is disclosed as consistency evidence,
not a powered test, matching REPLICATION_PLAN.md's reporting lock.

Novel-class terminal accuracy is NOT computed the same way for both dataset
families, and this is disclosed rather than glossed over:
- Original: taken from the already-saved stream uptake-repair artifact's
  ``terminal_accuracy`` (a rolling window on the live *stream*).
- New datasets (PlantDoc/PlantWild/PlantVillage): no stream-side uptake
  artifact exists for these cells, only fixed validation-population
  snapshots. The proxy used here is the final checkpoint's per-class recall
  on the validation split for that novel class. This is flagged as
  ``novel_terminal_accuracy_method`` in every row and called out in
  COMPARISON.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score

from ssl_hdc_pilot.scripts import aggregate_grid as ag

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication"
RESULTS_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "results" / "replication"
DEFAULT_CELLS_ROOT = RESULTS_ROOT / "cells"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "comparison"

DATASETS = ("PlantDoc", "PlantWild", "PlantVillage")
SIZES = ("vits16", "vitb16", "vitl16")
ARMS = tuple(f"frozen_{size}" for size in SIZES) + tuple(f"adapted_{size}" for size in SIZES)
LEARNERS = ("hdc", "ncm", "ranpac")
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SEED_COUNT = 5
CONTROL_LEARNERS = ("ncm", "ranpac")
TIE_TOLERANCE = 1e-9
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260917

FORBIDDEN_TEST_SPLIT_PARTS = ("trusted_splits", "test.csv")


def _guard_no_test_split(path: Path) -> None:
    parts = path.parts
    if any(parts[i:i + 2] == FORBIDDEN_TEST_SPLIT_PARTS for i in range(len(parts) - 1)) or path.name == "test.csv":
        raise RuntimeError(f"refusing to open trusted test split: {path}")


def _read_json(path: Path) -> Any:
    _guard_no_test_split(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cell_output_dir(cells_root: Path, dataset: str, arm: str, learner: str, seed: int) -> Path:
    return cells_root / dataset / arm / f"{learner}_s{seed}"


def novel_classes_for_dataset(dataset: str, *, replication_root: Path = REPLICATION_ROOT) -> list[str]:
    feasibility = _read_json(replication_root / "specs" / f"{dataset}_stream_feasibility.json")
    return list(feasibility["novel_classes_in_entry_order"])


def _load_final_predictions(output_dir: Path) -> list[dict[str, Any]]:
    index = _read_json(output_dir / "snapshot_index.json")
    final = max(index["snapshots"], key=lambda s: s["processed_stream_events"])
    return _read_json(output_dir / final["predictions"])["rows"]


def per_class_f1_from_rows(rows: list[dict[str, Any]], classes: list[str]) -> dict[str, float]:
    truth = [str(row["true_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    scores = f1_score(truth, predicted, labels=classes, average=None, zero_division=0)
    return {cls: float(score) for cls, score in zip(classes, scores)}


def load_new_cell(*, cells_root: Path, replication_root: Path, dataset: str, arm: str,
                   learner: str, seed: int) -> dict[str, Any] | None:
    output_dir = cell_output_dir(cells_root, dataset, arm, learner, seed)
    metrics_path = output_dir / "metrics.json"
    receipt_path = output_dir / "integrity_report.json"
    if not metrics_path.is_file() or not receipt_path.is_file():
        return None
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "PASS":
        return None
    metrics = _read_json(metrics_path)
    final = metrics["final_validation"]
    classes = sorted(final["per_class_recall"])
    rows = _load_final_predictions(output_dir)
    novel_classes = novel_classes_for_dataset(dataset, replication_root=replication_root)
    return {
        "dataset": dataset, "arm": arm, "learner": learner, "seed": seed,
        "macro_f1": final["macro_f1"], "accuracy": final["accuracy"], "ece": final["ece"],
        "mean_forgetting": metrics["forgetting"]["mean_forgetting"],
        "labels_consumed": metrics["labels_consumed"], "label_budget": metrics["label_budget"],
        "label_budget_match": metrics["label_budget_match"],
        "n_classes": len(classes), "classes": classes,
        "per_class_f1": per_class_f1_from_rows(rows, classes),
        "novel_terminal_accuracy": {cls: final["per_class_recall"].get(cls) for cls in novel_classes},
        "novel_terminal_accuracy_method": "validation_population_final_checkpoint_recall_proxy",
    }


def load_all_new_cells(*, cells_root: Path = DEFAULT_CELLS_ROOT,
                        replication_root: Path = REPLICATION_ROOT) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in ARMS:
            for learner in LEARNERS:
                for seed in SEEDS:
                    record = load_new_cell(cells_root=cells_root, replication_root=replication_root,
                                            dataset=dataset, arm=arm, learner=learner, seed=seed)
                    if record is not None:
                        records.append(record)
    return records


def load_original_cells(*, validation_root: Path = ag.DEFAULT_VALIDATION_ROOT,
                         capture_root: Path = ag.DEFAULT_CAPTURE_ROOT,
                         uptake_root: Path = ag.DEFAULT_UPTAKE_ROOT) -> list[dict[str, Any]]:
    if not validation_root.is_dir():
        return []
    cells = ag.load_all_cells(validation_root=validation_root, capture_root=capture_root, uptake_root=uptake_root)
    records: list[dict[str, Any]] = []
    for cell in cells:
        if cell["arm"] not in ("frozen", "adapted") or cell["learner"] not in ag.LABEL_MATCHED_LEARNERS:
            continue
        novel_terminal: dict[str, float] = {}
        if cell["uptake"]:
            for cls, entry in cell["uptake"]["classes"].items():
                novel_terminal[cls] = entry["terminal_accuracy"]["accuracy"]
        records.append({
            "dataset": "Original", "arm": cell["arm"], "learner": cell["learner"], "seed": cell["seed"],
            "macro_f1": cell["macro_f1"], "accuracy": cell["accuracy"], "ece": cell["ece"],
            "mean_forgetting": cell["mean_forgetting"], "labels_consumed": cell["labels_consumed"],
            "label_budget": cell["allowance"], "label_budget_match": None,
            "n_classes": len(cell["classes"]), "classes": list(cell["classes"]),
            "per_class_f1": cell["per_class_f1"], "novel_terminal_accuracy": novel_terminal,
            "novel_terminal_accuracy_method": "stream_uptake_repair_terminal_window",
        })
    return records


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def win_tie_loss(hdc_by_seed: dict[int, float], control_by_seed: dict[str, dict[int, float]]) -> dict[str, Any]:
    """Seed-wise HDC vs. best-of-controls comparison (see module docstring)."""
    common_seeds = sorted(set(hdc_by_seed) & set.intersection(*(set(v) for v in control_by_seed.values())))
    wins = ties = losses = 0
    diffs: list[float] = []
    for seed in common_seeds:
        best_control = max(control_by_seed[learner][seed] for learner in control_by_seed)
        diff = hdc_by_seed[seed] - best_control
        diffs.append(diff)
        if abs(diff) <= TIE_TOLERANCE:
            ties += 1
        elif diff > 0:
            wins += 1
        else:
            losses += 1
    return {"seeds_considered": len(common_seeds), "wins": wins, "ties": ties, "losses": losses, "diffs": diffs}


def paired_bootstrap_interval(diffs: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                               seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if not diffs:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "resamples": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return {"mean_diff": statistics.mean(diffs), "ci_low": lo, "ci_high": hi, "resamples": resamples,
            "excludes_zero": (lo > 0) or (hi < 0)}


def build_group_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["dataset"], record["arm"], record["learner"]), []).append(record)

    all_keys = {(r["dataset"], r["arm"], r["learner"]) for r in records}
    rows = []
    for dataset, arm, learner in sorted(all_keys):
        members = sorted(groups[(dataset, arm, learner)], key=lambda r: r["seed"])
        seeds_present = [m["seed"] for m in members]
        row: dict[str, Any] = {
            "dataset": dataset, "arm": arm, "learner": learner,
            "n_seeds": len(members), "seeds_present": seeds_present,
            "incomplete_seeds": len(members) < EXPECTED_SEED_COUNT,
        }
        for field in ("macro_f1", "accuracy", "ece", "mean_forgetting", "labels_consumed"):
            mean, sd = _mean_sd([m[field] for m in members])
            row[f"{field}_mean"], row[f"{field}_sd"] = mean, sd
        novel_means: list[float] = []
        for member in members:
            values = [v for v in member["novel_terminal_accuracy"].values() if v is not None]
            if values:
                novel_means.append(statistics.mean(values))
        mean, sd = _mean_sd(novel_means)
        row["novel_terminal_accuracy_mean"], row["novel_terminal_accuracy_sd"] = mean, sd
        row["n_classes_set"] = sorted({m["n_classes"] for m in members})
        row["class_count_flag"] = len(row["n_classes_set"]) > 1
        rows.append(row)
    return rows


def build_label_count_flags(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag (dataset, arm, seed) where learners disagree on labels_consumed."""
    groups: dict[tuple[str, str, int], dict[str, int]] = {}
    for record in records:
        key = (record["dataset"], record["arm"], record["seed"])
        groups.setdefault(key, {})[record["learner"]] = record["labels_consumed"]
    flags = []
    for (dataset, arm, seed), by_learner in sorted(groups.items()):
        if len(set(by_learner.values())) > 1:
            flags.append({"dataset": dataset, "arm": arm, "seed": seed, "labels_consumed_by_learner": by_learner})
    return flags


def build_win_tie_loss_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    for record in records:
        key = (record["dataset"], record["arm"])
        groups.setdefault(key, {}).setdefault(record["learner"], {})[record["seed"]] = record["macro_f1"]

    rows = []
    for (dataset, arm), by_learner in sorted(groups.items()):
        if "hdc" not in by_learner or not all(learner in by_learner for learner in CONTROL_LEARNERS):
            continue
        control_by_seed = {learner: by_learner[learner] for learner in CONTROL_LEARNERS}
        outcome = win_tie_loss(by_learner["hdc"], control_by_seed)
        bootstrap = paired_bootstrap_interval(outcome["diffs"])
        rows.append({
            "dataset": dataset, "arm": arm, "seeds_considered": outcome["seeds_considered"],
            "hdc_wins": outcome["wins"], "hdc_ties": outcome["ties"], "hdc_losses": outcome["losses"],
            "mean_diff_vs_best_control": bootstrap["mean_diff"],
            "bootstrap_ci_low": bootstrap["ci_low"], "bootstrap_ci_high": bootstrap["ci_high"],
            "bootstrap_excludes_zero": bootstrap.get("excludes_zero"),
        })
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


METRIC_FIELDS = ("macro_f1", "accuracy", "ece", "mean_forgetting", "labels_consumed",
                  "novel_terminal_accuracy")


def write_metric_csvs(output_dir: Path, group_rows: list[dict[str, Any]]) -> None:
    for field in METRIC_FIELDS:
        header = ["dataset", "arm", "learner", "n_seeds", "seeds_present", "mean", "sd",
                  "incomplete_seeds", "class_count_flag"]
        rows = [[r["dataset"], r["arm"], r["learner"], r["n_seeds"], ",".join(map(str, r["seeds_present"])),
                 r[f"{field}_mean"], r[f"{field}_sd"], r["incomplete_seeds"], r["class_count_flag"]]
                for r in group_rows]
        _write_csv(output_dir / f"{field}.csv", header, rows)


def write_win_tie_loss_csv(output_dir: Path, wtl_rows: list[dict[str, Any]]) -> None:
    header = ["dataset", "arm", "seeds_considered", "hdc_wins", "hdc_ties", "hdc_losses",
              "mean_diff_vs_best_control", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_excludes_zero"]
    rows = [[r[h] for h in header] for r in wtl_rows]
    _write_csv(output_dir / "win_tie_loss.csv", header, rows)


def write_label_flags_csv(output_dir: Path, flags: list[dict[str, Any]]) -> None:
    header = ["dataset", "arm", "seed", "labels_consumed_by_learner"]
    rows = [[f["dataset"], f["arm"], f["seed"], json.dumps(f["labels_consumed_by_learner"], sort_keys=True)]
            for f in flags]
    _write_csv(output_dir / "label_count_flags.csv", header, rows)


def build_comparison_markdown(*, group_rows: list[dict[str, Any]], wtl_rows: list[dict[str, Any]],
                               label_flags: list[dict[str, Any]], new_cell_count: int,
                               original_cell_count: int) -> str:
    lines = [
        "# Replication comparison",
        "",
        "Read-only aggregation over saved cells; nothing was launched or scored by this script.",
        f"- new-dataset cells read: {new_cell_count}",
        f"- Original (own dataset, B400) cells read: {original_cell_count}",
        "",
        "## Standing qualifiers",
        "",
        "- RanPAC-style is an ablation of the aggregation rule, not published RanPAC.",
        "- Five seeds (three for Original -- see below) are consistency evidence, not a powered test.",
        "- Validation is development-only; no held-out (test-split) evidence is claimed anywhere here.",
        "- Novel-class terminal accuracy is NOT computed identically across dataset families: Original "
        "uses the saved stream uptake-repair rolling window; PlantDoc/PlantWild/PlantVillage use a "
        "validation-population final-checkpoint recall proxy (no stream-side uptake artifact exists for "
        "those cells). Do not compare the two directly without accounting for this.",
        "",
        "## Per dataset x arm x learner (mean +/- sd over available seeds)",
        "",
        "| dataset | arm | learner | n_seeds | macro_f1 | forgetting | accuracy | ece | "
        "labels_consumed | novel_terminal_acc | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(mean: float | None, sd: float | None) -> str:
        if mean is None:
            return "MISSING"
        if sd is None:
            return f"{mean:.4f}"
        return f"{mean:.4f}+/-{sd:.4f}"

    for row in group_rows:
        flags = []
        if row["incomplete_seeds"]:
            flags.append(f"seeds={row['n_seeds']}/{EXPECTED_SEED_COUNT}")
        if row["class_count_flag"]:
            flags.append(f"class_counts={row['n_classes_set']}")
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['learner']} | {row['n_seeds']} | "
            f"{fmt(row['macro_f1_mean'], row['macro_f1_sd'])} | "
            f"{fmt(row['mean_forgetting_mean'], row['mean_forgetting_sd'])} | "
            f"{fmt(row['accuracy_mean'], row['accuracy_sd'])} | "
            f"{fmt(row['ece_mean'], row['ece_sd'])} | "
            f"{fmt(row['labels_consumed_mean'], row['labels_consumed_sd'])} | "
            f"{fmt(row['novel_terminal_accuracy_mean'], row['novel_terminal_accuracy_sd'])} | "
            f"{'; '.join(flags) if flags else '-'} |"
        )

    lines += ["", "## HDC vs. best control (NCM/RanPAC-style) win/tie/loss", "",
              "| dataset | arm | seeds | wins | ties | losses | mean diff | 95% bootstrap CI | excludes 0 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in wtl_rows:
        ci = ("MISSING" if row["bootstrap_ci_low"] is None
              else f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]")
        mean_diff = "MISSING" if row["mean_diff_vs_best_control"] is None else f"{row['mean_diff_vs_best_control']:.4f}"
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['seeds_considered']} | {row['hdc_wins']} | "
                     f"{row['hdc_ties']} | {row['hdc_losses']} | {mean_diff} | {ci} | {row['bootstrap_excludes_zero']} |")

    lines += ["", "## Label-count differences across learners (same dataset/arm/seed)", ""]
    if label_flags:
        for flag in label_flags:
            lines.append(f"- {flag['dataset']}/{flag['arm']} seed {flag['seed']}: "
                         f"{json.dumps(flag['labels_consumed_by_learner'], sort_keys=True)}")
    else:
        lines.append("- none observed among the cells read so far.")
    lines.append("")
    return "\n".join(lines)


def run(*, cells_root: Path, replication_root: Path, validation_root: Path, capture_root: Path,
        uptake_root: Path, output_dir: Path) -> dict[str, Any]:
    new_records = load_all_new_cells(cells_root=cells_root, replication_root=replication_root)
    original_records = load_original_cells(validation_root=validation_root, capture_root=capture_root,
                                            uptake_root=uptake_root)
    all_records = new_records + original_records

    group_rows = build_group_rows(all_records)
    wtl_rows = build_win_tie_loss_rows(all_records)
    label_flags = build_label_count_flags(all_records)
    comparison_md = build_comparison_markdown(group_rows=group_rows, wtl_rows=wtl_rows, label_flags=label_flags,
                                              new_cell_count=len(new_records),
                                              original_cell_count=len(original_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_csvs(output_dir, group_rows)
    write_win_tie_loss_csv(output_dir, wtl_rows)
    write_label_flags_csv(output_dir, label_flags)
    (output_dir / "COMPARISON.md").write_text(comparison_md, encoding="utf-8")

    return {"output_dir": str(output_dir), "new_cells_read": len(new_records),
            "original_cells_read": len(original_records), "groups": len(group_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-root", type=Path, default=DEFAULT_CELLS_ROOT)
    parser.add_argument("--replication-root", type=Path, default=REPLICATION_ROOT)
    parser.add_argument("--validation-root", type=Path, default=ag.DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--capture-root", type=Path, default=ag.DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--uptake-root", type=Path, default=ag.DEFAULT_UPTAKE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run(cells_root=args.cells_root, replication_root=args.replication_root,
                 validation_root=args.validation_root, capture_root=args.capture_root,
                 uptake_root=args.uptake_root, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
