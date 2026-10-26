"""Shared evaluation metrics for E0/E1/E2/E3.

Single source of truth for metric computation so every experiment, and the
E2-vs-E3 comparison in particular, is scored with byte-identical logic.

Behavior matches the original E0/E2 implementations:
- ECE uses 15 bins. Bins are left-closed [low, high); the final bin is
  inclusive [low, high]. (The original E1 script used a right-closed
  convention; the two agree on all real softmax outputs because they differ
  only on exact bin boundaries.)
- The stage is the third underscore-delimited field of a label
  (crop_disease_stage), e.g. "3_b7_1" -> stage "1".
"""

import csv
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def label_to_stage(label: str) -> str:
    return label.split("_")[2]


def expected_calibration_error(probs, targets, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = predictions == targets
    return expected_calibration_error_from_confidence(
        confidences,
        correctness,
        n_bins=n_bins,
    )


def expected_calibration_error_from_confidence(
    confidences, correctness, n_bins: int = 15
) -> float:
    """Compute ECE from top-class confidence and prediction correctness."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if confidences.ndim != 1 or correctness.shape != confidences.shape:
        raise ValueError("confidence and correctness must be matching 1D arrays")
    if len(confidences) == 0:
        raise ValueError("ECE requires at least one prediction")
    if n_bins < 1:
        raise ValueError("ECE requires at least one bin")
    ece = 0.0
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / float(len(correctness))) * abs(bin_acc - bin_conf)
    return float(ece)


def classification_metrics_from_labels(
    y_true_labels, y_pred_labels, classes=None
) -> Dict:
    """Shared accuracy/macro/per-class metrics for string-label predictions."""
    y_true_labels = np.asarray(y_true_labels)
    y_pred_labels = np.asarray(y_pred_labels)
    if y_true_labels.ndim != 1 or y_pred_labels.shape != y_true_labels.shape:
        raise ValueError("true and predicted labels must be matching 1D arrays")
    if len(y_true_labels) == 0:
        raise ValueError("classification metrics require at least one prediction")
    labels = list(classes) if classes is not None else sorted(set(y_true_labels.tolist()))
    recalls = recall_score(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "macro_f1": float(
            f1_score(
                y_true_labels,
                y_pred_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true_labels,
                y_pred_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            label: float(value) for label, value in zip(labels, recalls)
        },
    }


def stage_recall(y_true_labels, y_pred_labels) -> Dict[str, float]:
    true_stages = np.array([label_to_stage(label) for label in y_true_labels])
    pred_stages = np.array([label_to_stage(label) for label in y_pred_labels])
    recalls = {}
    for stage in ["0", "1", "2", "3"]:
        mask = true_stages == stage
        recalls[stage] = None if mask.sum() == 0 else float((pred_stages[mask] == true_stages[mask]).mean())
    return recalls


def multiclass_roc_auc(probs, targets, average):
    try:
        return float(
            roc_auc_score(
                targets,
                probs,
                labels=list(range(probs.shape[1])),
                multi_class="ovr",
                average=average,
            )
        )
    except ValueError:
        return None


def metrics_dict(eval_output, classes) -> Dict:
    """Compute the standard metric block.

    classes is the list of label strings in index order (classes[i] is the
    label for class index i).
    """
    targets = eval_output["targets"]
    preds = eval_output["preds"]
    probs = eval_output["probs"]
    y_true_labels = np.array([classes[int(i)] for i in targets])
    y_pred_labels = np.array([classes[int(i)] for i in preds])
    per_stage = stage_recall(y_true_labels, y_pred_labels)
    return {
        "loss": float(eval_output["loss"]),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "macro_roc_auc_ovr": multiclass_roc_auc(probs, targets, "macro"),
        "weighted_roc_auc_ovr": multiclass_roc_auc(probs, targets, "weighted"),
        "per_stage_recall": per_stage,
        "early_stage_recall": per_stage["1"],
        "ece": expected_calibration_error(probs, targets, n_bins=15),
    }


def per_class_metrics(eval_output, classes) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        eval_output["targets"],
        eval_output["preds"],
        labels=list(range(len(classes))),
        zero_division=0,
    )
    out = {}
    for idx, label in enumerate(classes):
        out[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return out


def save_confusion_matrix(path, eval_output, classes) -> None:
    cm = confusion_matrix(eval_output["targets"], eval_output["preds"], labels=list(range(len(classes))))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(classes))
        for label, row in zip(classes, cm):
            writer.writerow([label] + row.tolist())


def save_predictions(path, meta, eval_output, classes) -> None:
    """E2-style rich predictions CSV (reused by E3).

    meta is a DataFrame containing at least image, label, and stage columns.
    Adds pred_label, pred_stage, correct, confidence, and one prob_<class>
    column per class.
    """
    probs = eval_output["probs"]
    preds = eval_output["preds"]
    out = meta[["image", "label", "stage"]].copy()
    out["pred_label"] = [classes[int(i)] for i in preds]
    out["pred_stage"] = [label.split("_")[2] for label in out["pred_label"]]
    out["correct"] = out["label"] == out["pred_label"]
    out["confidence"] = probs.max(axis=1)
    for idx, label in enumerate(classes):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(path, index=False)
