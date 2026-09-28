"""Compare the five approaches on the test set: tables, ROC curves, calibration and significance tests."""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import auc, roc_curve

import plot_style
from config import LABEL_ENGLISH, LABEL_NAMES, NUM_CLASSES, PREDICTIONS_DIR, RESULTS_DIR
from metrics import (
    EXPERIMENT_LOG,
    bootstrap_macro_f1,
    compute_metrics,
    figure_path,
    plot_confusion_matrix,
    plot_learning_curves,
)

plt = plot_style.plt

MODELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "bilstm": "BiLSTM + attention",
    "textcnn": "TextCNN",
    "afriberta": "AfriBERTa",
}


def load_predictions():
    predictions = {}
    for key, name in MODELS.items():
        path = PREDICTIONS_DIR / f"{key}_test.csv"
        if path.exists():
            predictions[name] = pd.read_csv(path)
        else:
            print(f"missing predictions for {name}, skipping")
    return predictions


def probabilities_of(frame):
    return frame[[f"prob_{n}" for n in LABEL_NAMES]].to_numpy()


def comparison_table(predictions):
    rows = []
    for key, name in MODELS.items():
        if name not in predictions:
            continue
        frame = predictions[name]
        metrics = compute_metrics(frame["label"], probabilities_of(frame))
        with open(RESULTS_DIR / f"{key}_metrics.json") as f:
            saved = json.load(f)
        summary = saved.get("summary", {})
        low, high = bootstrap_macro_f1(frame["label"], frame["predicted"])
        rows.append({
            "model": name,
            "val_macro_f1": saved["validation"]["macro_f1"],
            "test_macro_f1": metrics["macro_f1"],
            "macro_f1_95ci": f"{low:.3f}-{high:.3f}",
            "seed_mean_std": (f"{summary['test_macro_f1_mean']:.3f} +/- {summary['test_macro_f1_std']:.3f}"
                              if summary else "deterministic"),
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "log_loss": metrics["log_loss"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "parameters": summary.get("parameters"),
            "train_seconds": summary.get("seconds"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    return table


def per_class_table(predictions):
    rows = {}
    for name, frame in predictions.items():
        metrics = compute_metrics(frame["label"], probabilities_of(frame))
        rows[name] = {label: metrics[f"f1_{label}"] for label in LABEL_NAMES}
    table = pd.DataFrame(rows).T
    table.to_csv(RESULTS_DIR / "per_class_f1.csv")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.8 / len(table)
    positions = np.arange(NUM_CLASSES)
    for i, (name, values) in enumerate(table.iterrows()):
        ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.9,
               color=plot_style.MODEL_COLORS[name], label=name)
    ax.set_xticks(positions, [f"{n}\n({LABEL_ENGLISH[n]})" for n in LABEL_NAMES])
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on the test set")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    plot_style.save(fig, figure_path("per_class_f1.png"))
    return table


def plot_roc_curves(predictions):
    """Macro-average one-vs-rest ROC curve per model."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    grid = np.linspace(0, 1, 500)
    for name, frame in predictions.items():
        probabilities = probabilities_of(frame)
        true_positive_rates = []
        for label_id in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(frame["label"] == label_id, probabilities[:, label_id])
            true_positive_rates.append(np.interp(grid, fpr, tpr))
        mean_tpr = np.mean(true_positive_rates, axis=0)
        ax.plot(grid, mean_tpr, color=plot_style.MODEL_COLORS[name], label=f"{name} (AUC {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=plot_style.GRID, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("false positive rate (log scale)")
    ax.set_ylabel("true positive rate")
    ax.set_title("Macro-average ROC (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    plot_style.save(fig, figure_path("roc_curves.png"))


def plot_calibration(predictions, bins=10):
    """Reliability of the top-class confidence. Log loss punishes over-confident mistakes."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rows = []
    for name, frame in predictions.items():
        probabilities = probabilities_of(frame)
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == frame["label"].to_numpy()
        edges = np.linspace(0, 1, bins + 1)
        which = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
        centers, accuracies, error = [], [], 0.0
        for b in range(bins):
            chosen = which == b
            if chosen.sum() >= 20:
                centers.append(confidence[chosen].mean())
                accuracies.append(correct[chosen].mean())
            if chosen.any():
                error += chosen.mean() * abs(confidence[chosen].mean() - correct[chosen].mean())
        rows.append({"model": name, "expected_calibration_error": round(error, 4)})
        ax.plot(centers, accuracies, marker="o", markersize=4, color=plot_style.MODEL_COLORS[name],
                label=f"{name} (ECE {error:.3f})")
    ax.plot([0, 1], [0, 1], color=plot_style.TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration on the test set")
    ax.legend(fontsize=8)
    plot_style.save(fig, figure_path("calibration.png"))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "calibration.csv", index=False)


def mcnemar_tests(predictions, table):
    """Exact McNemar test for every pair of models: do they make significantly different errors?"""
    order = table.sort_values("test_macro_f1", ascending=False)["model"].tolist()
    rows = []
    for i, first in enumerate(order):
        for second in order[i + 1:]:
            first_correct = predictions[first]["predicted"] == predictions[first]["label"]
            second_correct = predictions[second]["predicted"] == predictions[second]["label"]
            only_first = int((first_correct & ~second_correct).sum())
            only_second = int((~first_correct & second_correct).sum())
            total = only_first + only_second
            p_value = binomtest(only_first, total, 0.5).pvalue if total else 1.0
            rows.append({"better_model": first, "compared_with": second, "only_better_correct": only_first,
                         "only_other_correct": only_second, "p_value": float(f"{p_value:.3g}")})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "significance.csv", index=False)
    print(result.to_string(index=False))


def plot_experiment_progression():
    """Validation macro-F1 of every tracked experiment, in the order it was run."""
    log = pd.read_csv(EXPERIMENT_LOG)
    log = log[~log["experiment_id"].str.contains("-")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    start = 0
    ticks, labels = [], []
    for name in MODELS.values():
        rows = log[log["model"] == name].sort_values("experiment_id")
        if rows.empty:
            continue
        x = np.arange(start, start + len(rows))
        color = plot_style.MODEL_COLORS[name]
        ax.plot(x, rows["val_macro_f1"], marker="o", markersize=6, linewidth=1.2, color=color, label=name)
        ax.step(x, rows["val_macro_f1"].cummax(), where="post", color=color, linewidth=1, linestyle="--", alpha=0.7)
        ticks.extend(x)
        labels.extend(rows["experiment_id"])
        start += len(rows) + 1
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Validation macro-F1 of every tracked experiment (dashed line: best so far)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plot_style.save(fig, figure_path("experiment_progression.png"))


def plot_all_confusions(predictions):
    columns = 3
    rows = (len(predictions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.1 * rows), squeeze=False)
    for ax, (name, frame) in zip(axes.ravel(), predictions.items()):
        plot_confusion_matrix(frame["label"], frame["predicted"], name, None, ax=ax)
    for ax in axes.ravel()[len(predictions):]:
        ax.axis("off")
    fig.tight_layout()
    plot_style.save(fig, figure_path("confusion_all_models.png"))


def plot_all_learning_curves():
    """Learning curves of the final seed-42 run of each neural model, side by side."""
    log = pd.read_csv(EXPERIMENT_LOG)
    finals = log[log["change"].str.startswith("Final configuration")]
    order = ["BiLSTM + attention", "TextCNN", "AfriBERTa"]
    finals = finals[finals["model"].isin(order)]
    finals = finals.assign(position=finals["model"].map(order.index)).sort_values("position")
    if finals.empty:
        return
    fig, axes = plt.subplots(2, len(finals), figsize=(4.2 * len(finals), 6), squeeze=False)
    for column, (_, row) in enumerate(finals.iterrows()):
        with open(RESULTS_DIR / "history" / f"{row['experiment_id']}.json") as f:
            history = pd.DataFrame(json.load(f))
        key = [k for k, name in MODELS.items() if name == row["model"]][0]
        plot_learning_curves(history.to_dict("records"), f"{row['model']} learning curves",
                             figure_path(f"learning_curves_{key}.png"))
        top, bottom = axes[0, column], axes[1, column]
        top.plot(history["epoch"], history["train_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[0], label="train")
        top.plot(history["epoch"], history["val_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[1], label="validation")
        top.set_title(row["model"])
        top.set_ylabel("cross-entropy loss")
        top.legend(fontsize=8)
        bottom.plot(history["epoch"], history["val_macro_f1"], marker="o", markersize=4,
                    color=plot_style.MODEL_COLORS[row["model"]])
        bottom.set_xlabel("epoch")
        bottom.set_ylabel("validation macro-F1")
        for ax in (top, bottom):
            ax.set_xticks(history["epoch"])
    fig.tight_layout()
    plot_style.save(fig, figure_path("learning_curves_all.png"))


def main():
    plot_style.apply_style()
    predictions = load_predictions()
    table = comparison_table(predictions)
    per_class_table(predictions)
    plot_roc_curves(predictions)
    plot_calibration(predictions)
    mcnemar_tests(predictions, table)
    plot_experiment_progression()
    plot_all_confusions(predictions)
    plot_all_learning_curves()


if __name__ == "__main__":
    main()
