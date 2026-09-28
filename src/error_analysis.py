"""Where and why the models fail: confusions, length, rare words, code-switching and hard examples."""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import plot_style
from baselines import build_features
from compare_models import MODELS, load_predictions
from config import LABEL_NAMES, RESULTS_DIR
from data import load_splits
from metrics import compute_metrics, figure_path
from neural_models import BiLSTMClassifier
from preprocess import clean_splits
from sequence_data import PAD, UNK

plt = plot_style.plt
ENGLISH_WORDS = {w for w in ENGLISH_STOP_WORDS if len(w) > 2}
BYLINE_NAME = re.compile(r"^\s*(?:Na|NA|na)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,2})")
LENGTH_BINS = [0, 128, 256, 512, 1024, 10000]
LENGTH_LABELS = ["<128", "128-255", "256-511", "512-1023", "1024+"]


def load_test_articles():
    train, validation, test = load_splits()
    train, validation, test, _ = clean_splits(train, validation, test)
    return train, test


def top_confusions(predictions, summary):
    rows = []
    for name, frame in predictions.items():
        wrong = frame[frame["label"] != frame["predicted"]]
        pairs = Counter(zip(wrong["label"], wrong["predicted"]))
        for (true, predicted), count in pairs.most_common(4):
            rows.append({"model": name, "true": LABEL_NAMES[true], "predicted": LABEL_NAMES[predicted],
                         "count": count, "share_of_errors": round(count / len(wrong), 3)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "error_top_confusions.csv", index=False)
    summary["top_confusions"] = rows


def score_by_group(predictions, groups, group_order):
    rows = []
    for name, frame in predictions.items():
        for group in group_order:
            chosen = groups == group
            if chosen.sum() == 0:
                continue
            truth, guess = frame["label"][chosen], frame["predicted"][chosen]
            rows.append({"model": name, "group": group, "articles": int(chosen.sum()),
                         "accuracy": round((truth == guess).mean(), 4),
                         "macro_f1": round(f1_score(truth, guess, average="macro"), 4)})
    return pd.DataFrame(rows)


def by_length(predictions, test, summary):
    lengths = test["text"].str.split().str.len()
    groups = pd.cut(lengths, LENGTH_BINS, labels=LENGTH_LABELS, right=False).astype(str)
    table = score_by_group(predictions, groups.to_numpy(), LENGTH_LABELS)
    table.to_csv(RESULTS_DIR / "error_by_length.csv", index=False)
    summary["accuracy_by_length"] = table.to_dict("records")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in predictions:
        rows = table[table["model"] == name]
        ax.plot(rows["group"], rows["accuracy"], marker="o", markersize=5, color=plot_style.MODEL_COLORS[name],
                label=name)
    counts = table.drop_duplicates("group").set_index("group")["articles"]
    ax.set_xticks(range(len(counts)), [f"{g}\n(n={counts[g]})" for g in counts.index])
    ax.set_xlabel("article length in words")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy by article length")
    ax.legend(fontsize=8)
    plot_style.save(fig, figure_path("error_by_length.png"))


def by_rare_words(predictions, train, test, summary):
    counts = Counter(w for t in train["clean_text"] for w in t.split())
    rare_share = test["clean_text"].map(lambda t: np.mean([counts.get(w, 0) < 2 for w in t.split()]))
    groups = pd.qcut(rare_share, 4, labels=["lowest", "low", "high", "highest"]).astype(str).to_numpy()
    table = score_by_group(predictions, groups, ["lowest", "low", "high", "highest"])
    table.to_csv(RESULTS_DIR / "error_by_rare_words.csv", index=False)
    summary["accuracy_by_rare_word_quartile"] = table.to_dict("records")
    summary["rare_word_quartile_edges"] = [round(float(q), 4) for q in rare_share.quantile([0, .25, .5, .75, 1])]


def by_code_switching(predictions, test, summary):
    share = test["clean_text"].map(lambda t: np.mean([w in ENGLISH_WORDS for w in t.split()]))
    groups = np.where(share > 0.01, "mixed", "swahili_only")
    table = score_by_group(predictions, groups, ["swahili_only", "mixed"])
    table.to_csv(RESULTS_DIR / "error_by_code_switching.csv", index=False)
    summary["accuracy_by_code_switching"] = table.to_dict("records")


def hard_examples(predictions, test, summary, per_group=12):
    """Articles that every model gets wrong are either genuinely ambiguous or mislabelled."""
    names = list(predictions)
    correct = np.column_stack([(predictions[n]["predicted"] == predictions[n]["label"]).to_numpy() for n in names])
    models_correct = correct.sum(axis=1)
    summary["models_correct_distribution"] = {int(k): int(v) for k, v in Counter(models_correct).items()}

    best = max(names, key=lambda n: f1_score(predictions[n]["label"], predictions[n]["predicted"], average="macro"))
    best_frame = predictions[best]
    confidence = best_frame[[f"prob_{n}" for n in LABEL_NAMES]].max(axis=1)

    rows = []
    all_wrong = np.where(models_correct == 0)[0]
    confident_wrong = best_frame.index[(best_frame["predicted"] != best_frame["label"]) & (confidence > 0.9)]
    for kind, indices in [("all_models_wrong", all_wrong[:per_group]),
                          (f"{best}_confident_wrong", confident_wrong[:per_group])]:
        for i in indices:
            rows.append({
                "kind": kind,
                "test_index": int(i),
                "true": LABEL_NAMES[test["label"].iloc[i]],
                **{f"pred_{n}": LABEL_NAMES[predictions[n]["predicted"].iloc[i]] for n in names},
                "best_confidence": round(float(confidence.iloc[i]), 3),
                "words": len(test["text"].iloc[i].split()),
                "opening": " ".join(test["text"].iloc[i].split()[:45]),
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "error_hard_examples.csv", index=False)
    summary["best_model"] = best
    summary["best_model_confident_errors"] = int(len(confident_wrong))


def label_audit(predictions, test, summary, model="Logistic Regression", size=30):
    """Sample confident errors for a manual read. Judgments live in results/error_label_audit.csv."""
    frame = predictions[model]
    confidence = frame[[f"prob_{n}" for n in LABEL_NAMES]].max(axis=1)
    candidates = frame.index[(frame["predicted"] != frame["label"]) & (confidence > 0.9)]
    picked = sorted(np.random.default_rng(1).choice(candidates, size, replace=False))
    sample = pd.DataFrame({
        "test_index": picked,
        "true": [LABEL_NAMES[frame["label"][i]] for i in picked],
        "predicted": [LABEL_NAMES[frame["predicted"][i]] for i in picked],
        "confidence": [round(float(confidence[i]), 3) for i in picked],
        "opening": [" ".join(test["text"].iloc[i].split()[:70]) for i in picked],
    })
    sample.to_csv(RESULTS_DIR / "error_label_audit_sample.csv", index=False)

    judged_path = RESULTS_DIR / "error_label_audit.csv"
    if judged_path.exists():
        judged = pd.read_csv(judged_path)
        counts = judged["judgment"].value_counts().to_dict()
        audit = {"model": model, "candidates": int(len(candidates)), "sampled": int(len(judged)),
                 "label_wrong": int(counts.get("label_wrong", 0)), "ambiguous": int(counts.get("ambiguous", 0)),
                 "model_wrong": int(counts.get("model_wrong", 0))}
        with open(RESULTS_DIR / "error_label_audit.json", "w") as f:
            json.dump(audit, f, indent=2)
        summary["label_audit"] = audit


def byline_shortcut(summary):
    """How much of the topic can be guessed from the reporter's name alone?"""
    train, _, test = load_splits()
    train_names = train["text"].map(lambda t: (m := BYLINE_NAME.match(t)) and m.group(1).upper())
    test_names = test["text"].map(lambda t: (m := BYLINE_NAME.match(t)) and m.group(1).upper())
    frame = train.assign(name=train_names).dropna(subset=["name"])
    counts = frame["name"].value_counts()
    regular = counts[counts >= 15].index
    usual_topic = frame[frame["name"].isin(regular)].groupby("name")["label_name"].agg(lambda s: s.value_counts().index[0])
    known = test.assign(name=test_names)
    known = known[known["name"].isin(regular)]
    summary["byline_shortcut"] = {
        "train_share_with_byline": round(float(train_names.notna().mean()), 4),
        "regular_reporters": int(len(regular)),
        "test_articles_by_regular_reporters": int(len(known)),
        "accuracy_of_reporter_usual_topic_rule": round(float((known["name"].map(usual_topic) == known["label_name"]).mean()), 4),
    }
    return counts[counts >= 3].index.tolist()


def byline_ablation(names, summary):
    """Retrain the final Logistic Regression with reporter names deleted from every article."""
    with open(RESULTS_DIR / "logistic_regression_metrics.json") as f:
        original = json.load(f)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n.lower()) for n in sorted(names, key=len, reverse=True)) + r")\b")
    train, validation, test = load_splits()
    train, validation, test, _ = clean_splits(train, validation, test)
    for frame in [train, validation, test]:
        frame["clean_text"] = frame["clean_text"].map(lambda t: pattern.sub(" ", t))
    x_train, x_val, x_test = build_features(train, validation, test, word_ngrams=(1, 2), use_chars=True)
    log = pd.read_csv(RESULTS_DIR / "experiment_log.csv").set_index("experiment_id")
    c = json.loads(log.loc["L05", "details"])["model"].split("C=")[1].split(",")[0]
    model = LogisticRegression(C=float(c), max_iter=3000, class_weight="balanced").fit(x_train, train["label"])
    result = {
        "names_removed": len(names),
        "val_macro_f1": compute_metrics(validation["label"], model.predict_proba(x_val))["macro_f1"],
        "test_macro_f1": compute_metrics(test["label"], model.predict_proba(x_test))["macro_f1"],
        "original_val_macro_f1": original["validation"]["macro_f1"],
        "original_test_macro_f1": original["test"]["macro_f1"],
    }
    summary["byline_ablation"] = result
    print("byline ablation", result)


def bilstm_attention(test, summary, examples=6):
    """The words the BiLSTM attends to most, to check whether it relies on sensible cues."""
    path = RESULTS_DIR / "bilstm_model.pt"
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu")
    config = saved["config"]
    if config["pooling"] != "attention":
        return
    words = saved["words"]
    index = {w: i for i, w in enumerate(words)}
    model = BiLSTMClassifier(len(words), len(LABEL_NAMES), hidden_size=config["hidden_size"],
                             dropout=config["dropout"], pooling="attention")
    model.load_state_dict(saved["state_dict"])
    model.eval()

    rows = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(test), examples * 4, replace=False):
        tokens = test["clean_text"].iloc[i].split()[:config["max_length"]]
        ids = torch.tensor([[index.get(w, UNK) for w in tokens]])
        with torch.no_grad():
            logits, weights = model(ids, return_attention=True)
        weights = weights[0].numpy()
        top = np.argsort(weights)[::-1][:8]
        predicted = int(logits.argmax())
        rows.append({"test_index": int(i), "true": LABEL_NAMES[test["label"].iloc[i]],
                     "predicted": LABEL_NAMES[predicted],
                     "top_words": ", ".join(f"{tokens[j]} ({weights[j]:.2f})" for j in top if ids[0, j] != PAD),
                     "attention_on_top8": round(float(weights[top].sum()), 3)})
        if len(rows) >= examples:
            break
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "error_bilstm_attention.csv", index=False)
    summary["bilstm_attention_examples"] = rows


def main():
    plot_style.apply_style()
    predictions = load_predictions()
    train, test = load_test_articles()
    for frame in predictions.values():
        assert (frame["label"].to_numpy() == test["label"].to_numpy()).all(), "prediction rows out of order"

    summary = {"models": [MODELS[k] for k in MODELS if MODELS[k] in predictions]}
    top_confusions(predictions, summary)
    by_length(predictions, test, summary)
    by_rare_words(predictions, train, test, summary)
    by_code_switching(predictions, test, summary)
    hard_examples(predictions, test, summary)
    label_audit(predictions, test, summary)
    byline_ablation(byline_shortcut(summary), summary)
    bilstm_attention(test, summary)
    with open(RESULTS_DIR / "error_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "bilstm_attention_examples"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
