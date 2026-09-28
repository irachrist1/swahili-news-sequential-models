"""Exploratory analysis. Saves figures to figures/ and numbers to results/eda_summary.json."""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import plot_style
from config import FIGURES_DIR, LABEL_ENGLISH, LABEL_NAMES, RESULTS_DIR, make_output_dirs
from data import load_raw_data, load_splits, remove_bad_rows
from preprocess import GLUED_SENTENCE, LIGATURE_PAIR, clean_splits, learn_ligature_repairs

plt = plot_style.plt
ENGLISH_WORDS = {w for w in ENGLISH_STOP_WORDS if len(w) > 2}
BYLINE = re.compile(r"^\s*Na\s+[A-Z][A-Z]+")


def label_axis(names):
    return [f"{name}\n({LABEL_ENGLISH[name]})" for name in names]


def plot_class_distribution(train, validation, test, summary):
    counts = pd.DataFrame(
        {
            "train": train["label_name"].value_counts(),
            "validation": validation["label_name"].value_counts(),
            "test": test["label_name"].value_counts(),
        }
    ).loc[LABEL_NAMES]
    summary["class_counts"] = counts.to_dict()
    shares = counts["train"] / counts["train"].sum()
    summary["train_class_share"] = shares.round(4).to_dict()
    summary["imbalance_ratio"] = round(counts["train"].max() / counts["train"].min(), 2)

    order = shares.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    bars = ax.bar(label_axis(order), shares[order] * 100, color=plot_style.MAIN_COLOR, width=0.6)
    for bar, name in zip(bars, order):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{counts.loc[name, 'train']:,}",
            ha="center",
            fontsize=9,
            color=plot_style.TEXT_SECONDARY,
        )
    ax.set_ylabel("share of training articles (%)")
    ax.set_title(f"Class distribution (largest/smallest = {summary['imbalance_ratio']}x)")
    ax.grid(axis="x", visible=False)
    plot_style.save(fig, FIGURES_DIR / "eda_class_distribution.png")


def plot_lengths(train, summary):
    lengths = train["text"].str.split().str.len()
    summary["length_words"] = {
        "mean": round(lengths.mean(), 1),
        "median": int(lengths.median()),
        "p90": int(lengths.quantile(0.9)),
        "p95": int(lengths.quantile(0.95)),
        "max": int(lengths.max()),
        "share_over_256": round((lengths > 256).mean(), 4),
        "share_over_512": round((lengths > 512).mean(), 4),
    }
    by_class = train.assign(length=lengths).groupby("label_name")["length"].median()
    summary["median_length_by_class"] = by_class.astype(int).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].hist(lengths.clip(upper=1500), bins=60, color=plot_style.MAIN_COLOR)
    for cut, height in [(256, 0.9), (512, 0.6)]:
        axes[0].axvline(cut, color=plot_style.TEXT_SECONDARY, linestyle="--", linewidth=1)
        share = (lengths > cut).mean() * 100
        axes[0].text(cut + 15, axes[0].get_ylim()[1] * height, f"{share:.0f}% longer\nthan {cut}", fontsize=8)
    axes[0].set_xlabel("words per article (clipped at 1,500)")
    axes[0].set_ylabel("articles")
    axes[0].set_title("Article length")

    order = by_class.sort_values().index
    data = [lengths[train["label_name"] == name].clip(upper=1500) for name in order]
    axes[1].boxplot(
        data,
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        medianprops={"color": plot_style.SERIES_COLORS[1], "linewidth": 2},
        boxprops={"color": plot_style.MAIN_COLOR},
        whiskerprops={"color": plot_style.MAIN_COLOR},
        capprops={"color": plot_style.MAIN_COLOR},
    )
    axes[1].set_yticks(range(1, len(order) + 1), [f"{n} ({LABEL_ENGLISH[n]})" for n in order])
    axes[1].set_xlabel("words per article")
    axes[1].set_title("Length by class")
    plot_style.save(fig, FIGURES_DIR / "eda_lengths.png")


def plot_vocabulary(train, validation, test, summary):
    train_counts = Counter(w for t in train["clean_text"] for w in t.split())
    frequencies = np.array(sorted(train_counts.values(), reverse=True))
    total = frequencies.sum()
    summary["vocabulary"] = {
        "train_tokens": int(total),
        "unique_words": len(train_counts),
        "hapax_share_of_vocab": round(np.mean(frequencies == 1), 4),
    }

    coverage = {}
    for min_count in [1, 2, 3, 5, 10]:
        kept = frequencies[frequencies >= min_count]
        coverage[min_count] = {"vocab_size": int(len(kept)), "token_coverage": round(kept.sum() / total, 4)}
    summary["vocabulary"]["coverage_by_min_count"] = coverage

    for name, frame in [("validation", validation), ("test", test)]:
        words = [w for t in frame["clean_text"] for w in t.split()]
        oov = np.mean([train_counts.get(w, 0) < 2 for w in words])
        summary["vocabulary"][f"{name}_oov_rate_min_count_2"] = round(float(oov), 4)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ranks = np.arange(1, len(frequencies) + 1)
    axes[0].loglog(ranks, frequencies, color=plot_style.MAIN_COLOR)
    axes[0].set_xlabel("word rank")
    axes[0].set_ylabel("frequency")
    axes[0].set_title("Zipf plot of the training vocabulary")

    cumulative = np.cumsum(frequencies) / total
    axes[1].semilogx(ranks, cumulative * 100, color=plot_style.MAIN_COLOR)
    for size, height in [(20000, 60), (50000, 40)]:
        axes[1].axvline(size, color=plot_style.TEXT_SECONDARY, linestyle="--", linewidth=1)
        axes[1].text(size * 1.1, height, f"{size // 1000}k words\n{cumulative[size - 1] * 100:.1f}%", fontsize=8)
    axes[1].set_xlabel("vocabulary size (most frequent words kept)")
    axes[1].set_ylabel("token coverage (%)")
    axes[1].set_title("How much text a vocabulary covers")
    plot_style.save(fig, FIGURES_DIR / "eda_vocabulary.png")


def measure_noise(train, summary):
    texts = train["text"]
    repairs = learn_ligature_repairs(texts)
    pairs = Counter(m for t in texts.str.lower() for m in LIGATURE_PAIR.findall(t))
    repaired = sum(n for (a, b), n in pairs.items() if f"{a} {b}" in repairs)
    summary["noise"] = {
        "articles_with_glued_sentences": round(texts.map(lambda t: bool(GLUED_SENTENCE.search(t))).mean(), 4),
        "glued_sentences_per_article": round(texts.map(lambda t: len(GLUED_SENTENCE.findall(t))).mean(), 2),
        "ligature_split_candidates": int(sum(pairs.values())),
        "ligature_splits_repaired": int(repaired),
        "ligature_repair_rules": len(repairs),
        "articles_with_urls": int(texts.str.contains(r"https?://|www\.").sum()),
        "articles_starting_with_byline": round(texts.str.match(BYLINE).mean(), 4),
    }


def plot_code_switching(train, summary):
    def english_share(text):
        words = text.split()
        return sum(w in ENGLISH_WORDS for w in words) / max(len(words), 1)

    share = train["clean_text"].map(english_share)
    by_class = train.assign(share=share).groupby("label_name")["share"]
    summary["code_switching"] = {
        "articles_with_any_english_function_word": round((share > 0).mean(), 4),
        "articles_over_1_percent": round((share > 0.01).mean(), 4),
        "share_over_1_percent_by_class": by_class.apply(lambda s: round((s > 0.01).mean(), 4)).to_dict(),
    }

    values = pd.Series(summary["code_switching"]["share_over_1_percent_by_class"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(label_axis(values.index), values * 100, color=plot_style.MAIN_COLOR, height=0.55)
    ax.set_xlabel("articles where >1% of words are English function words (%)")
    ax.set_title("Code-switching with English by class")
    ax.grid(axis="y", visible=False)
    plot_style.save(fig, FIGURES_DIR / "eda_code_switching.png")


def top_words_per_class(train, summary, top_n=8):
    vectorizer = TfidfVectorizer(min_df=5, max_features=30000)
    features = vectorizer.fit_transform(train["clean_text"])
    vocabulary = np.array(vectorizer.get_feature_names_out())
    rows = {}
    for label_id, name in enumerate(LABEL_NAMES):
        in_class = (train["label"] == label_id).to_numpy()
        scores, _ = chi2(features, in_class)
        # chi2 is unsigned, so drop words that are more common outside the class.
        positive = np.asarray(features[in_class].mean(axis=0) > features[~in_class].mean(axis=0)).ravel()
        scores = np.where(positive, scores, 0)
        rows[name] = vocabulary[np.argsort(scores)[::-1][:top_n]].tolist()
    summary["top_words_per_class"] = rows
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "eda_top_words.csv", index=False)


def keep_part(text, k, part):
    words = text.split()
    return " ".join(words[:k] if part == "head" else words[-k:])


def positional_information(train, validation, summary):
    """Macro-F1 of a linear model that only sees the first or last k words.

    This tells us where in an article the topic signal sits, which guides truncation.
    """
    results = []
    for part in ["head", "tail"]:
        for k in [16, 32, 64, 128, 256, 512]:
            vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
            x_train = vectorizer.fit_transform(train["clean_text"].map(lambda t: keep_part(t, k, part)))
            x_val = vectorizer.transform(validation["clean_text"].map(lambda t: keep_part(t, k, part)))
            model = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
            model.fit(x_train, train["label"])
            score = f1_score(validation["label"], model.predict(x_val), average="macro")
            results.append({"part": part, "words": k, "macro_f1": round(score, 4)})
            print(part, k, round(score, 4))
    summary["positional_information"] = results

    frame = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for color, (part, label) in zip(plot_style.SERIES_COLORS, [("head", "first k words"), ("tail", "last k words")]):
        rows = frame[frame["part"] == part]
        ax.plot(rows["words"], rows["macro_f1"], marker="o", markersize=5, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256, 512], ["16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("words kept (k)")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Where the topic signal sits in an article")
    ax.legend()
    plot_style.save(fig, FIGURES_DIR / "eda_positional_information.png")


def subword_fertility(train, summary, sample_size=1000):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_small")
    sample = train.sample(sample_size, random_state=0)
    words = sample["text"].str.split().str.len()
    tokens = sample["text"].map(lambda t: len(tokenizer.tokenize(t)))
    summary["afriberta_tokens"] = {
        "tokens_per_word": round((tokens / words).mean(), 3),
        "median_tokens": int(tokens.median()),
        "share_over_512_tokens": round((tokens > 510).mean(), 4),
        "share_over_256_tokens": round((tokens > 254).mean(), 4),
    }


def main():
    make_output_dirs()
    plot_style.apply_style()
    summary = {}

    train_raw, test_raw = load_raw_data()
    _, _, summary["cleaning"] = remove_bad_rows(train_raw, test_raw)

    train, validation, test = load_splits()
    train, validation, test, _ = clean_splits(train, validation, test)

    plot_class_distribution(train, validation, test, summary)
    plot_lengths(train, summary)
    plot_vocabulary(train, validation, test, summary)
    measure_noise(train, summary)
    plot_code_switching(train, summary)
    top_words_per_class(train, summary)
    positional_information(train, validation, summary)
    subword_fertility(train, summary)

    with open(RESULTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
