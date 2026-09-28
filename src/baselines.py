"""Approaches 1 and 2: TF-IDF with Naive Bayes and with Logistic Regression.

Bag-of-words models ignore word order. They are here to measure how much the
sequential models gain from modelling order and context.
"""

import argparse

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from config import set_seed
from data import load_splits
from metrics import (
    compute_metrics,
    figure_path,
    log_experiment,
    plot_confusion_matrix,
    save_metrics,
    save_predictions,
)
from preprocess import clean_splits

OWNER = "Dan"


def build_features(train, validation, test, word_ngrams=(1, 1), use_chars=False):
    word_vectorizer = TfidfVectorizer(ngram_range=word_ngrams, min_df=2, max_features=300000, sublinear_tf=True)
    parts = [word_vectorizer.fit_transform(train["clean_text"])]
    others = [[word_vectorizer.transform(validation["clean_text"])], [word_vectorizer.transform(test["clean_text"])]]
    if use_chars:
        # Character n-grams share information between inflected forms such as "wachezaji" and "mchezaji".
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True
        )
        parts.append(char_vectorizer.fit_transform(train["clean_text"]))
        others[0].append(char_vectorizer.transform(validation["clean_text"]))
        others[1].append(char_vectorizer.transform(test["clean_text"]))
    return hstack(parts).tocsr(), hstack(others[0]).tocsr(), hstack(others[1]).tocsr()


def prepare(mode="full", remove_stopwords=False):
    train, validation, test = load_splits()
    return clean_splits(train, validation, test, mode=mode, remove_stopwords=remove_stopwords)[:3]


def run(experiment_id, model_name, make_model, splits, change, hypothesis, word_ngrams=(1, 1), use_chars=False):
    train, validation, test = splits
    x_train, x_val, x_test = build_features(train, validation, test, word_ngrams, use_chars)
    model = make_model()
    model.fit(x_train, train["label"])
    val_metrics = compute_metrics(validation["label"], model.predict_proba(x_val))
    log_experiment(experiment_id, model_name, OWNER, change, hypothesis, val_metrics,
                   extra={"word_ngrams": list(word_ngrams), "char_ngrams": use_chars, "model": str(model)})
    return model, val_metrics, (x_train, x_val, x_test)


def finish(experiment_id, model_key, model_name, model, features, splits, val_metrics, change):
    """Evaluate the selected configuration on the test set once and save its outputs."""
    _, x_val, x_test = features
    _, validation, test = splits
    val_probabilities = model.predict_proba(x_val)
    test_probabilities = model.predict_proba(x_test)
    test_metrics = compute_metrics(test["label"], test_probabilities)
    log_experiment(experiment_id, model_name, OWNER, change, "Selected configuration, evaluated once on test.",
                   val_metrics, test_metrics, extra={"model": str(model)})
    save_predictions(model_key, "validation", validation, val_probabilities)
    save_predictions(model_key, "test", test, test_probabilities)
    save_metrics(model_key, {"validation": val_metrics, "test": test_metrics})
    plot_confusion_matrix(test["label"], test_probabilities.argmax(axis=1),
                          f"{model_name} (test)", figure_path(f"confusion_{model_key}.png"))


def naive_bayes_experiments():
    raw = prepare(mode="raw")
    run("B01", "Naive Bayes", lambda: MultinomialNB(alpha=0.1), raw,
        "Raw lowercased text, word unigrams",
        "Starting point: how far pure word counts go without any cleaning.")

    clean = prepare()
    run("B02", "Naive Bayes", lambda: MultinomialNB(alpha=0.1), clean,
        "Cleaning: split glued sentences, repair ligatures, drop punctuation and numbers",
        "Glued sentences create fake words like 'huu.misafara'; fixing them should help.")

    no_stopwords = prepare(remove_stopwords=True)
    run("B03", "Naive Bayes", lambda: MultinomialNB(alpha=0.1), no_stopwords,
        "Cleaning + Swahili stopword removal",
        "Function words carry little topic signal, so removing them may reduce noise.")

    run("B04", "Naive Bayes", lambda: MultinomialNB(alpha=0.1), clean,
        "Word unigrams + bigrams",
        "Bigrams such as 'ligi kuu' or 'benki kuu' capture short local order.", word_ngrams=(1, 2))

    run("B05", "Naive Bayes", lambda: ComplementNB(alpha=0.3), clean,
        "ComplementNB (designed for imbalanced text), unigrams + bigrams",
        "Multinomial NB favours the majority class 'kitaifa'; ComplementNB corrects this.", word_ngrams=(1, 2))

    best = None
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0]:
        model, val_metrics, features = run(
            f"B06-a{alpha}", "Naive Bayes", lambda: ComplementNB(alpha=alpha), clean,
            f"ComplementNB alpha={alpha}", "Tune smoothing on validation macro-F1.", word_ngrams=(1, 2))
        if best is None or val_metrics["macro_f1"] > best[1]["macro_f1"]:
            best = (model, val_metrics, features, alpha)
    model, val_metrics, features, alpha = best
    finish("B07", "naive_bayes", "Naive Bayes", model, features, clean, val_metrics,
           f"Final: ComplementNB alpha={alpha}, cleaned text, unigrams + bigrams")


def logistic_regression_experiments():
    clean = prepare()

    def logistic(c=10, balanced=False):
        return lambda: LogisticRegression(C=c, max_iter=3000, class_weight="balanced" if balanced else None)

    run("L01", "Logistic Regression", logistic(), clean,
        "Cleaned text, word unigrams + bigrams, no class weights",
        "A discriminative linear model should beat Naive Bayes on the same features.", word_ngrams=(1, 2))

    run("L02", "Logistic Regression", logistic(balanced=True), clean,
        "Balanced class weights",
        "Weighting errors by inverse class frequency should raise recall on 'afya' and 'uchumi'.",
        word_ngrams=(1, 2))

    run("L03", "Logistic Regression", logistic(balanced=True), clean,
        "Add character 2-5 grams",
        "Swahili is agglutinative; character n-grams share evidence across word forms and misspellings.",
        word_ngrams=(1, 2), use_chars=True)

    best = None
    for c in [1, 3, 10, 30]:
        model, val_metrics, features = run(
            f"L04-C{c}", "Logistic Regression", logistic(c, balanced=True), clean,
            f"C={c} with word + character n-grams", "Tune regularisation on validation macro-F1.",
            word_ngrams=(1, 2), use_chars=True)
        if best is None or val_metrics["macro_f1"] > best[1]["macro_f1"]:
            best = (model, val_metrics, features, c)
    model, val_metrics, features, c = best
    finish("L05", "logistic_regression", "Logistic Regression", model, features, clean, val_metrics,
           f"Final: C={c}, balanced weights, word 1-2 grams + char 2-5 grams")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression", "all"], default="all")
    args = parser.parse_args()
    set_seed()
    if args.model in ["naive_bayes", "all"]:
        naive_bayes_experiments()
    if args.model in ["logistic_regression", "all"]:
        logistic_regression_experiments()
