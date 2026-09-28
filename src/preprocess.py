"""Text cleaning for Swahili news articles.

The corpus was scraped from news websites and has two systematic problems:
1. Sentences are glued together ("mwaka huu.Misafara").
2. The "fi"/"fl" ligature was broken during extraction ("Ofi si" instead of "Ofisi").
"""

import re
from collections import Counter

# Short list of very frequent Swahili function words, used only in the stopword experiment.
SWAHILI_STOPWORDS = {
    "na", "ya", "wa", "kwa", "la", "za", "katika", "ni", "kuwa", "hiyo", "hilo",
    "huo", "hizo", "hao", "cha", "vya", "pia", "hii", "huu", "hizi", "kama", "kwamba",
    "ili", "au", "lakini", "bado", "sana", "tu", "hata", "kutoka", "zaidi", "baada",
    "kabla", "hadi", "mimi", "yeye", "wao", "sisi", "wewe", "nyinyi", "amesema",
    "alisema", "ambao", "ambayo", "ambapo", "ambalo", "kila", "moja", "mwaka",
}

GLUED_SENTENCE = re.compile(r"([a-z0-9%\)])([.!?,;:])([A-Z“\"])")
LIGATURE_PAIR = re.compile(r"\b(\w*f[il]) ([a-z]\w*)")
URL = re.compile(r"https?://\S+|www\.\S+")
NUMBER = re.compile(r"\d+([.,]\d+)*")
NON_WORD = re.compile(r"[^a-z<>\s]")
SPACES = re.compile(r"\s+")


def split_glued_sentences(text):
    return GLUED_SENTENCE.sub(r"\1\2 \3", text)


def learn_ligature_repairs(texts, min_ratio=2, min_count=5):
    """Find "Xfi Y" pairs whose joined form "XfiY" is much more common in the corpus.

    Real words such as "safi na" (clean and) stay split because "safina" is rare.
    """
    lowered = " ".join(t.lower() for t in texts)
    word_counts = Counter(re.findall(r"\w+", lowered))
    pair_counts = Counter(LIGATURE_PAIR.findall(lowered))

    repairs = {}
    for (left, right), pair_count in pair_counts.items():
        joined_count = word_counts[left + right]
        if joined_count >= min_count and joined_count >= min_ratio * pair_count:
            repairs[f"{left} {right}"] = left + right
    return repairs


def repair_ligatures(text, repairs):
    def replace(match):
        pair = f"{match.group(1)} {match.group(2)}"
        joined = repairs.get(pair.lower())
        return match.group(1) + match.group(2) if joined else pair

    return LIGATURE_PAIR.sub(replace, text)


def light_clean(text, repairs):
    """Keep casing and punctuation. Used for the transformer, whose tokenizer handles both."""
    text = text.replace(" ", " ")
    text = split_glued_sentences(text)
    text = repair_ligatures(text, repairs)
    text = URL.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def full_clean(text, repairs, remove_stopwords=False):
    """Lowercase word-level cleaning for the TF-IDF, BiLSTM and CNN models."""
    text = light_clean(text, repairs).lower()
    text = NUMBER.sub(" <num> ", text)
    text = NON_WORD.sub(" ", text)
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in SWAHILI_STOPWORDS]
    return " ".join(words)


def raw_clean(text):
    """Minimum needed to tokenise on whitespace. Used as the "no preprocessing" control."""
    return SPACES.sub(" ", text.lower()).strip()


def clean_splits(train, validation, test, mode="full", remove_stopwords=False):
    """Add a clean_text column to every split. Ligature repairs are learned on train only."""
    repairs = learn_ligature_repairs(train["text"])
    for frame in [train, validation, test]:
        if mode == "raw":
            frame["clean_text"] = frame["text"].map(raw_clean)
        elif mode == "light":
            frame["clean_text"] = frame["text"].map(lambda t: light_clean(t, repairs))
        else:
            frame["clean_text"] = frame["text"].map(
                lambda t: full_clean(t, repairs, remove_stopwords)
            )
    return train, validation, test, repairs
