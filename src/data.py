"""Download the Swahili news corpus and build the train, validation and test splits."""

import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    DATASET_URL,
    LABEL_NAMES,
    SEED,
    VALIDATION_FRACTION,
    make_output_dirs,
)


def download_raw_data():
    make_output_dirs()
    for split in ["train", "test"]:
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"Downloading {split} split...")
            urllib.request.urlretrieve(DATASET_URL.format(split=split), path)


def load_raw_data():
    download_raw_data()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    return train, test


def remove_bad_rows(train, test):
    """Drop empty texts, duplicates, and test articles that also appear in train."""
    report = {"raw_train": len(train), "raw_test": len(test)}

    train = train[train["text"].str.split().str.len() >= 20]
    test = test[test["text"].str.split().str.len() >= 20]
    report["short_train_removed"] = report["raw_train"] - len(train)
    report["short_test_removed"] = report["raw_test"] - len(test)

    train = train.drop_duplicates(subset="text")
    test = test.drop_duplicates(subset="text")

    before = len(test)
    test = test[~test["text"].isin(set(train["text"]))]
    report["test_overlap_removed"] = before - len(test)

    report["clean_train"] = len(train)
    report["clean_test"] = len(test)
    return train.reset_index(drop=True), test.reset_index(drop=True), report


def load_splits():
    """Return train, validation and test DataFrames with columns text, label, label_name."""
    train, test = load_raw_data()
    train, test, _ = remove_bad_rows(train, test)

    train, validation = train_test_split(
        train,
        test_size=VALIDATION_FRACTION,
        stratify=train["label"],
        random_state=SEED,
    )

    splits = []
    for frame in [train, validation, test]:
        frame = frame.reset_index(drop=True).copy()
        frame["label_name"] = frame["label"].map(lambda i: LABEL_NAMES[i])
        splits.append(frame)
    return splits[0], splits[1], splits[2]


if __name__ == "__main__":
    train_raw, test_raw = load_raw_data()
    _, _, cleaning_report = remove_bad_rows(train_raw, test_raw)
    print(cleaning_report)
    train_df, val_df, test_df = load_splits()
    print(f"train={len(train_df)} validation={len(val_df)} test={len(test_df)}")
