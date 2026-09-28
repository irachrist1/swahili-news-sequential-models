"""Turn cleaned articles into padded sequences of word ids for the BiLSTM and TextCNN."""

import gzip
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import EMBEDDINGS_DIR, FASTTEXT_URL

PAD, UNK = 0, 1


class Vocabulary:
    def __init__(self, texts, min_count=2, max_size=50000):
        counts = Counter(word for text in texts for word in text.split())
        common = [w for w, c in counts.most_common(max_size) if c >= min_count]
        self.words = ["<pad>", "<unk>"] + common
        self.index = {word: i for i, word in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def encode(self, text, max_length):
        return [self.index.get(word, UNK) for word in text.split()[:max_length]] or [UNK]


class LengthBucketSampler:
    """Groups articles of similar length into the same batch so little padding is needed.

    Packed sequences are about 4x slower on Apple and CPU backends, so short padding is the cheaper fix.
    """

    def __init__(self, lengths, batch_size, shuffle, seed=0):
        self.order = np.argsort(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = [self.order[i:i + self.batch_size] for i in range(0, len(self.order), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter([batch.tolist() for batch in batches])

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size


def pad_batch(batch):
    longest = max(len(ids) for ids, _ in batch)
    # Round up to a multiple of 32 so the GPU backend sees few distinct shapes and reuses memory.
    longest = ((longest + 31) // 32) * 32
    ids = torch.full((len(batch), longest), PAD, dtype=torch.long)
    for row, (sequence, _) in enumerate(batch):
        ids[row, :len(sequence)] = torch.tensor(sequence)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return ids, labels


def make_loader(frame, vocabulary, max_length, batch_size, shuffle):
    """Batches come out in length order; loader.batch_sampler.order maps them back to frame rows."""
    sequences = [vocabulary.encode(t, max_length) for t in frame["clean_text"]]
    items = list(zip(sequences, frame["label"].tolist()))
    sampler = LengthBucketSampler([len(s) for s in sequences], batch_size, shuffle)
    return DataLoader(items, batch_sampler=sampler, collate_fn=pad_batch)


def download_fasttext():
    import urllib.request

    path = EMBEDDINGS_DIR / "cc.sw.300.vec.gz"
    if not path.exists():
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading fastText Swahili vectors (about 230 MB)...")
        urllib.request.urlretrieve(FASTTEXT_URL, path)
    return path


def load_fasttext_matrix(vocabulary, dimension=300):
    """Embedding matrix for our vocabulary. Words missing from fastText keep small random vectors."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.1, (len(vocabulary), dimension)).astype(np.float32)
    matrix[PAD] = 0
    found = set()
    with gzip.open(download_fasttext(), "rt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            word, rest = line.split(" ", 1)
            index = vocabulary.index.get(word.lower())
            # fastText lists frequent forms first, so keep the first vector seen for a lowercased word.
            if index is not None and index > UNK and index not in found:
                matrix[index] = np.asarray(rest.split(), dtype=np.float32)
                found.add(index)
    coverage = len(found) / (len(vocabulary) - 2)
    print(f"fastText covers {coverage:.1%} of the vocabulary")
    return torch.tensor(matrix), coverage
