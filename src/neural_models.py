"""Approach 3: a bidirectional LSTM with attention."""

import torch
from torch import nn

from sequence_data import PAD


class BiLSTMClassifier(nn.Module):
    """Reads the article left-to-right and right-to-left, so each word state sees its whole context.

    pooling="last" uses the final hidden states; pooling="attention" learns a weight for every word.
    """

    def __init__(self, vocab_size, num_classes, embedding_dim=300, hidden_size=128,
                 dropout=0.3, pooling="attention", embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, ids, return_attention=False):
        mask = ids != PAD
        states, _ = self.lstm(self.dropout(self.embedding(ids)))

        weights = None
        if self.pooling == "attention":
            scores = self.attention(states).squeeze(-1).masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            summary = (weights.unsqueeze(-1) * states).sum(dim=1)
        else:
            # Forward state at the last real word, backward state at the first word.
            half = states.size(2) // 2
            last_index = (mask.sum(dim=1) - 1).clamp(min=0)
            forward_last = states[torch.arange(ids.size(0)), last_index, :half]
            summary = torch.cat([forward_last, states[:, 0, half:]], dim=1)

        logits = self.output(self.dropout(summary))
        return (logits, weights) if return_attention else logits

