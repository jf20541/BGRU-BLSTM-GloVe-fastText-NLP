import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
    ):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False
        self.lstm = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim * 4, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        h0, _ = self.lstm(x)
        avg_pool = torch.mean(h0, 1)
        max_pool, _ = torch.max(h0, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.out(out)
        return self.sigmoid(out)
