import torch
import torch.nn as nn


class BILSTM(nn.Module):
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
        super(BILSTM, self).__init__()
        # embedding layer with vocab_size and num of embedding columns
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # collect of parameter from GloVe
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        # GloVe is pretrained embedding, dont need to train gradients on embedding weight
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
        # pass tokens through embedding layer
        x = self.embedding(x)
        h0, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(h0, 1)
        max_pool, _ = torch.max(h0, 1)
        # concat avg_pool and max_pool
        out = torch.cat((avg_pool, max_pool), 1)
        # dimensionality reduction to 1 and output with sigmoid
        out = self.out(out)
        return self.sigmoid(out)
