import torch
import torch.nn as nn

class CodeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
