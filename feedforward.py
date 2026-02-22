import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)
