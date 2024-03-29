import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def split_last(x, num_heads):
    # b s d
    b, s, d = x.size()
    w = d // num_heads
    return x.view(b, s, num_heads, w)


def merge_last(x):
    b, s, h, w = x.size()
    return x.reshape(b, s, -1)


class MHA(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # b s d
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # b s h w -> b h s w
        q, k, v = [split_last(a, self.num_heads).transpose(1, 2) for a in [q, k, v]]
        # b h s w @ b h w s -> b h s s
        scores = q @ k.transpose(-1, -2)
        scores = scores / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        # b h s s @ b h s w -> b h s w -> b s h w
        x = merge_last((scores @ v).transpose(1,2))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, mlp_dim)
        self.layer2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = self.layer2(F.gelu(self.layer1(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mha = MHA(dim, num_heads, dropout)
        self.mlp = MLP(dim, mlp_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.mha(self.norm1(x)))
        x = x + h
        h = self.drop(self.mlp(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, mlp_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# b s d
b = 10
s = 20
d = 100
data = torch.randn(b, s, d)
m = Transformer(d, 200, 4, 0.1, 4)
# m.eval()
o = m(data)
print(o.shape)
