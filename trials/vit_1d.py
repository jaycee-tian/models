import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, heads, dim_feedforward, dropout, activation):
        super().__init__()
        self.attention = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.activation(x)
        x = self.norm2(x)
        x = self.ff(x) + x
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_encoder_layers=6, dim_feedforward=3072, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_encoder_layers)
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ViT(nn.Module):
    def __init__(self, feature_size, patch_size, in_channels, dim, num_classes):
        super().__init__()

        patch_dim = patch_size * in_channels
        patch_num = feature_size // patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (pn ps) -> b pn (ps c)", ps=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.position_embedding = nn.Parameter(torch.randn(patch_num + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b = x.size(0)
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x = x + self.position_embedding
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, "b * d")
        x = self.fc(cls_tokens)
        return x


b = 2
c = 3
f = 256
data = torch.randn(b, c, f)
model = ViT(feature_size=f, patch_size=32, in_channels=c, dim=768, num_classes=40)
output = model(data)
print(output.shape)
