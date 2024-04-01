import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, dim=1000, ff_dim=2000, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=1000, inner_dim=1500, num_heads=5, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(dim, inner_dim)
        self.proj_k = nn.Linear(dim, inner_dim)
        self.proj_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # b pn d -> b pn ind
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # b pn ind -> b h pn h_ind
        q, k, v = [
            rearrange(p, "b pn (h ind) -> b h pn ind", h=self.num_heads)
            for p in [q, k, v]
        ]
        # b h pn h_ind @ b h h_ind pn -> b h pn pn
        scores = q @ k.transpose(2, 3)
        # b h pn pn
        scores = F.softmax(scores, dim=-1)

        scores = self.dropout(scores)
        # b h pn pn @ b h pn h_ind -> b h pn h_ind
        x = scores @ v
        # b h pn h_ind -> b pn ind
        x = rearrange(x, "b h pn h_ind -> b pn (h h_ind)", h=self.num_heads)
        # b pn ind -> b pn d
        x = self.to_out(x)
        x = self.dropout(x)

        return x


class Transformer(nn.Module):
    def __init__(self, num_blocks=12, dim=1000):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        Attention(),
                        FeedForward(),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.blocks:
            x = self.norm1(x)
            x = x + attn(x)
            x = self.norm2(x)
            x = x + ff(x)
        x = self.norm3(x)
        return x


class ViT_3D(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        patch_size=32,
        time_dim=100,
        patch_time=10,
        in_channels=3,
        dim=1000,
        classes=40,
        dropout=0.1,
    ):
        super().__init__()
        # patch_size
        patch_num = (feature_dim // patch_size) ** 2 * (time_dim // patch_time)
        patch_dim = patch_size * patch_size * patch_time * in_channels
        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b c (tn ts) (pnh psh) (pnw psw) -> b (tn pnh pnw) (ts psh psw c)",
                ts=patch_time,
                psh=patch_size,
                psw=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.transformer = Transformer()
        self.fc = nn.Linear(dim, classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_num + 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # b c f -> b patch_num patch_dim
        # 10 3 256 -> 10, 8, 1000
        x = self.patch_embedding(x)
        # 10 1 1000
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.size(0))
        print(cls_tokens.shape)
        # 10, 8, 1000 -> 10, 8, 1000
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)

        # case 1: mean
        # x = x.mean(dim =1)
        # case 2: cls
        # 10, 1000
        x = x[:, 0]

        x = self.fc(x)

        return x


inp = torch.randn(10, 3, 100, 256, 256)
model = ViT_3D()
out = model(inp)
print(out.shape)
