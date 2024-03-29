import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


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
        self.position_embedding = nn.Parameter(torch.randn(patch_num+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        # self.transformer = Transformer()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b = x.size(0)
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([self.cls_token, x], '* d')
        x = x + self.position_embedding
        # x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, 'b * d')
        x = self.fc(x)
        return x


b = 2
c = 3
f = 256
data = torch.randn(b, c, f)
model = ViT(feature_size=f, patch_size=32, in_channels=c, dim=768, num_classes=40)
output = model(data)
print(output.shape)
