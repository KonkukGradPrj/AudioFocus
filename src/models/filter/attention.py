import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from src.models.filter.base import BaseFilter


# helpers
def posemb_sincos_1d(patches, temperature=10000, dtype=torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    assert (dim % 2) == 0, 'feature dimension must be a multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = torch.arange(n, device=device).flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


class CrossAttention(nn.Module):
    def __init__(self, feat_dim=192, emb_dim=384, heads=4, dim_head=96):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(feat_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(emb_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(emb_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, emb_dim)

    def forward(self, emb, feat):
        if feat.dim() == 2:
            feat = feat.unsqueeze(1)

        q = rearrange(self.to_q(feat), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(emb), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(emb), 'b n (h d) -> b h n d', h=self.heads)
        # Compute attention scores and apply softmax
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)

        # Compute the output from the attention and values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class AttentionFilter(BaseFilter):
    name = 'attention'
    def __init__(self, feat_dim=192, emb_dim=384, n_head=4, hidden_dim=192):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.cross_attention = CrossAttention(feat_dim, emb_dim, heads=n_head, dim_head=emb_dim // n_head)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, hidden_dim, emb_dim)

    def forward(self, emb, feat):
        pe = posemb_sincos_1d(emb)
        x = rearrange(emb, 'b ... d -> b (...) d') + pe
        
        x = self.ln1(x)
        x = self.cross_attention(x, feat) + x  
        x = self.ln2(x)
        x = self.mlp(x) + x  
        return x