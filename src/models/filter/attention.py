import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from einops import rearrange
from einops.layers.torch import Rearrange
from whisper.model import sinusoids
from src.models.filter.base import BaseFilter


class CrossAttention(nn.Module):
    def __init__(self, feat_dim=192, emb_dim=384, seq_len=1500, heads=6, alpha=4, dim_head=96):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(feat_dim, heads * dim_head  // alpha, bias=False)
        self.to_k = nn.Linear(emb_dim, heads * dim_head  // alpha, bias=False)
        self.to_v = nn.Linear(emb_dim, heads * dim_head  // alpha, bias=False)
        self.to_out = nn.Linear(heads * dim_head  // alpha, emb_dim // alpha, bias=False)

        self.scales = nn.Parameter(torch.randn(seq_len, 1) * 0.01 + 1.0)  # Initialize scale close to 1 with small random noise
        self.biases = nn.Parameter(torch.zeros(seq_len, 1))  # Initialize bias to 0

        self._init_weights_uniform(self.to_v)
        self._init_weights_uniform(self.to_out)

        self.register_buffer("positional_embedding", sinusoids(seq_len, feat_dim))

        
    def _init_weights_uniform(self, m):
        """
        Initialize weights and biases normal distribution in the range with 1e-4 std.
        """
        if isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight, mean=0.0, std=0.0001)
            nn.init.uniform_(m.weight, a=-0.001, b=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, emb, feat):
        # Expand feat to match emb dimensions
        if feat.dim() == 2:
            # feat = feat.unsqueeze(1).expand(-1, emb.size(1), -1)    # batch emb => batch seq_len emb
            # feat =(feat + self.positional_embedding).to(feat.dtype)  
            feat = feat.unsqueeze(1) * self.scales + self.biases  # Apply learnable scale and bias
            feat = feat.expand(-1, emb.size(1), -1)  # Match sequence length of emb      
        
        q = rearrange(self.to_q(feat), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(emb), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(emb), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale # b, h, n, d
        attn = F.softmax(dots, dim=-1)

        # Compute the output from the attention and values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, feat_dim=192, emb_dim=384, seq_len=1500, n_head=3, alpha=4, drop_prob=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = nn.LayerNorm(emb_dim // alpha)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        self.cross_attention = CrossAttention(feat_dim, emb_dim, seq_len=seq_len, heads=n_head, alpha=alpha, dim_head=emb_dim // n_head)
        self.ffn = nn.Sequential(nn.Linear(emb_dim // alpha, emb_dim // alpha, bias=False), nn.ReLU(),
                                 nn.Linear(emb_dim // alpha, emb_dim, bias=False), nn.ReLU(), nn.Dropout(p=drop_prob))
        
    def forward(self, emb, feat):
        emb = self.cross_attention(emb, feat)

        emb = self.dropout1(emb)
        emb = self.norm1(emb)

        emb = self.ffn(emb)
      
        emb = self.dropout2(emb)
        emb = self.norm2(emb)
        return emb


class AttentionFilter(BaseFilter):
    name = 'attention'
    def __init__(self, feat_dim=192, emb_dim=384, seq_len=1500, n_head=3, alpha=4, drop_prob=0.1):
        super().__init__()

        self.dropout0 = nn.Dropout(p=drop_prob)
        self.ln0 = nn.LayerNorm(emb_dim)
        
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(feat_dim, emb_dim, seq_len, n_head, alpha, drop_prob),
            CrossAttentionBlock(feat_dim, emb_dim, seq_len, n_head, alpha, drop_prob),
            CrossAttentionBlock(feat_dim, emb_dim, seq_len, n_head, alpha, drop_prob),
            CrossAttentionBlock(feat_dim, emb_dim, seq_len, n_head, alpha, drop_prob),
        ])

    def forward(self, emb, feat, idx=-1):
        """
        Forward pass with optional block selection.
        
        Args:
            emb (Tensor): The embedding tensor. (N, seq_len, emb_dim)
            feat (Tensor): Feature tensor. (N, seq_len, emb_dim)
            idx (int, optional): Block index to run. Defaults to -1, which runs all blocks.

        Returns:
            Tensor: The output tensor after processing.
        """
        _emb  = emb

        # Add positional encoding to the embedding
        # emb = (emb + self.positional_embedding).to(emb.dtype)        
        if idx == -1:
            for block in self.blocks:
                emb = block(emb, feat)
        else:
            emb = self.blocks[idx](emb, feat)
        
        # residual conn
        emb = emb + _emb
        return emb