import torch
import torch.nn as nn

from src.models.filter.base import BaseFilter

class LinearFilter(BaseFilter):
    name = 'linear'
    def __init__(self, in_dim=192, hidden_dim=288, out_dim=384):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.LayerNorm(out_dim),
                                   nn.ReLU())
        
        self.shift = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.LayerNorm(out_dim),
                                   nn.ReLU())

        # Apply the custom weight and bias initialization with a uniform range
        self.scale.apply(self._init_weights_gaus)
        self.shift.apply(self._init_weights_gaus)

    def _init_weights_gaus(self, m):
        """
        Initialize weights and biases uniformly in the range [-1e-4, 1e-4].
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, emb, feat):
        # repeat to fit the shape. (N, emb_dim) => (N, seq_len, emb_dim)
        alpha = self.scale(feat).unsqueeze(1).repeat(1, emb.shape[1], 1)
        beta = self.shift(feat).unsqueeze(1).repeat(1, emb.shape[1], 1)

        # sanitiy check
        # print(emb -(alpha * emb + beta + emb))
        
        # residual connection
        emb = alpha * emb + beta + emb
        return (emb)