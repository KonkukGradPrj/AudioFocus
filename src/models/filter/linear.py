import torch
import torch.nn as nn

from src.models.filter.base import BaseFilter

class LinearFilter(BaseFilter):
    name = 'linear'
    def __init__(self, in_dim=192, hidden_dim=288, out_dim=384):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.ReLU())
        
        self.shift = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.ReLU())

    def forward(self, emb, feat):
        # repeat to fit the shape. (N, emb_dim) => (N, seq_len, emb_dim)
        alpha = self.scale(feat).unsqueeze(1).repeat(1, emb.shape[1], 1)
        beta = self.shift(feat).unsqueeze(1).repeat(1, emb.shape[1], 1)
        return alpha * emb + beta