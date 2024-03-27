import torch
import torch.nn as nn

from src.models.filter.base import BaseFilter

class MLPFilter(BaseFilter):
    name = 'mlp'
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.w1 = nn.Sequential(nn.Linear(512, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU())
        
        self.b1 = nn.Sequential(nn.Linear(512, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU())

        self.w2 = nn.Sequential(nn.Linear(512, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU())
        
        self.b2 = nn.Sequential(nn.Linear(512, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU())

    def forward(self, emb, feat):
        emb = self.w1(feat) * emb + self.b1(emb)
        emb = self.w2(feat) * emb + self.b2(emb)

        return emb