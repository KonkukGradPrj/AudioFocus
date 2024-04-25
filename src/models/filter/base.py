import torch
import torch.nn as nn
from abc import *


class BaseFilter(nn.Module):
    name = 'base'
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name
    
    def forward(self, emb, feat):
        return emb
    
    def _init_weights(self, m):
        """
        Initialize weights and biases in zeros.
        """
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)