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
    
    