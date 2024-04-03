import torch
import torch.nn as nn
from abc import *

class BaseASR(nn.Module, metaclass=ABCMeta):
    name='base'
    def __init__(self):
        super().__init__()
    
    @classmethod
    def get_name(cls):
        return cls.name
    
    @abstractmethod
    def encode(self, wav):
        pass
    
    
    @abstractmethod
    def transcribe(self, emb):
        # return text
        pass