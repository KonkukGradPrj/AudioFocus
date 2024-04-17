import torch
import torch.nn as nn
from abc import *


class BaseSpeaker(nn.Module, metaclass=ABCMeta):
    name = 'base'
    def __init__(self):
        super().__init__()
        self.model = None

    @classmethod
    def get_name(cls):
        return cls.name
    