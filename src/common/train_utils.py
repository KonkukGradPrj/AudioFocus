import numpy as np
import math
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from einops import rearrange


def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    torch.backends.cudnn.benchmark = True 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

def collate_fn(batch):
    # Find the longest sequence
    max_length = max([item[0].shape[1] for item in batch])
    
    batch_target_waveforms = []
    batch_mixed_waveforms = []
    batch_texts = []
    batch_additional_waveforms = []

    for target_waveform, mixed_waveform, text, additional_waveform in batch:
        # Pad target and mixed waveforms to max_length
        padded_target = torch.nn.functional.pad(target_waveform, (0, max_length - target_waveform.shape[1]))
        padded_mixed = torch.nn.functional.pad(mixed_waveform, (0, max_length - mixed_waveform.shape[1]))
        padded_additional = torch.nn.functional.pad(additional_waveform, (0, max_length - additional_waveform.shape[1]))
        
        batch_target_waveforms.append(padded_target)
        batch_mixed_waveforms.append(padded_mixed)
        batch_additional_waveforms.append(padded_additional)
        batch_texts.append(text)
    
    # Stack all items to create batch tensors
    batch_target_waveforms = torch.stack(batch_target_waveforms).squeeze()
    batch_mixed_waveforms = torch.stack(batch_mixed_waveforms).squeeze()
    batch_additional_waveforms = torch.stack(batch_additional_waveforms).squeeze()
    # Texts don't need padding or stacking, so they can be handled normally
    
    return batch_target_waveforms, batch_mixed_waveforms, batch_texts, batch_additional_waveforms