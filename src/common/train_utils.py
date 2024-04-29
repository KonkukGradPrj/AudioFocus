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


class L1MSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Initializes the L1MSELoss class.
        
        Parameters:
        alpha (float): Weighting factor for balancing L1 and MSE losses. Defaults to 0.5.
        """
        super(L1MSELoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Forward pass for computing the weighted sum of L1 and MSE losses.
        
        Parameters:
        predictions (Tensor): The predicted outputs.
        targets (Tensor): The ground truth labels.
        
        Returns:
        Tensor: The calculated loss.
        """
        loss_l1 = self.l1_loss(predictions, targets)
        loss_mse = self.mse_loss(predictions, targets)
        return self.alpha * loss_l1 + (1 - self.alpha) * loss_mse
    

class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, predictions, targets):
        # Ensure the input tensors are of floating point type
        targets = targets.float()
        predictions = predictions.float()

        # Calculate the signal power
        signal_power = torch.norm(targets, p=2)**2

        # Calculate the error power
        error_power = torch.norm(targets - predictions, p=2)**2

        # Compute the SNR loss
        snr_loss = 10 * torch.log10(signal_power / error_power)

        return snr_loss
    
# https://arxiv.org/pdf/1911.02411 => quiet strong to normalized errors see https://wandb.ai/hyeonsio/AudioFocus/runs/9czqhfza   
class TriSRLoss(nn.Module):
    def __init__(self, beta=0.3):
        super(TriSRLoss, self).__init__()
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, init_predictions, targets):
        pos_distance = self.l1_loss(predictions, targets)
        neg_distance = self.mse_loss(predictions, init_predictions)

        tri_loss = pos_distance + self.beta * neg_distance
        return tri_loss, pos_distance, neg_distance