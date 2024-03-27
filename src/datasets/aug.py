import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

def add_background_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def time_stretch(waveform, stretch_factor_min=0.8, stretch_factor_max=1.25):
    stretch_factor = np.random.uniform(stretch_factor_min, stretch_factor_max)
    return T.TimeStretch(n_freq=201, fixed_rate=stretch_factor)(waveform)

def pitch_shift(waveform, sample_rate, n_steps=2.5):
    n_steps = np.random.uniform(-n_steps, n_steps)
    return T.PitchShift(sample_rate, n_steps)(waveform)

# Function to apply transformations based on a config dict
def apply_audio_transforms(waveform, cfg):
    if cfg.get('background_noise', False):
        waveform = add_background_noise(waveform, cfg['noise_level'])
    if cfg.get('time_stretch', False):
        waveform = time_stretch(waveform)
    if cfg.get('pitch_shift', False):
        waveform = pitch_shift(waveform, cfg['sample_rate'])
    
    # SpecAugment
    if cfg.get('specaugment', False):
        waveform = T.FrequencyMasking(freq_mask_param=cfg['freq_mask_param'])(waveform)
        waveform = T.TimeMasking(time_mask_param=cfg['time_mask_param'])(waveform)

    return waveform