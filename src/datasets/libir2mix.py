import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LIBRISPEECH
# from src.common.train_utils import collate_fn

import os
import random

class Libri2Mix(Dataset):
    def __init__(self, root_dir="/home/hyeons/workspace/AudioFocus/data/dataset", train=True, download=False):
        """
        Initialize the dataset by downloading LibriSpeech and preparing the data.
        """
        if train:
            url = 'train-clean-100'
        else:
            url = 'test-clean'

        self.dataset = LIBRISPEECH(root=root_dir, url=url, download=download)
        self.noise_multiplier = 0.9
        self.root_dir =root_dir + '/LibriSpeech/' + url

    def _find_additional_sample_path(self, speaker_id, chapter_id):
        """
        Find a path for an additional sample for the given speaker, ideally from a different chapter.
        """
        speaker_path = os.path.join(self.root_dir, speaker_id)
        chapters = [name for name in os.listdir(speaker_path) if os.path.isdir(os.path.join(speaker_path, name)) and name != chapter_id]
        if not chapters:
            selected_chapter = chapter_id
        else:
            selected_chapter = random.choice(chapters)
            
        selected_chapter_path = os.path.join(speaker_path, selected_chapter)
        utterances = [file for file in os.listdir(selected_chapter_path) if file.endswith('.flac')]
        selected_utterance = random.choice(utterances)
        return os.path.join(selected_chapter_path, selected_utterance)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target_waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        
        # Find a noise voice
        noise_idx = (idx + 1) % len(self.dataset)
        while self.dataset[noise_idx][3] == speaker_id:
            noise_idx = (noise_idx + 1) % len(self.dataset)
        noise_waveform, _, _, _, _, _ = self.dataset[noise_idx]

        # Process the noise voice
        noise_waveform = noise_waveform * self.noise_multiplier

        # Adjust waveforms to be of the same length by creating new tensors
        if target_waveform.shape[1] > noise_waveform.shape[1]:
            pad_length = target_waveform.shape[1] - noise_waveform.shape[1]
            noise_waveform_padded = torch.nn.functional.pad(noise_waveform, (0, pad_length))
        else:
            noise_waveform_padded = noise_waveform[:, :target_waveform.shape[1]]

        # Mix target and noise voice
        mixed_waveform = target_waveform + noise_waveform_padded

        additional_sample_path = self._find_additional_sample_path(str(speaker_id), str(chapter_id))
        additional_sample_waveform, _ = torchaudio.load(additional_sample_path)
        return mixed_waveform, target_waveform, utterance, additional_sample_waveform


if __name__ == "__main__":
    def collate_fn(batch):
        # Find the longest sequence
        max_length = max([item[0].shape[1] for item in batch])
        
        batch_target_waveforms = []
        batch_mixed_waveforms = []
        batch_texts = []
        batch_additional_waveforms = []

        for target_waveform, mixed_waveform, text, additional_waveform in batch:
            # Pad target and mixed waveforms to max_length
            print(target_waveform.shape)
            padded_target = torch.nn.functional.pad(target_waveform, (0, max_length - target_waveform.shape[1]))
            padded_mixed = torch.nn.functional.pad(mixed_waveform, (0, max_length - mixed_waveform.shape[1]))
            padded_additional = torch.nn.functional.pad(additional_waveform, (0, max_length - additional_waveform.shape[1]))
            
            batch_target_waveforms.append(padded_target)
            batch_mixed_waveforms.append(padded_mixed)
            batch_additional_waveforms.append(padded_additional)
            batch_texts.append(text)
        
        # Stack all items to create batch tensors
        batch_target_waveforms = torch.stack(batch_target_waveforms)
        batch_mixed_waveforms = torch.stack(batch_mixed_waveforms)
        batch_additional_waveforms = torch.stack(batch_additional_waveforms)
        # Texts don't need padding or stacking, so they can be handled normally
        
        return batch_target_waveforms, batch_mixed_waveforms, batch_texts, batch_additional_waveforms
    
    root_dir = "/home/hyeons/workspace/AudioFocus/data/dataset"
    custom_dataset = Libri2Mix(root_dir=root_dir, train=False, download=True)
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)

    for batch_idx, (clean_voices, mixed_voices, texts, additional_voices) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Clean Voices Shape: {clean_voices.shape}")
        print(f"Mixed Voices Shape: {mixed_voices.shape}")
        # If additional_voices is None for some samples in the batch, handle accordingly
        # For simplicity, this example directly prints shapes and assumes additional_voices are always provided
        print(f"Additional Voices Shape: {additional_voices.shape}")
        print(f"Texts: {texts}")
        # Here, you would typically process your batch (e.g., feeding it to a model)
        
        if batch_idx == 1:  # For demonstration, only iterate through two batches
            break