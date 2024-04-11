import torch
import torchaudio
import json
import os
import random
import tqdm
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset

class Libri2Mix(Dataset):
    def __init__(self, root_dir="/home/shu/Desktop/Yongjin/gradproj/model/AudioFocus/data/dataset/", train=True, download=False):
        """
        Initialize the dataset by downloading LibriSpeech and preparing or loading the metadata.
        """
        self.root_dir = root_dir
        self.url = 'train-clean-100' if train else 'test-clean'
        self.dataset = LIBRISPEECH(root=self.root_dir, url=self.url, download=download)
        self.root_dir = os.path.join(self.root_dir, "LibriSpeech/" + self.url)
        self.metadata_path = os.path.join(self.root_dir, "metadata.json")
        
        self.noise_multiplier = 0.9

        if not os.path.exists(self.metadata_path):
            self._generate_metadata()
        else:
            self._load_metadata()

    def _generate_metadata(self):
        """
        Generate metadata for each item in the dataset and save it to a file.
        """
        metadata = []
        for idx in tqdm.tqdm(range(len(self.dataset))):
            _, _, _, speaker_id, chapter_id, _ = self.dataset[idx]
            additional_sample_path = self._find_additional_sample_path(str(speaker_id), str(chapter_id))
            # Assuming noise_idx calculation and other steps are similar to __getitem__
            # We save the index rather than actual path to keep the metadata light
            noise_idx = (idx + 1) % len(self.dataset)
            while self.dataset[noise_idx][3] == speaker_id:
                noise_idx = (noise_idx + 1) % len(self.dataset)
            
            metadata_item = {
                "target_idx": idx,
                "noise_idx": noise_idx,
                "additional_sample_path": additional_sample_path
            }
            metadata.append(metadata_item)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        self.metadata = metadata

    def _load_metadata(self):
        """
        Load metadata from the previously saved file.
        """
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

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
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata_item = self.metadata[idx]
        target_waveform, sample_rate, utterance, _, _, _ = self.dataset[metadata_item['target_idx']]
        noise_waveform, _, _, _, _, _ = self.dataset[metadata_item['noise_idx']]
        
        # Process the noise waveform
        noise_waveform = noise_waveform * self.noise_multiplier

        # Adjust waveforms to be of the same length by creating new tensors
        if target_waveform.shape[1] > noise_waveform.shape[1]:
            pad_length = target_waveform.shape[1] - noise_waveform.shape[1]
            noise_waveform_padded = torch.nn.functional.pad(noise_waveform, (0, pad_length))
        else:
            noise_waveform_padded = noise_waveform[:, :target_waveform.shape[1]]

        # Mix target and noise voice
        mixed_waveform = target_waveform + noise_waveform_padded

        # Load the additional sample waveform
        additional_sample_path = metadata_item['additional_sample_path']
        additional_sample_waveform, _ = torchaudio.load(additional_sample_path)

        return mixed_waveform, target_waveform, utterance, additional_sample_waveform