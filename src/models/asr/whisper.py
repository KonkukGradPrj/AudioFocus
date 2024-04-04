from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

import whisper
from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio
from whisper.decoding import *      

from src.models.asr.base import BaseASR

# https://github.com/openai/whisper/tree/main/whisper

class Whisper(DecodingTask, BaseASR):
    name = 'whisper'
    def __init__(self, opt='tiny.en'):
        BaseASR.__init__(self)
        
        model = whisper.load_model(opt)
        options = whisper.DecodingOptions(language="en")
        DecodingTask.__init__(self, model, options)
        
        self.asr_encoder = self.model.encoder
        self.asr_decoder = self.model.decoder

        for param in self.asr_encoder.parameters():
            param.requires_grad = False

        for param in self.asr_decoder.parameters():
            param.requires_grad = False

    def encode(self, wav):
        mel = []
        for w in wav:
            w = whisper.pad_or_trim(w.flatten())
            m = whisper.log_mel_spectrogram(w)
            mel.append(m)
        mel = torch.stack(mel).squeeze().to(wav.device)
        return self.asr_encoder(mel)
    
    def transcribe(self, emb):
        """
        output: texts
        """
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = emb.shape[0]
        audio_features: Tensor = emb  
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [[t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]
        
        return texts
    