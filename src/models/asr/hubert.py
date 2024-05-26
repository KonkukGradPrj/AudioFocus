from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
import tensorflow as tf

import whisper
from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio
from transformers import AutoProcessor, TFHubertForCTC
import torchaudio
from src.models.asr.base import BaseASR
from transformers import (
    SpeechEncoderDecoderModel,
    Speech2Text2Processor,
)

class Hubert(BaseASR):
    name = 'hubert'
    def __init__(self):
        BaseASR.__init__(self)
        self.model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de").cuda()
        self.processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
    
    # (batch, )
    def encode(self, input_voice):
        emb = []
        for iv in input_voice:
            inp = self.processor(iv, sampling_rate=16_000, return_tensors="pt").to('cuda:0')
            e =self.model.generate(inputs=inp["input_values"], attention_mask=inp["attention_mask"])
            emb.append(e)
        emb = torch.stack([whisper.pad_or_trim(e.flatten()) for e in emb])
        return emb
    
    
    def transcribe(self, emb):
        """
        output: texts
        """
        text  = self.processor.batch_decode(emb)
        return text
