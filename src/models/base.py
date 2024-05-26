import torch
import torch.nn as nn
import whisper
from copy import deepcopy
import ipdb
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, asr_model, filter, speaker_model):
        super().__init__()

        self.asr_model = asr_model
        self.filter = filter
        self.speaker_model = speaker_model
        
        self.init_filter = deepcopy(self.filter)
    

    def whisper_forward(self, input_voice, target_voice=None, filter_every=False):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            embedding_vector
        """
        # mel-specto
        emb = [whisper.log_mel_spectrogram(whisper.pad_or_trim(iv.flatten())) for iv in input_voice]
        emb = torch.stack(emb).squeeze().to(input_voice[0].device)
        
        mid_layer_embeddings = []

        if target_voice is not None: # filtering using target voice
            feat = self.speaker_model.extract_feature(target_voice)        
            if filter_every:
                for idx in range(4): # constant. fix asr model into whisper en.tiny
                    emb = self.asr_model.encode(emb, idx)
                    res_emb = emb

                    filter_emb = self.filter(emb, feat, idx)
                    init_emb = self.init_filter(emb, feat, idx)
                    emb = res_emb + filter_emb - init_emb
                    
                    mid_layer_embeddings.append(emb)
            else:
                emb = self.asr_model.encode(emb)
                res_emb = emb

                filter_emb = self.filter(emb, feat)
                init_emb = self.init_filter(emb, feat)
                emb = res_emb + filter_emb - init_emb
        else:
            if filter_every:
                for idx in range(4): # constant. fix asr model into whisper en.tiny
                    emb = self.asr_model.encode(emb, idx)
                    mid_layer_embeddings.append(emb)
            else:
                emb = self.asr_model.encode(emb)
        
        return emb, mid_layer_embeddings
    
    def hubert_forward(self, input_voice, target_voice=None, filter_every=False):
        emb = self.asr_model.encode(input_voice)
        if target_voice is not None: # filtering using target voice
            feat = self.speaker_model.extract_feature(target_voice)        
            filter_emb = self.filter(emb, feat)
            init_emb = self.init_filter(emb, feat)
            emb = emb + filter_emb - init_emb
            
        return emb, []
    
    def forward(self, input_voice, target_voice=None, filter_every=False, whisper=False):
        if whisper:
            return self.whisper_forward(input_voice, target_voice, filter_every)
        else:
            return self.hubert_forward(input_voice, target_voice)
        
    @torch.no_grad()
    def transcribe(self, input_voice, target_voice=None, filter_every=False):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            text
        """
        emb, mid_layer_embeddings = self.forward(input_voice, target_voice, filter_every)
        transcriptions = []
        for trans in self.asr_model.transcribe(emb):
            transcriptions.append(trans.upper().lstrip())
            
        return emb, mid_layer_embeddings, transcriptions