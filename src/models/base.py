import torch
import torch.nn as nn
import whisper
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, asr_model, filter, speaker_model):
        super().__init__()

        self.asr_model = asr_model
        self.filter = filter
        self.speaker_model = speaker_model
        
        self.init_filter = deepcopy(self.filter)
        for param in self.init_filter.parameters():
            param.requires_grad = False
    

    def forward(self, input_voice, target_voice=None, filter_every=False):
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
        
        old_emb = emb
        mid_layer_embeddings = []
        old_mid_layer_embeddings = []

        if target_voice is not None: # filtering using target voice
            feat = self.speaker_model.extract_feature(target_voice)        
            if filter_every:
                for idx in range(4): # constant. fix asr model into whisper en.tiny
                    emb = self.asr_model.encode(emb, idx)
                    filter_emb = self.filter(emb, feat, idx)
                    init_emb = self.init_filter(emb, feat, idx)
                    emb = emb + filter_emb - init_emb
                    mid_layer_embeddings.append(emb)
            else:
                emb = self.asr_model.encode(emb)
                filter_emb = self.filter(emb, feat)
                init_emb = self.init_filter(emb, feat)
                emb = emb + filter_emb - init_emb
        
        else:
            if filter_every:
                for idx in range(4): # constant. fix asr model into whisper en.tiny
                    emb = self.asr_model.encode(emb, idx)
                    mid_layer_embeddings.append(emb)
            else:
                emb = self.asr_model.encode(emb)

        return emb, mid_layer_embeddings, old_emb, old_mid_layer_embeddings
    
        
    @torch.no_grad()
    def transcribe(self, input_voice, target_voice=None, filter_every=False):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            text
        """
        emb, _, old_emb, _ = self.forward(input_voice, target_voice, filter_every)
        transcriptions = []
        for trans in self.asr_model.transcribe(emb):
            transcriptions.append(trans.upper().lstrip())
            
        return emb, old_emb, transcriptions