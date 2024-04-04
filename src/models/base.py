import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, asr_model, filter, speaker_model):
        super().__init__()

        self.asr_model = asr_model
        self.filter = filter
        self.speaker_model = speaker_model
    
    @torch.no_grad()
    def transcribe(self, input_voice, target_voice=None):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            text
        """
        if target_voice is not None:
            feat = self.speaker_model.extract_feature(target_voice)        
            emb = self.asr_model.encode(input_voice)
            emb = self.filter(emb, feat)
        else:
            emb = self.asr_model.encode(input_voice)
            
        transcriptions = []

        for trans in self.asr_model.transcribe(emb):
            transcriptions.append(trans.upper().lstrip())
        return transcriptions
    
    def forward(self, input_voice, target_voice=None):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            embedding_vector
        """
        if target_voice is not None:
            feat = self.speaker_model.extract_feature(target_voice)        
            emb = self.asr_model.encode(input_voice)
            emb = self.filter(emb, feat)
        else:
            emb = self.asr_model.encode(input_voice)
        return emb