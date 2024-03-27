import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, asr_model, filter, speaker_model):
        super().__init__()

        self.asr_model = asr_model
        self.filter = filter
        self.speaker_model = speaker_model
        
    def forward(self, mixed_voice, target_voice):
        """
        input: 
            mixed_voice: mixed voice
            target_voice: sample voice of target voice

        output: 
            vector
        """
        with torch.no_grad():
            feat = self.speaker_model.extract_feature(target_voice)        
            emb = self.asr_model.encode(mixed_voice)
            
        emb = self.filter(emb, feat)
        return self.asr_model.decode(emb)

    def test(self, mixed_voice, target_voice): 
        """
        return text, vector
        """
        with torch.no_grad():
            feat = self.speaker_model.extract_feature(target_voice)
            emb = self.asr_model.encode(mixed_voice)
            emb = self.filter(emb, feat)
            
            text, vec = self.asr_model.run(emb)
            return text, vec