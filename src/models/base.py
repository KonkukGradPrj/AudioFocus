import torch
import torch.nn as nn
import whisper

class Model(nn.Module):
    def __init__(self, asr_model, filter, speaker_model):
        super().__init__()

        self.asr_model = asr_model
        self.filter = filter
        self.speaker_model = speaker_model
    
    @torch.no_grad()
    def transcribe(self, input_voice, target_voice=None, filter_every=False):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            text
        """
        emb = self.forward(input_voice, target_voice, filter_every)
        transcriptions = []

        for trans in self.asr_model.transcribe(emb):
            transcriptions.append(trans.upper().lstrip())
        return transcriptions
    

    def forward(self, input_voice, target_voice=None, filter_every=False):
        """
        input: 
            mixed_voice: input voice. mixed voce if target voice is not none else target voice.
            target_voice: sample voice of target voice

        output: 
            embedding_vector
        """
        # mel-specto
        mel = [whisper.log_mel_spectrogram(whisper.pad_or_trim(w.flatten())) for w in input_voice]
        mel = torch.stack(mel).squeeze().to(input_voice[0].device)

        if target_voice is not None: # filtering using target voice
            feat = self.speaker_model.extract_feature(target_voice)        
            if filter_every:
                emb = mel
                for idx in range(4): # constant. fix asr model into whisper en.tiny
                    emb = self.asr_model.encode(emb, idx)
                    emb = self.filter(emb, feat, idx)
            else:
                emb = self.asr_model.encode(mel)
                emb = self.filter(emb, feat)
        else:
            emb = self.asr_model.encode(mel)
        return emb