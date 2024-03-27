import torch
import nemo.collections.asr as nemo_asr
from src.models.speaker.base import BaseSpeaker

# https://huggingface.co/nvidia/speakerverification_en_titanet_large
# speaker_model.get_embedding("wav")

class TitaNet(BaseSpeaker):
    name = 'titanet'
    def __init__(self, opt="nvidia/speakerverification_en_titanet_large"):
        super(TitaNet, self).__init__()
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(opt)
    
    @torch.no_grad
    def extract_feature(self, wav):
        features = []
        for w in wav:
            w = w.squeeze().cpu()
            feat, _ = self.model.infer_segment(w)
            features.append(feat)
        features = torch.stack(features).squeeze().to(wav.device)
        return features