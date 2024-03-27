from .asr import *  
from .filter import *
from .speaker import *
from .base import Model
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses


ASR = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseASR)}

FILTER = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseFilter)}

SPEAKER = {subclass.get_name():subclass
         for subclass in all_subclasses(BaseSpeaker)}


def build_model(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    asr_cfg = cfg['asr']
    filter_cfg = cfg['filter']
    speaker_cfg = cfg['speaker']
    
    asr_type = asr_cfg.pop('type')
    filter_type = filter_cfg.pop('type')
    speaker_type = speaker_cfg.pop('type')

    # asr
    asr_cls = ASR[asr_type]
    asr_model = asr_cls(**asr_cfg)

    # filter
    filter_cls = FILTER[filter_type]
    filter = filter_cls(**filter_cfg)

    # speaker
    speaker_cls = SPEAKER[speaker_type]
    speaker_model = speaker_cls(**speaker_cfg)

    # model
    model = Model(asr_model=asr_model, filter=filter, speaker_model=speaker_model)
    
    return model