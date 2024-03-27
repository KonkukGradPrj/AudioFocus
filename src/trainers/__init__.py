from .base import BaseTrainer
from dotmap import DotMap
from omegaconf import OmegaConf


def build_trainer(cfg,
                  device,
                  train_loader,
                  test_loader,
                  logger,
                  model,
                  vanilla):
    
    OmegaConf.resolve(cfg)
    cfg = DotMap(OmegaConf.to_container(cfg))

    return BaseTrainer(cfg=cfg,
                       device=device,
                       train_loader=train_loader,
                       test_loader=test_loader,
                       logger=logger,
                       model=model,
                       vanilla=vanilla)