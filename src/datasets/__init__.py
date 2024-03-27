from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.datasets.aug import apply_audio_transforms
from src.datasets.libir2mix import Libri2Mix
from src.common.train_utils import collate_fn

def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # initialize dataset
    dataset_type = cfg['type']

    if dataset_type in ['Libri2Mix']:
        # train_cfg = cfg['train']
        # test_cfg = cfg['test']
        dataloader_cfg = cfg['dataloader']
        
        # train_transforms = apply_audio_transforms(train_cfg.pop('transforms'))
        # test_transforms = apply_audio_transforms(test_cfg.pop('transforms'))

        train_dataset = Libri2Mix(train=True)
        test_dataset = Libri2Mix(train=False)
    
        train_loader = DataLoader(train_dataset, **dataloader_cfg, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, **dataloader_cfg, shuffle=False, collate_fn=collate_fn)

    else:
        raise NotImplemented
    
    return train_loader, test_loader