import numpy as np
import tqdm
import random
import wandb
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from evaluate import load
from src.common.schedulers import CosineAnnealingWarmupRestarts


class BaseTrainer():
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 test_loader,
                 logger,
                 model,
                 vanilla):
        
        super().__init__()

        self.cfg = cfg  
        self.device = device
        self.logger = logger
        self.model = model.to(self.device)
        self.vanilla = vanilla.to(self.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = self._build_optimizer(cfg.optimizer_type, cfg.optimizer)

        cfg.scheduler.first_cycle_steps = len(train_loader) 
        self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.scheduler)

        self.loss_fn = nn.CrossEntropyLoss()
        self.wer = load("wer") # https://huggingface.co/learn/audio-course/en/chapter5/evaluation
                
        self.epoch = 0
        self.step = 0

        
    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer, **scheduler_cfg)

    def train(self):
        cfg = self.cfg
        test_every = cfg.base_test_every
        num_epochs = cfg.base_epochs
        
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        test_logs = {}
        test_logs['epoch'] = self.epoch
        if test_every != -1:
            self.model.eval()
            test_logs.update(self.test())

        self.logger.update_log(**test_logs)
        self.logger.log_to_wandb(self.step)
        
        for _ in range(int(num_epochs)):                
            # train        
            self.model.train()
            for mixed_voices, clean_voices, _, target_voices in tqdm.tqdm(self.train_loader):   
                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                
                with torch.no_grad():
                    _, target = self.vanilla(clean_voices)
                _, predict = self.model(mixed_voices, target_voices)
                
                loss = self.loss_fn(predict, target)

                # backward
                scaler.scale(loss).backward()

                # gradient clipping
                # unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                # log        
                train_logs = {}                
                train_logs['train_loss'] = loss.item()
                train_logs['lr'] = self.lr_scheduler.get_lr()[0]

                self.logger.update_log(**train_logs)
                if self.step % cfg.log_every == 0:
                    self.logger.log_to_wandb(self.step)

                # proceed
                self.lr_scheduler.step()
                self.step += 1

            self.epoch += 1

            # log evaluation
            test_logs = {}
            test_logs['epoch'] = self.epoch
            if (self.epoch % test_every == 0) and (test_every != -1):
                self.model.eval()
                test_logs.update(self.test())
            self.logger.update_log(**test_logs)
            self.logger.log_to_wandb(self.step)

    def test(self) -> dict: 
        cfg = self.cfg
               
        ####################
        ## performance 
        loss_list, wer_list = [], [] 
        N = 0  
        for mixed_voices, clean_voices, target_text, target_voices in tqdm.tqdm(self.test_loader):   
            with torch.no_grad():     
                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                predict_text, predict_vec = self.model(mixed_voices, target_voices)
                _, target_vec = self.vanilla(clean_voices)
                loss = self.loss_fn(predict_vec, target_vec)
                wer = self.wer.compute(references=target_text, predictions=predict_text)
                
            n = len(target_text)
            loss_list.append(loss * n)
            wer_list.append(wer * n)
            N += n
            break

        loss = torch.sum(torch.stack(loss_list)).item() / N
        wer = torch.sum(torch.tensor(wer_list)).item() / N
        
        pred_logs = {
            'test_loss': loss,
            'test_wer': wer,
        }

        test_logs = {}
        test_logs.update(pred_logs)

        return test_logs