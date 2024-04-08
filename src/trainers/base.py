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

        self.wer = load("wer") # https://huggingface.co/learn/audio-course/en/chapter5/evaluation
        self.base_wer = None
        self.epoch = 0
        self.step = 0

        
    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.filter.parameters(), **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer, **scheduler_cfg)

    def train(self):
        cfg = self.cfg
        test_every = cfg.test_every
        num_epochs = cfg.epochs
        
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        test_logs = {}
        test_logs['epoch'] = self.epoch
        
        self.model.eval()
        test_logs.update(self.test())

        self.logger.update_log(**test_logs)
        self.logger.log_to_wandb(self.step)

        temperature_scale = cfg.temperature ** 2
        for _ in range(int(num_epochs)):                
            # train        
            self.model.train()
            for mixed_voices, clean_voices, _, target_voices in tqdm.tqdm(self.train_loader):   
                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                with torch.no_grad():
                    target = self.vanilla(clean_voices)

                # distill loss
                soft_targets = nn.functional.softmax(target / cfg.temperature, dim=-1)
                student_logits = self.model(mixed_voices, target_voices)
                soft_prob = nn.functional.log_softmax(student_logits / cfg.temperature, dim=-1)
                
                distill_loss =  torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * temperature_scale                

                # MSE loss
                mse_loss = F.mse_loss(student_logits, target)

                # MAE Loss
                l1_loss = F.l1_loss(student_logits, target)
                
                # backward
                loss = distill_loss + mse_loss + l1_loss
                
                scaler.scale(loss).backward()
                
                ################
                #  gradient norm
                gradient_norms = {name: torch.norm(param.grad).item() for name, param in self.model.filter.named_parameters() \
                                if param.grad is not None}
                gradients = []
                for _, norm in gradient_norms.items():
                    gradients.append(norm)  
                avg_gradients = sum(gradients) / len(gradients) if gradients else 0



                # with torch.no_grad():
                #     predict_text = self.model.transcribe(mixed_voices, target_voices)
                #     vanllia_text = self.vanilla.transcribe(mixed_voices)
                # wer = self.wer.compute(references=target_text, predictions=predict_text)
                # vanilla_wer = self.wer.compute(references=target_text, predictions=vanllia_text)
                # print("vanilla: ", vanllia_text[0])
                # print("predict:", predict_text[0])
                # print("target:",target_text[0])
                # print("train_wer: ", wer)
                # print("vanilla_wer: ", vanilla_wer)
                # print("loss: ", loss)

                # gradient clipping
                # unscales the gradients of optimizer's assigned params in-place
                
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.filter.parameters(), cfg.clip_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                # log        
                train_logs = {}                
                train_logs['train_loss'] = loss.item()
                train_logs['distill_loss'] = distill_loss.item()
                train_logs['mse_loss'] = mse_loss.item()
                train_logs['l1_loss'] = l1_loss.item()
                train_logs['lr'] = self.lr_scheduler.get_lr()[0]

                # train_logs['train_wer'] = wer
                # train_logs['vanilla_wer'] = vanilla_wer

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
            self.model.eval()
            test_logs.update(self.test())
            self.logger.update_log(**test_logs)
            self.logger.log_to_wandb(self.step)

    def test(self):
        wer = 0.0
        base_wer = 0.0
        
        N = 0
        for mixed_voices, clean_voices, target_text, target_voices in tqdm.tqdm(self.test_loader):
            with torch.no_grad():     
                n = len(target_text)

                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                predict_text = self.model.transcribe(mixed_voices, target_voices)
                wer += self.wer.compute(references=target_text, predictions=predict_text) * n
                
                # calculate baseline in the first time.
                if self.base_wer is None:
                    baseline_text = self.vanilla.transcribe(mixed_voices)
                    base_wer += self.wer.compute(references=target_text, predictions=baseline_text) * n
                
                N += n
                
        wer /= N
        if self.base_wer is None:
            self.base_wer = base_wer / N

        test_logs = {'test_wer': wer, 'base_wer': self.base_wer}
        print(test_logs)
        return test_logs