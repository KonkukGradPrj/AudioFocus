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
from src.common.train_utils import L1MSELoss, SNRLoss, TriSRLoss


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
        self.baseline_text = None
        self.epoch = 0
        self.step = 0
        
        if cfg.loss == 'l2':
            self.loss_fn = nn.MSELoss()                    
        elif cfg.loss == 'l1':
            self.loss_fn = nn.L1Loss()
        elif cfg.loss == 'snr':
            self.loss_fn = SNRLoss()
        elif cfg.loss == 'tri':
            self.loss_fn = TriSRLoss(beta=cfg.beta)
        else:
            self.loss_fn = L1MSELoss()

    def _build_optimizer(self, optimizer_type, optimizer_cfg):
        if optimizer_type == 'adamw':
            return optim.AdamW(self.model.filter.parameters(), **optimizer_cfg)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.filter.parameters(), lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, scheduler_cfg):
        return CosineAnnealingWarmupRestarts(optimizer=optimizer, **scheduler_cfg)

    def train(self):
        cfg = self.cfg
        num_epochs = cfg.epochs
        
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        test_logs = {}
        # test_logs['epoch'] = self.epoch
        
        # self.model.eval()
        # test_logs.update(self.test())

        # self.logger.update_log(**test_logs)
        self.logger.log_to_wandb(self.step)
        
        for _ in range(int(num_epochs)):                
            # train        
            self.model.train()
            for mixed_voices, clean_voices, _, target_voices in tqdm.tqdm(self.train_loader):   
                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                
                with torch.no_grad():
                    target, target_emb_list = self.vanilla(clean_voices, filter_every=cfg.filter_every)
                    init_predict, init_emb_list = self.vanilla(mixed_voices, filter_every=cfg.filter_every)
                
                predict, predict_emb_list = self.model(mixed_voices, target_voices, cfg.filter_every)
                
                train_logs = {}             
                loss = 0
                if cfg.loss == 'tri':
                    if cfg.filter_every:
                        for idx, (predict, init, target) in enumerate(zip(predict_emb_list, init_emb_list, target_emb_list)):                    
                            layer_loss, layer_pos_dist, layer_neg_dist = self.loss_fn(predict, init, target)
                            loss += layer_loss
                            train_logs[f'loss_{idx}'], train_logs[f'pos_dist_{idx}'], train_logs[f'neg_dist_{idx}'] = layer_loss, layer_pos_dist, layer_neg_dist
                    else:
                        loss, _, _ = self.loss_fn(predict, init_predict, target)

                else:
                    if cfg.filter_every:
                        for idx, (predict, target) in enumerate(zip(predict_emb_list, target_emb_list)):                    
                            layer_loss = self.loss_fn(predict, target)
                            train_logs[f'loss_{idx}'] = layer_loss
                            loss += layer_loss
                    else:
                        loss = self.loss_fn(predict, target)
                
                scaler.scale(loss).backward()
                
                ################
                #  gradient norm
                gradient_norms = {name: torch.norm(param.grad).item() for name, param in self.model.filter.named_parameters() \
                                if param.grad is not None}
                gradients = []
                for _, norm in gradient_norms.items():
                    gradients.append(norm)  
                avg_gradients = sum(gradients) / len(gradients) if gradients else 0

                ################
                #  weight scale
                weight_scale = 0.0
                for param in self.model.filter.parameters():
                    weight_scale += torch.sum(param ** 2)
                weight_scale = torch.sqrt(weight_scale)
                
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.filter.parameters(), cfg.clip_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                ################
                # log        
                train_logs['train_loss'] = loss.item()
                train_logs['lr'] = self.lr_scheduler.get_lr()[0]
                train_logs['avg_gradients'] = avg_gradients
                train_logs['weight_scale'] = weight_scale

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

    # on testing, losses is computed only on the last layer
    def test(self):
        test_wer = 0.0
        base_wer = 0.0
        oracle_wer = 0.0

        N = 0
        for mixed_voices, clean_voices, target_text, target_voices in tqdm.tqdm(self.test_loader):
            with torch.no_grad():     
                n = len(target_text)

                mixed_voices, clean_voices, target_voices = mixed_voices.to(self.device), clean_voices.to(self.device), target_voices.to(self.device)
                predict_emb, predict_text = self.model.transcribe(mixed_voices, target_voices, self.cfg.filter_every)

                test_wer += self.wer.compute(references=target_text, predictions=predict_text) * n

                baseline_emb, baseline_text = self.vanilla.transcribe(mixed_voices)
                oracle_emb, oracle_text = self.vanilla.transcribe(clean_voices)
                
                if self.cfg.loss =='tri':
                    test_loss, _, _ = self.loss_fn(predict_emb, baseline_emb, oracle_emb)
                else:
                    test_loss = self.loss_fn(predict_emb, oracle_emb)

                N += n

                # calculate baseline in the first time.
                if self.base_wer is None:
                    base_wer += self.wer.compute(references=target_text, predictions=baseline_text) * n
                    oracle_wer += self.wer.compute(references=target_text, predictions=oracle_text) * n

                    self.baseline_text = baseline_text[0]
                    if self.cfg.loss == 'tri':
                        self.base_loss, _, _ = self.loss_fn(baseline_emb, baseline_emb, oracle_emb)
                    else:
                        self.base_loss = self.loss_fn(baseline_emb, oracle_emb)
                
        test_wer /= N
        if self.base_wer is None:
            self.base_wer = base_wer / N
            self.oracle_wer = oracle_wer / N

        test_logs = {'test_wer': test_wer,
                     'test_loss': test_loss.item(),
                     'base_wer': self.base_wer, 
                     'base_loss': self.base_loss.item(),
                     'oracle_wer': self.oracle_wer}
        
        print(test_logs)
        print('target:', target_text[0])
        print('vanilla:', self.baseline_text)
        print('predict:', predict_text[0])
        return test_logs