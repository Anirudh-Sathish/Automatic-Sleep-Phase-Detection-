import os
import json
import argparse
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loss import SupConLoss
from data_loader import EEGDataLoader
from model import MainModel


class OneFoldTrainer:
    def __init__(self,fold):
        self.fold = fold
        
        
        self.tp_cfg = {
        "mode": "pretrain",
        "max_epochs": 10,
        "batch_size": 12,
        "lr": 0.0005,
        "weight_decay": 0.0001,
        "temperature": 0.07,
        "val_period": 50,
        "early_stopping": {
            "mode": "min",
            "patience": 20
        }}

        self.es_cfg = {
            "mode": "min",
            "patience": 20
        }

        # set device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        
        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = SupConLoss(temperature=0.07)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', "Sleep-EDF-2018")
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel()
        # print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        # model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        # print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        dataloader_args = {'batch_size':12, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}
        train_dataset = EEGDataLoader(self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            labels = labels.view(-1).to(self.device)

            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(self.device)
            outputs = self.model(inputs)[0]

            f1, f2 = torch.split(outputs, [labels.size(0), labels.size(0)], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss += self.criterion(features, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), train_loss / (i + 1)))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.train()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)[0]

            features = outputs.unsqueeze(1).repeat(1, 2, 1)
            loss += self.criterion(features, labels)

            eval_loss += loss.item()
            
            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), eval_loss / (i + 1)))

        return eval_loss
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break

def main():
    num_splits = 10
    # for fold in range(1, num_splits + 1):
    #     trainer = OneFoldTrainer(fold)
    #     trainer.run()
    trainer = OneFoldTrainer(1)
    trainer.run()


if __name__ == "__main__":
    main()