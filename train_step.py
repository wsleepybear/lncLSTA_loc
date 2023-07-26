import logging
import math
from multiprocessing.pool import Pool
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
import utils
import pytorch_lightning as pl
import Config
from model.bilstm_gcn import lncG


class model_step(pl.LightningModule):
    def __init__(self,path,model,logger):
        super(model_step, self).__init__()
        self.model=model
        self.path=path
        self.args = Config.parse_args()
    def forward(self, *args, **kwargs) :
        return super().forward(*args, **kwargs)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        return [optimizer]
        

    def validation_step(self,batch, batch_idx):
        feature=batch[0][0]
        label=torch.tensor(batch[1][0]).to(self.device).long()
        one_label=torch.cat([i.unsqueeze(0) for i in batch[1][1]])

        pred=self.model(feature)
        label=label
        loss_function=utils.FocalCrossEntropyLoss()
        loss=loss_function(pred,label)
        dict = {'loss': loss,
                'label': one_label,
                'pred': pred,
                }
        return dict
    def validation_epoch_end(self, outputs):
        loss = [x['loss'] for x in outputs]
        label = torch.cat([x['label'] for x in outputs])
        pred = torch.cat([x['pred'] for x in outputs])
        epoch_loss = torch.stack(loss).mean()  # Combine losses
        myMetic= utils.Metric(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        miAUC=myMetic.miauc()
        maAUC=myMetic.maauc()
        acc=myMetic.accuracy_multiclass()
        f1=myMetic.fscore_class()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('miAUC', miAUC, prog_bar=True)
        self.log('maAUC', maAUC, prog_bar=True)
        self.log('f1', f1, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        dict = {'val_loss': epoch_loss,
                'miAUC': miAUC,
                'maAUC': maAUC,
                'acc':acc,
                'f1':f1}
        return dict

    def test_step(self,batch, batch_idx):
        feature=batch[0][0]
        label=torch.tensor(batch[1][0]).to(self.device).long()
        one_label=torch.cat([i.unsqueeze(0) for i in batch[1][1]])

        pred=self.model(feature)

        label=label
        loss_function=utils.FocalCrossEntropyLoss()
        loss=loss_function(pred,label)
        dict = {'loss': loss,
                'label': one_label,
                'pred': pred,
                }
        return dict
    def test_epoch_end(self, outputs):
        loss = [x['loss'] for x in outputs]
        label = torch.cat([x['label'] for x in outputs])
        pred = torch.cat([x['pred'] for x in outputs])
        epoch_loss = torch.stack(loss).mean().item()  # Combine losses
        myMetic= utils.Metric(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        miAUC=myMetic.miauc()
        maAUC=myMetic.maauc()
        acc=myMetic.accuracy_multiclass()
        f1=myMetic.fscore_class()
        dict = {'val_loss': epoch_loss,
                'miAUC': miAUC,
                'maAUC': maAUC,
                'acc':acc,
                'f1':f1}
        utils.record(self.path,miAUC,maAUC,acc,f1)
        utils.record('result/',miAUC,maAUC,acc,f1)
        return dict

