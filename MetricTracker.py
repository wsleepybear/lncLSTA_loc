import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
class MetricTracker(Callback):
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.miAUC=[]
        self.maAUC=[]
        self.f1=[]

    def on_validation_epoch_end(self, trainer, module):
        self.loss.append(trainer._results['validation_epoch_end.val_loss'].value.cpu().numpy()) # track them
        self.miAUC.append(trainer._results['validation_epoch_end.miAUC'].value.cpu().numpy()) # track them
        self.maAUC.append(trainer._results['validation_epoch_end.maAUC'].value.cpu().numpy()) # track them
        self.accuracy.append(trainer._results['validation_epoch_end.acc'].value.cpu().numpy()) # track them
        self.f1.append(trainer._results['validation_epoch_end.f1'].value.cpu().numpy()) # track them