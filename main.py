
#优化代码
from model.model import lncLSTA
from train_step import model_step
import dataset
import Config
import os
from utils import *
from torch.utils.data import DataLoader,Subset
import matplotlib
from MetricTracker import MetricTracker
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():

    args = Config.parse_args() 
    data=dataset.RNAdataset(args.RNA_data)
    num_label=data.num_label
    k=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    index=0
    nowtime=time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    path=os.path.join(args.save,nowtime)
    os.mkdir(path, mode=0o775 )
    logger = init_logger()
    model=lncLSTA()
    logger.info(model)
    record=pd.DataFrame(data=None,columns=['miAUC','maAUC','acc','f1'])
    record.to_csv(path+'/record.csv',index=None)
    record.to_csv('result/record.csv',index=None)
    
    for train_index, test_index in k.split(data,num_label):
        model_path = os.path.join(path, 'model.pth%s'%index)
        train_dataset, test_dataset = Subset(data,train_index),Subset(data,test_index)
        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
        test_loader=DataLoader(test_dataset,batch_size=args.test_batch_size,collate_fn=collate_fn)
        metricTracker = MetricTracker()
        model=lncLSTA()
        classifier=model_step(path,model,logger)
        trainer = pl.Trainer(gpus =1,  
                            max_epochs=args.epochs,
                            callbacks=[metricTracker,
                            EarlyStopping(monitor="loss", mode="max", patience=60),
                            ModelCheckpoint(save_top_k=1, monitor="loss", mode="max", save_on_train_epoch_end=False)])#训练
        
        trainer.fit(classifier,train_loader,test_loader)
        trainer.test(dataloaders=test_loader)
        model = classifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,path=path,model=model,logger=logger)#加载最好的模型
        torch.save(model.model.state_dict(),model_path)
        index+=1
    

if __name__ == "__main__":
    main()

